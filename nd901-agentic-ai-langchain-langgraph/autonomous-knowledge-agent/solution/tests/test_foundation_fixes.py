"""Regression tests for the September foundation fixes; no external services."""

from contextlib import contextmanager
import sqlite3
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError

from agentic import intake, persist
from agentic.state import CaseState
from agentic.tools import customer_tools, knowledge_tools, ticket_tools
from data.models import udahub
from utils import get_session
from test_intake import seeded_resources, trusted_config
from test_persist import resolved_state
from test_review_regressions import ScoredStore, knowledge_config
from test_ticket_tools import engine, resolved_outcome


@pytest.fixture
def intake_resources():
    """Dispose both isolated databases after each Intake regression."""
    support, business = seeded_resources()
    yield trusted_config(support, business)
    support.dispose()
    business.dispose()


def run_intake_graph(state, config):
    """Apply Intake's delta through the real CaseState reducers."""
    def prepare(current: CaseState) -> CaseState:
        return intake.prepare_case(current, config, store=InMemoryStore())

    graph = StateGraph(CaseState)
    graph.add_node("prepare", prepare)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", END)
    return graph.compile().invoke(state)


def test_invalid_intake_cannot_reuse_saved_reply(engine, monkeypatch):
    """A failed preparation cannot release an older reply even if Persist runs."""
    state = resolved_state()
    state.update(persistence_result={"saved": True}, messaging_disabled=False)
    state["messages"].append(AIMessage(content="Old reply", id="old-reply"))
    lookup = Mock(side_effect=AssertionError("No lookup permitted"))
    monkeypatch.setattr(intake, "load_ticket_context", lookup)
    config = {"configurable": {"thread_id": "ticket-1"}}
    merged = run_intake_graph(state, config)
    assert lookup.call_count == 0
    assert merged["preparation_result"]["outcome"] == "failed"
    assert merged["approved_response"] is None
    assert merged["persistence_result"] is None
    assert merged["outcome"] is None
    assert merged["handoff"] is None
    assert merged["messaging_disabled"] is True
    result = persist.persist_case(merged, config, udahub_engine=engine, store=InMemoryStore())
    assert result["persistence_result"]["saved"] is False
    assert "messages" not in result
    with get_session(engine) as session:
        assert session.query(udahub.TicketMessage).count() == 0


def test_same_turn_keeps_budgets_and_terminal_state(intake_resources, monkeypatch):
    """Reentry must neither spend another lookup nor renew any allowance."""
    initial = {"messages": [HumanMessage(content="Help", id="turn-1")]}
    merged = run_intake_graph(initial, intake_resources)
    spent = {**merged["technical_attempts"], "operations:lookup": 2}
    controls = {
        "technical_attempts": spent, "workflow_steps": 12, "model_calls": 8,
        "original_review_attempts": 2, "acknowledgement_review_attempts": 2,
        "classification_status": "completed", "classification": {"urgency": "urgent"},
        "messaging_disabled": True, "handoff": {"message_id": "handoff-1"},
        "outcome": "escalated",
    }
    merged.update(controls)
    lookup = Mock(side_effect=AssertionError("Same turn is already prepared"))
    monkeypatch.setattr(intake, "load_ticket_context", lookup)
    second = run_intake_graph(merged, intake_resources)
    assert lookup.call_count == 0
    for field, value in controls.items():
        assert second[field] == value
    assert second["trace"] == merged["trace"]


@pytest.mark.parametrize("boundary", ["load_ticket_context", "lookup_customer"])
@pytest.mark.parametrize("failure_count,retryable,expected_calls", [(1, True, 2), (2, True, 2), (1, False, 1)])
def test_intake_retry_budget(boundary, failure_count, retryable, expected_calls, intake_resources, monkeypatch):
    """Only a transient failure gets a second attempt; reentry cannot add a third."""
    actual = getattr(intake, boundary)
    calls = []

    def failing_then_real(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) <= failure_count:
            return {
                "outcome": "database_error" if boundary == "load_ticket_context" else "technical_failure",
                "records": [], "error_code": "transient_database_error" if retryable else "database_error",
                "reason": "Controlled database failure.", "retryable": retryable,
            }
        return actual(*args, **kwargs)

    monkeypatch.setattr(intake, boundary, failing_then_real)
    state = {"messages": [HumanMessage(content="Help", id="turn-1")]}
    result = run_intake_graph(state, intake_resources)
    assert len(calls) == expected_calls
    if expected_calls == 2:
        assert calls[0] == calls[1]
    expected = "ready" if retryable and failure_count == 1 else "failed"
    assert result["preparation_result"]["outcome"] == expected
    key = "ticket" if boundary == "load_ticket_context" else "customer"
    assert result["preparation_result"]["lookup_attempts"][key] == expected_calls
    assert result["trace"][-1]["payload"]["lookup_attempts"][key] == expected_calls
    repeated = run_intake_graph(result, intake_resources)
    assert len(calls) == expected_calls
    assert repeated["technical_attempts"] == result["technical_attempts"]


@pytest.mark.parametrize("boundary,outcome", [("load_ticket_context", "missing_ticket"), ("lookup_customer", "not_found")])
def test_intake_never_retries_completed_absence(boundary, outcome, intake_resources, monkeypatch):
    """Even an inconsistent retry flag cannot turn absence into a retry loop."""
    lookup = Mock(return_value={"outcome": outcome, "records": [], "retryable": True})
    monkeypatch.setattr(intake, boundary, lookup)
    result = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
    assert lookup.call_count == 1
    assert result["preparation_result"]["outcome"] != "ready"


@pytest.mark.parametrize("boundary", ["load_ticket_context", "lookup_customer"])
def test_intake_unknown_exception_is_sanitized_without_retry(boundary, intake_resources, monkeypatch):
    """Unclassified exceptions must not leak details or receive a retry."""
    lookup = Mock(side_effect=RuntimeError("secret database endpoint"))
    monkeypatch.setattr(intake, boundary, lookup)
    result = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
    assert lookup.call_count == 1
    assert result["preparation_result"]["retryable"] is False
    assert "secret database endpoint" not in str(result)


@pytest.mark.parametrize("score", [-0.02, 1.01, float("nan"), float("inf"), True])
def test_relevance_score_validity_and_gate(knowledge_config, score):
    """Finite raw scores are evidence; non-finite scores are technical failures."""
    document = Document(page_content="Stored policy.", metadata={"article_id": "article1", "account_id": "account1"})
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([(document, score)])
    result = knowledge_tools.search_knowledge("policy", knowledge_config)
    if score in (-0.02, 1.01):
        assert result["status"] == "completed"
        assert result["gate_records"][0]["score"] == score
        assert result["gate_records"][0]["gate_pass"] is (score >= 0.7)
    else:
        assert result["error_code"] == "invalid_relevance_score"
        assert result["retryable"] is False


@pytest.mark.parametrize("boundary", ["customer", "save"])
@pytest.mark.parametrize("failure,expected", [
    (OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: timeout")), False),
    (OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: deadlock")), False),
    (OperationalError("SELECT", {}, sqlite3.OperationalError("database is locked")), True),
    (PoolTimeoutError("pool exhausted"), True),
])
def test_shared_database_retry_policy(boundary, failure, expected, monkeypatch):
    """Database names and generic wording cannot classify permanent errors as transient."""
    @contextmanager
    def failing_session(_engine):
        raise failure
        yield

    module = customer_tools if boundary == "customer" else ticket_tools
    monkeypatch.setattr(module, "get_session", failing_session)
    result = (module.lookup_customer(object(), customer_id="customer-1") if boundary == "customer"
              else module.save_ticket_outcome("ticket-1", resolved_outcome(), object()))
    assert result["retryable"] is expected
    assert "SELECT" not in result["reason"]


@pytest.mark.parametrize("error_kind,expected", [("timeout", True), ("connection", True), ("rate_limit", True), ("authentication", False)])
def test_provider_retry_policy(knowledge_config, error_kind, expected):
    """Classify installed embedding-provider exceptions without invoking the provider."""
    request = httpx.Request("POST", "https://example.invalid/embeddings")
    if error_kind == "timeout":
        error = APITimeoutError(request=request)
    elif error_kind == "connection":
        error = APIConnectionError(request=request)
    else:
        status = 429 if error_kind == "rate_limit" else 401
        cls = RateLimitError if status == 429 else AuthenticationError
        error = cls("private provider error", response=httpx.Response(status, request=request), body=None)
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore(error=error)
    result = knowledge_tools.search_knowledge("policy", knowledge_config)
    assert result["retryable"] is expected
    assert "private provider error" not in str(result)


@pytest.mark.parametrize("outcome", ["resolved", "information_requested"])
def test_escalated_ticket_cannot_be_reopened(engine, outcome):
    """A terminal human handoff is protected before any new support writes."""
    with get_session(engine) as session:
        session.get(udahub.TicketMetadata, "ticket-1").status = "escalated"
    payload = resolved_outcome()
    payload["outcome"] = outcome
    payload["approved_response"]["outcome"] = outcome
    result = ticket_tools.save_ticket_outcome("ticket-1", payload, engine)
    assert result["saved"] is False
    assert result["error_code"] == "ticket_already_escalated"
    assert result["retryable"] is False
    with get_session(engine) as session:
        assert session.get(udahub.TicketMetadata, "ticket-1").status == "escalated"
        assert session.query(udahub.TicketMessage).count() == 0
        assert session.query(udahub.TraceEvent).count() == 0


def test_fallback_handoff_is_returned_after_real_save(engine):
    """The checkpoint must contain the same handoff that the support database saved."""
    state = resolved_state()
    state.update(outcome="escalated", approved_response=None, review_result=None, handoff=None)
    result = persist.persist_case(state, {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
    assert result["persistence_result"]["saved"] is True
    assert result["handoff"]["handoff_source"] == "deterministic_fallback"
    with get_session(engine) as session:
        import json
        saved = session.get(udahub.TicketMessage, result["handoff"]["message_id"])
        assert json.loads(saved.content) == {**result["handoff"], "message_type": "human_handoff"}


def test_new_turn_resets_only_current_turn_allowances(intake_resources):
    """A different stable human ID gets new counters while ticket history survives."""
    state = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
    state.update(original_review_attempts=2, messaging_disabled=True,
                 identity_correction={"asked": True}, technical_attempts={"spent-old-key": 2})
    state["messages"].append(HumanMessage(content="New question", id="turn-2"))
    result = run_intake_graph(state, intake_resources)
    assert result["original_review_attempts"] == 0
    assert result["messaging_disabled"] is False
    assert result["identity_correction"] == {"asked": True}
    assert result["technical_attempts"] == {
        "turn-2:intake:ticket:ticket-1": 1,
        "turn-2:intake:customer:customer-1": 1,
    }
    assert len(result["trace"]) == 2


def test_memory_summary_is_checkpoint_only_after_core_commit(engine):
    """Document the supported audit boundary with real database assertions."""
    result = persist.persist_case(resolved_state(), {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
    assert result["persistence_result"]["saved"] is True
    assert result["persistence_result"]["memory"]["status"] == "succeeded"
    assert result["trace"][-1]["event_type"] == "memory_write_summary"
    with get_session(engine) as session:
        assert session.get(udahub.TraceEvent, "turn-1:persist:outcome") is not None
        assert session.get(udahub.TraceEvent, "turn-1:persist:memory") is None


def test_persist_rejects_unhashable_outcome_without_sql(engine):
    """Malformed outcome values produce a controlled failure instead of TypeError."""
    state = resolved_state()
    state["outcome"] = []
    result = persist.persist_case(state, {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
    assert result["persistence_result"]["error_code"] == "invalid_case_outcome"
    assert result["persistence_result"]["attempts"] == 0
    assert "messages" not in result
