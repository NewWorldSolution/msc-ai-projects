"""Regression checks for the tool corrections; only in-memory resources are used."""

from contextlib import contextmanager
from datetime import datetime
import json
import sqlite3
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from agentic.tools import customer_tools as customer
from agentic.tools import experience_tools as experience
from agentic.tools import knowledge_tools as knowledge
from agentic.tools import reservation_tools as reservation
from agentic.tools import subscription_tools as subscription
from agentic.tools import ticket_tools as ticket
from agentic.tools.database_errors import is_retryable_database_error
from agentic.tools.handoff_tools import assemble_fallback_handoff
from agentic.tools.memory_tools import save_customer_memory
from data.models import cultpass, udahub
from utils import get_session


@pytest.fixture
def business_engine():
    engine = create_engine("sqlite://")
    cultpass.Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add_all([
            cultpass.User(user_id="c1", full_name="One", email="one@example.com"),
            cultpass.User(user_id="c2", full_name="Two", email="two@example.com"),
            cultpass.Subscription(subscription_id="s1", user_id="c1",
                                  status="active", tier="basic", monthly_quota=3),
        ])
        for number in (1, 2):
            session.add(cultpass.Experience(
                experience_id=f"e{number}", title="Concert", description="Stored",
                location="Warsaw", when=datetime(2026, 9, 15, 10),
                slots_available=5, is_premium=False,
            ))
        session.add_all([
            cultpass.Reservation(reservation_id="r1", user_id="c1",
                                 experience_id="e1", status="confirmed"),
            cultpass.Reservation(reservation_id="r2", user_id="c1",
                                 experience_id="e2", status="confirmed"),
            cultpass.Reservation(reservation_id="other", user_id="c2",
                                 experience_id="e1", status="confirmed"),
        ])
    yield engine
    engine.dispose()


def config(engine, mode="single"):
    return {"configurable": {
        "cultpass_engine": engine, "trusted_customer_id": "c1",
        "turn_id": "turn1", "issue_id": "issue1", "lookup_key": "lookup1",
        "attempt": 1, "lookup_mode": mode,
    }}


def test_customer_result_and_fallback_are_json_serializable(business_engine):
    result = customer.lookup_customer_tool.invoke(
        {"customer_id": "c1"}, config=config(business_engine),
    )
    assert isinstance(result["records"][0]["created_at"], str)
    json.dumps(result)
    handoff = assemble_fallback_handoff({
        "ticket_id": "ticket1", "turn_id": "turn1", "account_id": "account1",
        "customer_id": "c1", "messages": [HumanMessage(content="Help", id="turn1")],
        "tool_results": [result], "account_context": {"user_id": "c1", "is_blocked": True},
    })
    assert handoff["account_context"]["is_blocked"] is True
    assert handoff["customer_id"] == "c1"
    json.dumps(handoff)


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_unique_lookup_success_and_provenance(business_engine, tool, args):
    result = tool.invoke(args, config=config(business_engine))
    assert result["outcome"] == "found"
    assert len(result["records"]) == 1
    assert result["turn_id"] == "turn1"
    assert result["issue_id"] == "issue1"
    assert result["lookup_key"] == "lookup1"
    assert result["attempt"] == 1
    json.dumps(result)


@pytest.mark.parametrize("mode,expected", [("single", "ambiguous"), ("list", "found")])
def test_descriptive_cardinality_and_customer_scope(business_engine, mode, expected):
    experiences = experience.lookup_experiences.invoke(
        {"title": "Concert"}, config=config(business_engine, mode),
    )
    reservations = reservation.lookup_reservations.invoke(
        {"customer_id": "c1", "event_name": "Concert"}, config=config(business_engine, mode),
    )
    assert experiences["outcome"] == reservations["outcome"] == expected
    assert len(experiences["records"]) == len(reservations["records"]) == 2
    assert {row["user_id"] for row in reservations["records"]} == {"c1"}


def test_no_customer_required_for_experience_and_foreign_reservation_hidden(business_engine):
    trusted = config(business_engine)
    trusted["configurable"].pop("trusted_customer_id")
    assert experience.lookup_experiences.invoke(
        {"experience_id": "e1"}, config=trusted,
    )["outcome"] == "found"
    assert reservation.lookup_reservations.invoke(
        {"customer_id": "c1", "reservation_id": "other"}, config=config(business_engine),
    )["outcome"] == "not_found"


@pytest.mark.parametrize("tool,args", [
    (customer.lookup_customer_tool, {"customer_id": "c2"}),
    (subscription.lookup_subscription, {"customer_id": "c2"}),
    (reservation.lookup_reservations, {"customer_id": "c2"}),
])
def test_customer_switch_rejected_without_database_query(business_engine, tool, args):
    statements = []
    # api: before_cursor_execute observes SQL actually sent to the database.
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    result = tool.invoke(args, config=config(business_engine))
    assert result["error_code"] == "customer_scope_mismatch"
    assert statements == []


@pytest.mark.parametrize("tool", [
    subscription.lookup_subscription, reservation.lookup_reservations,
])
def test_omitted_customer_returns_missing_input(business_engine, tool):
    assert tool.invoke({}, config=config(business_engine))["outcome"] == "missing_input"


@pytest.mark.parametrize("tool,args", [
    (reservation.lookup_reservations, {"customer_id": "c1"}),
    (experience.lookup_experiences, {}),
])
@pytest.mark.parametrize("date_text", ["9999-12-31", "2026-02-30", "20260915"])
def test_invalid_date_rejected_before_sql(business_engine, tool, args, date_text):
    statements = []
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    result = tool.invoke({**args, "event_date": date_text}, config=config(business_engine))
    assert result["outcome"] == "missing_input"
    assert statements == []


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_missing_schema_is_nonretryable(tool, args):
    engine = create_engine("sqlite://")
    try:
        result = tool.invoke(args, config=config(engine))
        assert result["outcome"] == "technical_failure"
        assert result["retryable"] is False
    finally:
        engine.dispose()


@pytest.mark.parametrize("module,tool,args", [
    (subscription, subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation, reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience, experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_locked_database_is_retryable(business_engine, module, tool, args):
    @contextmanager
    def locked(_engine):
        raise OperationalError("SELECT", {}, sqlite3.OperationalError("database is locked"))
        yield

    with patch.object(module, "get_session", locked):
        result = tool.invoke(args, config=config(business_engine))
    assert result["outcome"] == "technical_failure"
    assert result["retryable"] is True


def test_unknown_database_error_is_not_transient():
    assert is_retryable_database_error(SQLAlchemyError("timeout in configuration")) is False
    error = OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: timeout"))
    assert is_retryable_database_error(error) is False


def test_bad_memory_type_and_ticket_outcome_are_controlled(business_engine):
    result = save_customer_memory("account1", "c1", {
        "memory_id": "m1", "memory_type": [], "content": "Stored",
        "ticket_id": "ticket1", "turn_id": "turn1", "outcome": "resolved",
        "source_refs": ["article1"],
    }, store=InMemoryStore())
    assert result["error_code"] == "invalid_memory_payload"
    result = ticket.save_ticket_outcome(
        "ticket1", {"turn_id": "turn1", "outcome": []}, business_engine,
    )
    assert result["error_code"] == "invalid_case_outcome"


def test_invalid_ticket_id_performs_no_sql(business_engine):
    statements = []
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    assert ticket.load_ticket_context("", business_engine)["outcome"] == "missing_input"
    assert statements == []


class ScoredStore:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates
        self.error = error

    def similarity_search_with_relevance_scores(self, query, *, k, filter):
        assert filter == {"account_id": "account1"}
        if self.error is not None:
            raise self.error
        return self.candidates


@pytest.fixture
def knowledge_config():
    engine = create_engine("sqlite://")
    udahub.Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add(udahub.Account(account_id="account1", account_name="CultPass"))
        session.add(udahub.Knowledge(article_id="article1", account_id="account1",
                                    title="Policy", content="Stored policy.", tags="policy"))
    yield {"configurable": {
        "udahub_engine": engine, "trusted_account_id": "account1",
        "turn_id": "turn1", "issue_id": "issue1",
        "retrieval_settings": {"threshold": 0.7, "candidate_count": 3,
                               "embedding_model": "test",
                               "score_convention": "higher_is_more_relevant"},
    }}
    engine.dispose()


@pytest.mark.parametrize("account,text,expected", [
    ("account1", "Stored policy.", None),
    ("account2", "Stored policy.", "invalid_source_metadata"),
    ("account1", "Invented policy.", "source_content_mismatch"),
])
def test_knowledge_sources(knowledge_config, account, text, expected):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([
        (Document(page_content=text, metadata={"article_id": "article1", "account_id": account}), 0.9),
    ])
    result = knowledge.search_knowledge("policy", knowledge_config)
    assert result["error_code"] == expected
    if expected is None:
        assert result["status"] == "completed"
        assert result["articles"][0]["content"] == "Stored policy."
        assert result["gate_records"][0]["gate_pass"] is True
    else:
        assert result["status"] == "technical_failure"
        assert result["articles"] == []


def test_empty_and_missing_knowledge_are_distinct(knowledge_config):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([])
    empty = knowledge.search_knowledge("policy", knowledge_config)
    missing = knowledge.search_knowledge("", knowledge_config)
    assert empty["status"] == "completed"
    assert empty["gate_records"][0]["score"] is None
    assert missing["status"] == "missing_input"


@pytest.mark.parametrize("error,retryable", [
    (ValueError("timeout must be positive"), False),
    (TimeoutError("temporary timeout"), True),
])
def test_retrieval_retry_classification(knowledge_config, error, retryable):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore(error=error)
    result = knowledge.search_knowledge("policy", knowledge_config)
    assert result["status"] == "technical_failure"
    assert result["retryable"] is retryable


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c2"}),
    (reservation.lookup_reservations, {"customer_id": "c2", "reservation_id": "absent"}),
    (experience.lookup_experiences, {"experience_id": "absent"}),
])
def test_completed_absence_is_not_a_failure(business_engine, tool, args):
    trusted = config(business_engine)
    trusted["configurable"]["trusted_customer_id"] = "c2"
    result = tool.invoke(args, config=trusted)
    assert result["outcome"] == "not_found"
    assert result["records"] == []
    assert result["retryable"] is False


def test_bad_memory_after_core_success_keeps_the_reply(monkeypatch):
    import agentic.persist as persist
    state = {
        "ticket_id": "ticket1", "turn_id": "turn1", "account_id": "account1",
        "customer_id": "c1", "outcome": "resolved",
        "messages": [HumanMessage(content="Help", id="turn1")],
        "approved_response": {"message_id": "reply1", "draft_id": "draft1",
                              "outcome": "resolved", "content": "Approved reply"},
        "review_result": {"draft_id": "draft1", "verdict": "approved"},
        "memory_updates": [{"memory_id": "m1", "memory_type": [], "content": "Stored",
                            "source_refs": ["article1"]}],
    }
    calls = []

    def saved(*args):
        calls.append(args)
        return {"saved": True, "approved_response": state["approved_response"]}

    monkeypatch.setattr(persist, "save_ticket_outcome", saved)
    result = persist.persist_case(
        state, {"configurable": {"thread_id": "ticket1"}},
        udahub_engine=object(), store=InMemoryStore(),
    )
    assert len(calls) == 1
    assert result["persistence_result"]["saved"] is True
    assert result["persistence_result"]["memory"]["status"] == "failed"
    assert result["messages"][0].content == "Approved reply"