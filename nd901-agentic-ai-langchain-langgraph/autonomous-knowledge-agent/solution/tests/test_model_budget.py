"""Credential-free regressions for shared reservations and budget handoffs."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agentic.agents import classifier, supervisor
from agentic.model_budget import ModelCallBudgetExceeded, build_model_call_guard


class ScriptedModel(FakeMessagesListChatModel):
    """Drive the real ReAct loop without provider calls."""

    invocations: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        self.invocations += 1
        return super()._generate(*args, **kwargs)


@pytest.fixture
def case():
    return {
        "ticket_id": "ticket-1",
        "account_id": "account-1",
        "customer_id": "customer-1",
        "turn_id": "turn-1",
        "messages": [HumanMessage(content="Help me sign in", id="turn-1")],
        "preparation_result": {"outcome": "ready", "ticket_lookup_outcome": "success"},
        "classification_status": "pending",
        "classification": None,
        "account_context": {"is_blocked": False},
        "technical_attempts": {},
        "model_calls": 0,
        "messaging_disabled": False,
        "trace": [{"event_id": "intake"}],
    }


@pytest.fixture
def assessment():
    return classifier.TicketClassification(
        issues=[classifier.IssueAssessment(
            category="account_access", task="Explain login recovery",
            missing_information=[], missing_information_reasons={},
            needs_operations=False, required_business_facts=[],
        )],
        urgency="normal", urgency_reason="No deadline stated", complexity="simple",
    )


def make_model(duplicate=False):
    calls = [{"id": "call-1", "name": "route_to_classifier", "args": {}}]
    if duplicate:
        calls.append({"id": "call-2", "name": "route_to_classifier", "args": {}})
    return ScriptedModel(responses=[
        AIMessage(content="", tool_calls=calls), AIMessage(content="Assessment received."),
    ])


def test_guard_preserves_existing_count_and_denies_without_increment():
    case = {"model_calls": 1}
    guard = build_model_call_guard(case, max_model_calls=2)
    guard()
    assert case["model_calls"] == 2
    with pytest.raises(ModelCallBudgetExceeded):
        guard()
    assert case["model_calls"] == 2


@pytest.mark.parametrize("case", [{}, {"model_calls": None}, {"model_calls": True},
                                  {"model_calls": -1}, {"model_calls": "0"}])
def test_guard_rejects_missing_or_malformed_count(case):
    before = deepcopy(case)
    with pytest.raises(ValueError):
        build_model_call_guard(case, max_model_calls=2)()
    assert case == before


@pytest.mark.parametrize("limit", [None, True, 0, -1, "2"])
def test_guard_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        build_model_call_guard({"model_calls": 0}, max_model_calls=limit)


@pytest.mark.parametrize("first_error", [None, TimeoutError("SECRET"),
                                       OutputParserException("SECRET"), RuntimeError("SECRET")])
def test_real_loop_shares_counter_with_classifier(monkeypatch, case, assessment, first_error):
    effects = [assessment] if first_error is None else [first_error, assessment]
    calls = Mock(side_effect=effects)
    monkeypatch.setattr(classifier, "assess_request", calls)
    model = make_model()
    guard = build_model_call_guard(case, max_model_calls=8)
    routing_tool = supervisor.build_classifier_routing_tool(
        case, classifier_model=model, reserve_model_call=guard,
    )
    agent = supervisor.build_supervisor_agent(model, [routing_tool], reserve_model_call=guard)
    agent.invoke({"messages": [HumanMessage(content="Classify the prepared case.")]},
                 config={"max_concurrency": 1, "recursion_limit": 20})
    expected_classifier_calls = 2 if isinstance(first_error, (TimeoutError, OutputParserException)) else 1
    assert calls.call_count == expected_classifier_calls
    assert model.invocations == 2
    assert case["model_calls"] == 2 + expected_classifier_calls
    assert case["trace"][0] == {"event_id": "intake"}
    assert "SECRET" not in json.dumps(case["trace"])
    assert case["classification_status"] == ("failed" if isinstance(first_error, RuntimeError) else "completed")


def test_duplicate_tool_call_does_not_spend_classifier_budget_again(monkeypatch, case, assessment):
    calls = Mock(return_value=assessment)
    monkeypatch.setattr(classifier, "assess_request", calls)
    model = make_model(duplicate=True)
    guard = build_model_call_guard(case, max_model_calls=3)
    routing_tool = supervisor.build_classifier_routing_tool(case, classifier_model=model, reserve_model_call=guard)
    graph = supervisor.build_supervisor_agent(model, [routing_tool], reserve_model_call=guard)
    result = graph.invoke({"messages": [HumanMessage(content="Classify.")]},
                          config={"max_concurrency": 1, "recursion_limit": 20})
    assert calls.call_count == 1
    assert case["model_calls"] == 3
    assert len([message for message in result["messages"] if message.type == "tool"]) == 2


@pytest.mark.parametrize("limit, initial, status, classifier_calls, supervisor_calls", [
    (1, 0, "failed", 0, 1),
    (2, 0, "completed", 1, 1),
    (2, 2, "skipped", 0, 0),
])
def test_supervisor_exhaustion_returns_unsaved_handoff(monkeypatch, case, assessment,
        limit, initial, status, classifier_calls, supervisor_calls):
    case["model_calls"] = initial
    case["approved_response"] = {"body": "old approval"}
    before = deepcopy(case)
    calls = Mock(return_value=assessment)
    monkeypatch.setattr(classifier, "assess_request", calls)
    model = make_model()
    config = {"configurable": {"thread_id": "ticket-1", "limits": {
        "max_graph_steps": 20, "max_model_calls": limit, "call_timeout_seconds": 5.0,
    }}}
    original_config = deepcopy(config)
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert result.goto == "persist"
    assert result.update["outcome"] == "escalated"
    assert result.update["approved_response"] is None
    assert result.update["messaging_disabled"] is True
    assert result.update["acknowledgement_permitted"] is False
    assert result.update["classification_status"] == status
    assert result.update["handoff"]["handoff_source"] == "deterministic_fallback"
    assert result.update["handoff"]["message_id"] == "turn-1:handoff"
    assert "messages" not in result.update and "persistence_result" not in result.update
    assert all(event["event_id"] != "intake" for event in result.update["trace"])
    assert {**case, **result.update}["model_calls"] == limit
    assert calls.call_count == classifier_calls
    assert model.invocations == supervisor_calls
    assert case == before and config == original_config
    json.dumps(result.update)


def test_classifier_preserves_attempts_when_retry_is_denied(monkeypatch, case):
    calls = Mock(side_effect=TimeoutError("SECRET"))
    monkeypatch.setattr(classifier, "assess_request", calls)
    guard = build_model_call_guard(case, max_model_calls=1)
    result = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=guard)
    assert case["model_calls"] == 1
    assert "model_calls" not in result
    assert calls.call_count == 1
    assert result["technical_attempts"]["turn-1: classifier: calls"] == 1
    assert result["classification_status"] == "failed"
    assert result["trace"][0]["payload"]["error_code"] == "model_call_budget_exhausted"
    assert "SECRET" not in json.dumps(result)


def test_tool_budget_exception_reaches_caller(case):
    @tool
    def route_to_classifier() -> str:
        """Demonstrate an exhausted tool guard."""
        raise ModelCallBudgetExceeded("exhausted")
    model = make_model()
    guard = build_model_call_guard(case, max_model_calls=3)
    graph = supervisor.build_supervisor_agent(model, [route_to_classifier], reserve_model_call=guard)
    with pytest.raises(ModelCallBudgetExceeded):
        graph.invoke({"messages": [HumanMessage(content="Classify.")]},
                     config={"max_concurrency": 1, "recursion_limit": 20})
    assert model.invocations == 1
