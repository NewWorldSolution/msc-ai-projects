"""Credential-free regressions for shared reservations and budget handoffs."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agentic.agents import classifier, supervisor
from agentic.model_budget import ModelCallBudgetExceeded, build_model_call_guard


class ScriptedModel(FakeMessagesListChatModel):
    """A fake chat model that counts its invocations and never calls a provider."""

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


@pytest.mark.parametrize("limit, initial, dispatch", [(1, 0, True), (2, 0, True), (2, 2, False)])
def test_supervisor_dispatches_without_spending_calls_or_saves_exhausted_turn(
        monkeypatch, case, assessment, limit, initial, dispatch):
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
    assert result.update["approved_response"] is None
    assert "messages" not in result.update and "persistence_result" not in result.update
    assert calls.call_count == 0 and model.invocations == 0
    assert case == before and config == original_config
    assert {**case, **result.update}["model_calls"] == initial
    if dispatch:
        assert result.goto == "classifier"
        assert result.update["assignment"]["agent"] == "classifier"
        assert "classification_status" not in result.update
    else:
        assert result.goto == "persist"
        assert result.update["outcome"] == "escalated"
        assert result.update["messaging_disabled"] is True
        assert result.update["acknowledgement_permitted"] is False
        assert result.update["classification_status"] == "skipped"
        assert result.update["handoff"]["handoff_source"] == "deterministic_fallback"
        assert result.update["handoff"]["message_id"] == "turn-1:handoff"
        assert all(event["event_id"] != "intake" for event in result.update["trace"])
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


