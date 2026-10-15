"""First dispatch and specialist continuation, without APIs."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic.agents import classifier, supervisor
from test_model_budget import case, assessment, make_model, ScriptedModel


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "ticket-1", "limits": {
        "max_graph_steps": 20, "max_model_calls": 10, "call_timeout_seconds": 5.0,
    }}}


def resume(case, config):
    model = Mock()
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert not model.mock_calls
    return result


def completed(case, assessment, agent):
    case["classification_status"] = "completed"
    case["classification"] = assessment.model_dump()
    case["issues"] = [{**assessment.issues[0].model_dump(),
                       "issue_id": "turn-1:issue:1", "missing_details": [],
                       "article_ids": [], "initial_retrieval_attempted": False,
                       "targeted_retry_attempted": False}]
    case["assignment"] = {"agent": agent, "issue_ids": ["turn-1:issue:1"]}
    return case


def test_operations_first_dispatch_is_focused_and_not_escalated(
    monkeypatch, case, assessment, config
):
    ordinary_issue = assessment.issues[0].model_copy(deep=True)

    assessment.issues[0].needs_operations = True
    assessment.issues[0].required_business_facts = ["Subscription status"]
    assessment.issues.append(ordinary_issue)

    monkeypatch.setattr(
        classifier,
        "assess_request",
        Mock(return_value=assessment),
    )

    # Supply the response for Supervisor's new lookup-planning call.
    planning_model = Mock()
    planning_model.invoke.return_value = {
        "issues": [
            {
                "issue_id": "turn-1:issue:1",
                "reservation_mode": None,
                "experience_mode": None,
            }
        ]
    }

    def structured_output(self, schema, **kwargs):
        assert schema is supervisor.OperationsLookupPlan
        return planning_model

    # Replace the fake model's structured-output adapter for this test.
    monkeypatch.setattr(
        ScriptedModel,
        "with_structured_output",
        structured_output,
    )

    model = make_model()
    # Classification has returned from its separate graph node.
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    before = deepcopy(case)

    result = supervisor.supervise_case(
        case,
        config,
        model=model,
        classifier_model=model,
    )

    assert result.goto == "operations"
    assert result.update["assignment"]["agent"] == "operations"
    assert result.update["assignment"]["issue_ids"] == ["turn-1:issue:1"]

    # Subscription lookup requires no reservation or experience mode.
    assert result.update["assignment"]["lookup_modes"] == {
        "turn-1:issue:1": {}
    }

    planning_model.invoke.assert_called_once()

    # One classification call plus the existing lookup-planning call.
    assert result.update["model_calls"] == 2

    assert result.update.get("outcome") != "escalated"
    assert not result.update.get("escalation_reason")
    assert case == before


def test_failed_classifier_clears_declared_approval(monkeypatch, case, config):
    case["approved_response"] = {"body": "old"}
    case["classification_status"] = "failed"
    monkeypatch.setattr(classifier, "assess_request", Mock(side_effect=RuntimeError("private")))
    model = make_model()
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert result.goto == "escalation"
    assert result.update["approved_response"] is None
    assert "response_approval" not in result.update


def test_classifier_assignment_clears_prior_review_without_model_coordination(case, config):
    case["review_result"] = {"outcome": "pass", "draft_id": "old"}
    model = ScriptedModel(responses=[AIMessage(content="Unused")])
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert result.goto == "classifier"
    assert result.update["review_result"] is None
    assert "review_reason" not in result.update
    assert model.invocations == 0


@pytest.mark.parametrize("outcome, expected", [
    ("found", "knowledge"), ("not_found", "knowledge"),
    ("missing_input", "response"), ("ambiguous", "response"),
    ("technical_failure", "escalation"),
])
def test_operations_return_uses_actual_outcome(case, assessment, config, outcome, expected):
    completed(case, assessment, "operations")
    case["issues"][0]["needs_operations"] = True
    case["issues"][0]["required_business_facts"] = ["Subscription status"]
    case["tool_results"] = [{"turn_id": "turn-1", "issue_id": "turn-1:issue:1",
                             "lookup_key": "subscription:customer-1", "attempt": 1,
                             "outcome": outcome, "records": []}]
    before = deepcopy(case)
    result = resume(case, config)
    assert result.goto == expected
    assert result.update["assignment"]["agent"] == expected
    assert "tool_results" not in result.update
    assert "messages" not in result.update
    assert "model_calls" not in result.update
    assert case == before


def test_operations_cannot_reuse_another_turns_results(case, assessment, config):
    completed(case, assessment, "operations")
    case["tool_results"] = [{"turn_id": "old-turn", "issue_id": "turn-1:issue:1",
                             "lookup_key": "subscription", "attempt": 1, "outcome": "found"}]
    assert resume(case, config).goto == "escalation"


def test_knowledge_return_supported_routes_to_draft_without_reclassification(case, assessment, config):
    completed(case, assessment, "knowledge")
    case["issues"][0].update(coverage="supported", article_ids=["article-1"],
                             initial_retrieval_attempted=True)
    case["retrieved_articles"] = [{"article_id": "article-1", "excerpt": "Stored guidance"}]
    result = resume(case, config)
    assert result.goto == "response"
    assert result.update["outcome"] == "resolved"
    assert result.update["approved_response"] is None
    assert result.update["response_stage"] == "original"


@pytest.mark.parametrize("coverage", [None, "evidence_problem", "supported"])
def test_missing_or_invalid_knowledge_evidence_never_approves(case, assessment, config, coverage):
    completed(case, assessment, "knowledge")
    case["issues"][0]["coverage"] = coverage
    assert resume(case, config).goto == "escalation"


def test_targeted_retry_is_not_reissued_on_unchanged_return(case, assessment, config):
    completed(case, assessment, "knowledge")
    case["issues"][0].update(coverage="targeted_retry_needed", article_ids=["article-1"],
                             initial_retrieval_attempted=True)
    case["retrieved_articles"] = [{"article_id": "article-1"}]
    first = resume(case, config)
    assert first.goto == "knowledge"
    assert first.update["assignment"]["retrieval_mode"] == "targeted"
    case.update(first.update)
    assert resume(case, config).goto == "escalation"


def test_unavailable_fact_does_not_repeat_same_operations_assignment(case, assessment, config):
    completed(case, assessment, "operations")
    issue = case["issues"][0]
    issue["required_business_facts"] = ["Subscription status"]
    case["tool_results"] = [{"turn_id": "turn-1", "issue_id": issue["issue_id"],
                             "lookup_key": "subscription", "attempt": 1, "outcome": "not_found"}]
    case.update(resume(case, config).update)
    issue["coverage"] = "awaiting_business_facts"
    assert resume(case, config).goto == "response"


def test_response_return_does_not_restart_first_dispatch(case, assessment, config):
    completed(case, assessment, "response")
    case["response_draft"] = {"draft_id": "draft-1", "body": "Proposed reply"}
    result = resume(case, config)
    assert result.goto == "escalation"
    assert result.update["messaging_disabled"] is True
    assert result.update["review_result"]["error_code"] == "invalid_review_state"


def test_classifier_dispatch_returns_only_changed_fields(monkeypatch, case, assessment, config):
    case["tool_results"] = [{"lookup_key": "previous", "outcome": "found"}]
    before = deepcopy(case)
    assess = Mock(return_value=assessment)
    monkeypatch.setattr(classifier, "assess_request", assess)
    model = make_model()
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert result.goto == "classifier"
    assert result.update["assignment"]["agent"] == "classifier"
    assert not {"model_calls", "classification_status", "messages", "tool_results", "ticket_id", "account_context"} & result.update.keys()
    # The dispatch is a routing decision, so it is the only trace entry the Supervisor adds here.
    assert [event["event_type"] for event in result.update["trace"]] == ["routing_decision"]
    assert result.update["trace"][0]["payload"]["destination"] == "classifier"
    assess.assert_not_called()
    assert model.invocations == 0 and case == before


def test_step_seven_applies_append_only_deltas_without_message_leakage(monkeypatch, case, assessment, config):
    from langgraph.graph import END, START, StateGraph
    from agentic.state import CaseState

    old_result = {"lookup_key": "old", "outcome": "found"}
    new_result = {"lookup_key": "new", "outcome": "not_found"}
    case["tool_results"] = [old_result]
    before = deepcopy(case)
    classified = completed(deepcopy(case), assessment, "classifier")

    def specialist_delta(state, **kwargs):
        return {
            "classification_status": "completed",
            "classification": classified["classification"],
            "issues": classified["issues"],
            "trace": [{"event_id": "new-event"}],
            "tool_results": [new_result],
        }

    model = make_model()
    graph = StateGraph(CaseState)
    graph.add_node("supervisor", lambda state: supervisor.supervise_case(
        state, config, model=model, classifier_model=model))
    graph.add_node("classifier", specialist_delta)
    graph.add_node("knowledge", lambda state: {})
    graph.add_edge(START, "supervisor")
    graph.add_edge("classifier", "supervisor")
    graph.add_edge("knowledge", END)
    result = graph.compile().invoke(case)
    assert result["tool_results"] == [old_result, new_result]
    # Routing decisions add their own event; the append-only check ignores them.
    assert [event["event_id"] for event in result["trace"]
            if event.get("event_type") != "routing_decision"] == ["intake", "new-event"]
    assert any(event.get("event_type") == "routing_decision" for event in result["trace"])
    assert result["messages"] == before["messages"]
    assert case == before
