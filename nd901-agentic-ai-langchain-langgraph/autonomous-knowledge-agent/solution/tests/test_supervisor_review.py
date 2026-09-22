"""Exact-draft review and routing regressions with no external model calls."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from langchain_core.exceptions import OutputParserException
from langgraph.errors import GraphRecursionError

from agentic.agents import supervisor
from agentic.model_budget import build_model_call_guard
from agentic.tools.handoff_tools import assemble_fallback_handoff
from test_model_budget import case, assessment
from test_supervisor_routes import completed, config


@pytest.fixture
def review_case(case, assessment):
    completed(case, assessment, "response")
    case.update(channel="chat", outcome="information_requested", response_stage="original",
                original_review_attempts=0, acknowledgement_review_attempts=0,
                review_history=[], acknowledgement_permitted=False,
                response_draft={"draft_id": "draft-1", "body": "Which reservation do you mean?",
                                "channel": "chat", "outcome": "information_requested"})
    return case


def assessment_result(**changes):
    return dict(draft_id="draft-1", outcome="information_requested", grounded=True,
                complete=True, correct_outcome=True, appropriate_format=True,
                grounding_findings=["No unsupported business claim."],
                completeness_findings=["Requests the missing detail."], revision_reasons=[]) | changes


def model_with(*effects):
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = list(effects)
    return model


def run(case, config, model):
    return supervisor.supervise_case(case, config, model=model, classifier_model=Mock())


def merge(case, result):
    updated = deepcopy(case)
    for key, value in result.update.items():
        if key in ("trace", "tool_results"):
            updated.setdefault(key, []).extend(value)
        else:
            updated[key] = value
    return updated


def test_review_returns_explanations_without_mutating_case(review_case):
    model = model_with(assessment_result())
    reserve = Mock()
    before = deepcopy(review_case)
    result = supervisor.review_response(review_case, model=model, reserve_model_call=reserve)
    assert result["verdict"] == "approved"
    assert result["grounding_findings"] and result["completeness_findings"]
    assert review_case == before
    reserve.assert_called_once_with()


@pytest.mark.parametrize("change", [
    {"draft_id": "other"}, {"outcome": "resolved"},
    {"complete": False}, {"revision_reasons": ["Contradictory revision"]},
])
def test_review_rejects_mismatched_or_inconsistent_assessment(review_case, change):
    with pytest.raises(OutputParserException):
        supervisor.review_response(review_case, model=model_with(assessment_result(**change)), reserve_model_call=Mock())


@pytest.mark.parametrize("change", [{"body": "   "}, {"draft_id": None}, {"channel": "email"}])
def test_invalid_draft_does_not_spend_model_budget(review_case, change):
    review_case["response_draft"].update(change)
    guard = Mock()
    model = Mock()
    with pytest.raises(ValueError):
        supervisor.review_response(review_case, model=model, reserve_model_call=guard)
    guard.assert_not_called()
    assert not model.mock_calls


def test_approval_uses_exact_wording_and_persist_contract(review_case, config):
    before = deepcopy(review_case)
    result = run(review_case, config, model_with(assessment_result()))
    assert result.goto == "persist"
    assert result.update["approved_response"] == {
        "message_id": "turn-1:reply", "draft_id": "draft-1",
        "outcome": "information_requested", "content": review_case["response_draft"]["body"],
    }
    assert result.update["review_result"]["verdict"] == "approved"
    assert result.update["original_review_attempts"] == 1
    assert result.update["model_calls"] == 1
    assert "messages" not in result.update
    assert review_case == before
    json.dumps(result.update)


def test_email_keeps_reviewed_subject_and_body(review_case, config):
    review_case["channel"] = "email"
    review_case["response_draft"].update(channel="email", subject="Reservation details")
    result = run(review_case, config, model_with(assessment_result()))
    assert result.update["approved_response"]["content"] == "Subject: Reservation details\n\nWhich reservation do you mean?"


@pytest.mark.parametrize("error, calls, reviews", [
    (TimeoutError("PRIVATE"), 2, 1),
    (OutputParserException("PRIVATE"), 2, 1),
    (RuntimeError("PRIVATE"), 1, 0),
])
def test_review_retries_only_permitted_failures(review_case, config, error, calls, reviews):
    model = model_with(error, assessment_result())
    result = run(review_case, config, model)
    assert model.with_structured_output.return_value.invoke.call_count == calls
    assert result.update.get("original_review_attempts", 0) == reviews
    assert result.update["model_calls"] == calls
    assert "PRIVATE" not in json.dumps(result.update)
    assert result.goto == ("persist" if reviews else "escalation")


@pytest.mark.parametrize("error", [TimeoutError("PRIVATE"), OutputParserException("PRIVATE")])
def test_repeated_review_failure_never_consumes_content_review(review_case, config, error):
    result = run(review_case, config, model_with(error, error))
    assert result.goto == "escalation"
    assert result.update["messaging_disabled"] is True
    assert result.update["review_result"]["verdict"] == "technical_failure"
    assert "original_review_attempts" not in result.update
    assert result.update["model_calls"] == 2


def test_exhausted_review_budget_builds_handoff_without_model(review_case, config):
    review_case["model_calls"] = config["configurable"]["limits"]["max_model_calls"]
    model = model_with(assessment_result())
    result = run(review_case, config, model)
    assert result.goto == "persist"
    assert result.update["handoff"]["handoff_source"] == "deterministic_fallback"
    model.with_structured_output.return_value.invoke.assert_not_called()


def test_second_original_rejection_allows_one_acknowledgement_stage(review_case, config):
    rejected = assessment_result(grounded=False, revision_reasons=["Unsupported claim."])
    first = run(review_case, config, model_with(rejected))
    assert first.goto == "response"
    updated = merge(review_case, first)
    updated["response_draft"]["draft_id"] = "draft-2"
    second = run(updated, config, model_with({**rejected, "draft_id": "draft-2"}))
    assert second.goto == "escalation"
    assert second.update["original_review_attempts"] == 2
    assert second.update["acknowledgement_permitted"] is True
    updated = merge(updated, second)
    updated["handoff"] = {**assemble_fallback_handoff(updated), "handoff_source": "generated"}
    acknowledgement = run(updated, config, Mock())
    assert acknowledgement.goto == "response"
    assert acknowledgement.update["response_stage"] == "acknowledgement"
    updated = merge(updated, acknowledgement)
    updated["response_draft"] = {"draft_id": "ack-1", "body": "Your case is prepared for human review.",
                                 "channel": "chat", "outcome": "escalated"}
    first_ack = run(updated, config, model_with({**rejected, "draft_id": "ack-1", "outcome": "escalated"}))
    assert first_ack.goto == "response"
    updated = merge(updated, first_ack)
    updated["response_draft"]["draft_id"] = "ack-2"
    final = run(updated, config, model_with({**rejected, "draft_id": "ack-2", "outcome": "escalated"}))
    assert final.goto == "persist"
    assert final.update["acknowledgement_review_attempts"] == 2
    assert final.update["messaging_disabled"] is True
    assert updated["original_review_attempts"] == 2


def test_exact_approval_replay_is_free_but_changed_draft_is_not_approved(review_case, config):
    updated = merge(review_case, run(review_case, config, model_with(assessment_result())))
    model = Mock()
    replay = run(updated, config, model)
    assert replay.goto == "persist"
    assert "model_calls" not in replay.update
    assert "original_review_attempts" not in replay.update
    assert not model.mock_calls
    updated["response_draft"]["body"] = "A different unreviewed answer."
    result = run(updated, config, model)
    assert result.goto == "escalation"
    assert result.update["approved_response"] is None


def test_missing_identity_first_requests_correction(case, config):
    case["preparation_result"]["outcome"] = "customer_not_found"
    result = run(case, config, Mock())
    assert result.goto == "response"
    assert result.update["identity_correction"]["asked"] is True
    case["identity_correction"] = {"asked": True}
    assert run(case, config, Mock()).goto == "escalation"


@pytest.mark.parametrize("error, calls", [(TimeoutError("PRIVATE"), 2), (GraphRecursionError("PRIVATE"), 1),
                                         (RuntimeError("PRIVATE"), 1)])
def test_coordination_failure_reaches_safe_persistence(monkeypatch, case, config, error, calls):
    agent = Mock()
    agent.invoke.side_effect = error
    monkeypatch.setattr(supervisor, "build_supervisor_agent", Mock(return_value=agent))
    result = run(case, config, Mock())
    assert result.goto == "persist"
    assert agent.invoke.call_count == calls
    assert result.update["handoff"]["handoff_source"] == "deterministic_fallback"
    assert "PRIVATE" not in json.dumps(result.update)


def test_step_allowance_stops_deterministic_reentry(case, config):
    case["workflow_steps"] = config["configurable"]["limits"]["max_graph_steps"]
    assert run(case, config, Mock()).goto == "persist"


def test_missing_escalation_result_does_not_restart_generation(case, config):
    case.update(outcome="escalated", assignment={"agent": "escalation"})
    result = run(case, config, Mock())
    assert result.goto == "persist"
    assert result.update["handoff"]["handoff_source"] == "deterministic_fallback"


def test_blocker_preserves_review_allowance_and_does_not_repeat_unchanged_work(review_case, config):
    review_case.update(original_review_attempts=1,
                       draft_blocker={"issue_ids": ["turn-1:issue:1"], "reason": "Missing applicable guidance"})
    model = Mock()
    result = run(review_case, config, model)
    assert result.goto == "knowledge"
    assert "original_review_attempts" not in result.update
    assert not model.mock_calls
    updated = merge(review_case, result)
    updated["assignment"] = {"agent": "response", "issue_ids": ["turn-1:issue:1"]}
    updated["draft_blocker"] = review_case["draft_blocker"]
    repeated = run(updated, config, model)
    assert repeated.goto == "escalation"
    assert "original_review_attempts" not in repeated.update


def test_review_context_excludes_old_turn_lookup_results(review_case):
    review_case["tool_results"] = [{"turn_id": "old", "records": ["OLD"]},
                                   {"turn_id": "turn-1", "records": ["CURRENT"]}]
    model = model_with(assessment_result())
    supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0]
    context = json.loads(prompt.messages[-1].content.split("\n", 1)[1])
    assert context["tool_result"] == [{"turn_id": "turn-1", "records": ["CURRENT"]}]


def test_evidence_change_invalidates_cached_approval(review_case, config):
    updated = merge(review_case, run(review_case, config, model_with(assessment_result())))
    updated["retrieved_articles"] = [{"article_id": "new", "excerpt": "Changed guidance"}]
    result = run(updated, config, Mock())
    assert result.goto == "escalation"
    assert result.update["approved_response"] is None


def test_invalid_handoff_is_replaced_before_any_customer_reply(case, config):
    case.update(outcome="escalated", handoff={"turn_id": "old", "handoff_source": "generated"})
    result = run(case, config, Mock())
    assert result.goto == "persist"
    assert result.update["handoff"]["turn_id"] == "turn-1"
    assert result.update["messaging_disabled"] is True


@pytest.mark.parametrize("with_evidence", [True, False])
def test_resolved_review_requires_actual_referenced_evidence(review_case, with_evidence):
    review_case["outcome"] = "resolved"
    review_case["response_draft"].update(outcome="resolved", addressed_issue_ids=["turn-1:issue:1"])
    review_case["issues"][0].update(coverage="supported", article_ids=["article-1"])
    if with_evidence:
        review_case["retrieved_articles"] = [{"article_id": "article-1", "excerpt": "Stored guidance", "gate_pass": True}]
    result = supervisor.review_response(review_case, model=model_with(assessment_result(outcome="resolved")),
                                         reserve_model_call=Mock())
    assert result["verdict"] == ("approved" if with_evidence else "rejected")


def test_acknowledgement_cannot_be_approved_without_handoff(review_case):
    review_case["outcome"] = "escalated"
    review_case["response_draft"]["outcome"] = "escalated"
    result = supervisor.review_response(review_case, model=model_with(assessment_result(outcome="escalated")),
                                         reserve_model_call=Mock())
    assert result["verdict"] == "rejected"
