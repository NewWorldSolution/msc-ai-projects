"""Every Supervisor exit is a routing_decision event; the readable routing path; memory dedupe."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from agentic.agents import supervisor
import utils
from test_model_budget import case, assessment, make_model  # noqa: F401
from test_supervisor_routes import config, resume  # noqa: F401
from test_supervisor_review import review_case, model_with, assessment_result  # noqa: F401


def existing_escalation(case):
    case["account_context"]["ticket_status"] = "escalated"
    return "persist"


def blocked_customer(case):
    case["account_context"]["is_blocked"] = True
    return "escalation"


def step_allowance_exhausted(case):
    case["workflow_steps"] = 20
    return "persist"


def budget_exhausted(case):
    case["model_calls"] = 10
    return "persist"


def classifier_dispatch(case):
    return "classifier"


@pytest.mark.parametrize("arrange", [existing_escalation, blocked_customer, step_allowance_exhausted,
                                     budget_exhausted, classifier_dispatch])
def test_early_exits_record_their_routing_decision(case, config, arrange):
    expected = arrange(case)
    result = resume(case, config)
    assert result.goto == expected
    events = [event for event in result.update["trace"] if event["event_type"] == "routing_decision"]
    assert len(events) == 1
    event = events[0]
    assert event["payload"]["destination"] == expected
    assert event["node"] == "supervisor" and event["turn_id"] == "turn-1"
    assert event["event_id"].startswith("turn-1:routing:")
    all_ids = [e["event_id"] for e in case["trace"]] + [e["event_id"] for e in result.update["trace"]]
    assert len(all_ids) == len(set(all_ids))
    json.dumps(result.update)


def test_terminal_events_stay_unique_across_two_exits_in_one_turn(case, config):
    blocked_customer(case)
    first = resume(case, config)
    merged = deepcopy(case)
    for key, value in first.update.items():
        merged[key] = merged.get(key, []) + value if key == "trace" else value
    # The Escalation agent returns an invalid handoff; the Supervisor exits again without advancing steps.
    merged["handoff"] = {"turn_id": "wrong-turn"}
    second = resume(merged, config)
    assert second.goto == "persist"
    ids = [e["event_id"] for e in merged["trace"]] + [e["event_id"] for e in second.update["trace"]]
    assert len(ids) == len(set(ids))


def test_routing_path_reads_only_routing_events():
    events = [{"event_type": "classification"}]
    for destination in ("classifier", "knowledge", "response", "persist"):
        events.append({"event_type": "routing_decision", "payload": {"destination": destination}})
    assert utils.routing_path(events, channel="email", outcome="resolved", reply_present=True) == (
        "email -> intake -> supervisor -> classifier -> supervisor -> knowledge -> supervisor "
        "-> response -> supervisor -> persist -> customer (answer)")
    assert utils.routing_path(events, channel="chat", outcome="information_requested").endswith("persist -> customer (question)")
    assert utils.routing_path(events, channel="chat", outcome="escalated").endswith("persist -> human support (no customer reply)")
    assert utils.routing_path(events, channel="chat", outcome="escalated", reply_present=True).endswith(
        "persist -> human support + customer (receipt)")
    assert utils.routing_path(events[:3], channel="chat") == (
        "chat -> intake -> supervisor -> classifier -> supervisor -> knowledge -> supervisor")
    assert utils.routing_path([], outcome="resolved") == "message -> intake -> supervisor"


def resolved_review(review_case):
    review_case.update(outcome="resolved", memory_updates=[])
    review_case["response_draft"].update(
        outcome="resolved", addressed_issue_ids=[review_case["issues"][0]["issue_id"]],
        body="Welcome back. Use the verified password reset steps.", history_acknowledgement="Welcome back.",
        claim_to_evidence=[{"claim": "Use the verified password reset steps.", "issue_ids": [], "source_ids": ["article:1"]}])
    review_case["issues"][0].update(coverage="supported", article_ids=["article-1"])
    review_case["retrieved_articles"] = [{"article_id": "article-1", "excerpt": "Use the verified password reset steps.",
                                          "gate_pass": True}]
    return review_case


def test_review_does_not_fail_grounding_on_the_acknowledgement(review_case):
    resolved_review(review_case)
    model = model_with(assessment_result(outcome="resolved", unsupported_sentences=["Welcome back."]))
    result = supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "approved" and result["findings"]["grounded"] is True
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0].messages[0].content
    assert "do not list it as unsupported" in prompt
    listed_prose = assessment_result(outcome="resolved", unsupported_sentences=["Resets expire in a day."])
    result = supervisor.review_response(review_case, model=model_with(listed_prose), reserve_model_call=Mock())
    assert result["verdict"] == "rejected"


def test_known_preference_is_not_written_to_memory_again(review_case, config):
    quote = "Please keep replies concise."
    resolved_review(review_case)
    review_case["classification"]["stated_preferences"] = [quote]
    fresh = supervisor.supervise_case(deepcopy(review_case), config, model=model_with(assessment_result(outcome="resolved")))
    assert [item["memory_type"] for item in fresh.update["memory_updates"]] == ["resolved_issue", "preference"]
    review_case["customer_memory"] = [{"memory_id": "old", "memory_type": "preference", "outcome": "resolved",
                                       "content": quote.upper()}]
    returning = supervisor.supervise_case(review_case, config, model=model_with(assessment_result(outcome="resolved")))
    assert returning.goto == "persist"
    assert [item["memory_type"] for item in returning.update["memory_updates"]] == ["resolved_issue"]
