"""Response drafting tests: fake models only, no provider or database writes."""

from copy import deepcopy
import json
import threading
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from agentic.agents import response, supervisor


@pytest.fixture
def case():
    return dict(
        ticket_id="ticket1", account_id="account1", customer_id="customer1", turn_id="turn1",
        channel="chat", outcome="resolved", response_stage="original", messaging_disabled=False,
        acknowledgement_permitted=False, original_review_attempts=0, acknowledgement_review_attempts=0,
        classification_status="completed", model_calls=0, workflow_steps=0, technical_attempts={}, trace=[],
        preparation_result={"outcome": "ready", "ticket_lookup_outcome": "success"},
        account_context={"is_blocked": False}, messages=[HumanMessage(content="How do I cancel?", id="turn1")],
        assignment={"agent": "response", "task": "Explain cancellation using the accepted guidance.", "issue_ids": ["issue1"]},
        issues=[{"issue_id": "issue1", "task": "Explain cancellation", "coverage": "supported", "article_ids": ["article1"]}],
        retrieved_articles=[dict(account_id="account1", turn_id="turn1", issue_id="issue1", article_id="article1",
                                 gate_pass=True, score=0.9, threshold=0.5, title="Cancellation", content="Use the cancellation form.",
                                 excerpt="Use the cancellation form.")], tool_results=[],
        approved_response={"content": "OLD"}, response_draft=None, review_result=None, review_history=[],
    )


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "ticket1", "limits": {
        "max_model_calls": 10, "max_graph_steps": 20, "call_timeout_seconds": 1.0}}}


def draft(**changes):
    return {"draft": dict(outcome="resolved", channel="chat", subject=None, greeting=None,
                          body="Use the cancellation form.", closing=None, addressed_issue_ids=["issue1"],
                          claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article:1"])],
                          clarification_questions=[], history_acknowledgement=None, **changes), "blocker": None}


def model_with(*effects):
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = list(effects)
    return model


def call(case, config, *effects):
    model = model_with(*effects)
    before = deepcopy(case)
    result = response.draft_response(case, config, model=model)
    assert case == before
    assert not {"messages", "outcome", "assignment", "issues", "review_result", "review_history",
                "original_review_attempts", "acknowledgement_review_attempts"} & result.keys()
    assert result["approved_response"] is None
    json.dumps(result, allow_nan=False)
    return result, model


def test_resolved_draft_and_source_ids(case, config):
    result, model = call(case, config, draft())
    value = result["response_draft"]
    assert value["body"] == "Use the cancellation form."
    assert value["draft_id"].startswith("turn1:draft:")
    assert value["source_catalog"]["article:1"]["data"]["article_id"] == "article1"
    assert result["model_calls"] == 1 and result["draft_blocker"] is None
    model.with_structured_output.assert_called_once_with(response.ResponseGeneration)


def test_email_assembles_reviewed_body(case, config):
    case["channel"] = "email"
    value = draft()["draft"]
    value.update(channel="email", subject="Cancellation guidance", greeting="Hello,", closing="Kind regards")
    result, _ = call(case, config, {"draft": value, "blocker": None})
    assert result["response_draft"]["body"] == "Hello,\n\nUse the cancellation form.\n\nKind regards"


@pytest.mark.parametrize("change", [
    {"outcome": "escalated"}, {"channel": "email"}, {"body": "  "},
    {"addressed_issue_ids": []}, {"addressed_issue_ids": ["issue1", "issue1"]},
    {"subject": "Not chat"}, {"claim_to_evidence": []},
    {"claim_to_evidence": [{"claim": "Use the cancellation form.", "issue_ids": ["issue1"], "source_ids": ["invented"]}]},
    {"claim_to_evidence": [{"claim": "Use the cancellation form.", "issue_ids": ["issue1"], "source_ids": ["assignment"]}]},
    {"claim_to_evidence": [{"claim": "Absent wording", "issue_ids": ["issue1"], "source_ids": ["article:1"]}]},
])
def test_invalid_output_is_terminal_after_one_repair(case, config, change):
    value = draft()
    value["draft"].update(change)
    result, model = call(case, config, value, value)
    assert result["response_draft"] is None and result["draft_blocker"] is None
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_output"
    assert model.with_structured_output.return_value.invoke.call_count == 2


@pytest.mark.parametrize("update", [
    {"account_id": "foreign"}, {"turn_id": "old"}, {"issue_id": "other"}, {"gate_pass": False},
])
def test_unavailable_evidence_returns_blocker_without_model(case, config, update):
    case["retrieved_articles"][0].update(update)
    result, model = call(case, config)
    assert result["draft_blocker"]["issue_ids"] == ["issue1"]
    assert result["model_calls"] == 0
    assert not model.mock_calls


def test_model_blocker_is_not_a_draft_or_failure(case, config):
    result, _ = call(case, config, {"draft": None, "blocker": {
        "issue_ids": ["issue1"], "reason": "Need a timing condition.", "missing_evidence": ["Cancellation deadline."]}})
    assert result["response_draft"] is None
    assert result["trace"][-1]["event_type"] == "response_blocked"


@pytest.mark.parametrize("value", [
    {"draft": None, "blocker": None},
    {"draft": draft()["draft"], "blocker": {"issue_ids": ["issue1"], "reason": "Gap", "missing_evidence": ["Gap"]}},
    {"draft": None, "blocker": {"issue_ids": ["other"], "reason": "Gap", "missing_evidence": ["Gap"]}},
])
def test_result_requires_exactly_one_valid_variant(case, config, value):
    result, model = call(case, config, value, value)
    assert result["trace"][-1]["event_type"] == "response_generation_failed"
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_output"
    assert model.with_structured_output.return_value.invoke.call_count == 2


def clarification(case, *, identity=False):
    case.update(outcome="information_requested", retrieved_articles=[])
    case["assignment"]["task"] = "Ask for the reservation ID." if not identity else "Ask for corrected identifying details."
    if identity:
        case.update(customer_id=None, issues=[], classification_status="skipped",
                    preparation_result={"outcome": "customer_not_found", "ticket_lookup_outcome": "success"})
        case["assignment"].pop("issue_ids")
    ids = [] if identity else ["issue1"]
    return {"draft": dict(outcome="information_requested", channel="chat", subject=None, greeting=None,
                          body="Could you provide the identifying details?", closing=None, addressed_issue_ids=ids,
                          claim_to_evidence=[], clarification_questions=[dict(question="Could you provide the identifying details?",
                          issue_ids=ids, basis_source_ids=["assignment"])], history_acknowledgement=None), "blocker": None}


@pytest.mark.parametrize("identity", [False, True])
def test_clarification_needs_no_article_or_invented_issue(case, config, identity):
    result, _ = call(case, config, clarification(case, identity=identity))
    assert result["response_draft"]["outcome"] == "information_requested"


def test_clarification_without_question_basis_fails(case, config):
    value = clarification(case)
    value["draft"]["clarification_questions"][0]["basis_source_ids"] = ["invented"]
    result, model = call(case, config, value, value)
    assert result["response_draft"] is None
    assert result["trace"][-1]["payload"]["detail"] == "A clarification question lacks a supplied basis."
    assert model.with_structured_output.return_value.invoke.call_count == 2


def acknowledgement(case):
    case.update(outcome="escalated", response_stage="acknowledgement", acknowledgement_permitted=True,
                handoff={"handoff_source": "generated", "message_id": "turn1:handoff", "turn_id": "turn1", "issue_ids": ["issue1"]})
    case["assignment"].pop("issue_ids")
    return {"draft": dict(outcome="escalated", channel="chat", subject=None, greeting=None,
                          body="Your request requires human assistance.", closing=None, addressed_issue_ids=["issue1"],
                          # A receipt carries no claims.
                          claim_to_evidence=[], clarification_questions=[], history_acknowledgement=None), "blocker": None}


def test_acknowledgement_uses_prepared_handoff(case, config):
    result, _ = call(case, config, acknowledgement(case))
    assert result["response_draft"]["outcome"] == "escalated"


@pytest.mark.parametrize("change", [{"handoff_source": "deterministic_fallback"}, {"turn_id": "old"}])
def test_invalid_handoff_never_calls_model(case, config, change):
    value = acknowledgement(case)
    case["handoff"].update(change)
    model = model_with(value)
    with pytest.raises(ValueError):
        response.draft_response(case, config, model=model)
    assert not model.mock_calls


@pytest.mark.parametrize("error,expected_calls", [(TimeoutError("SECRET"), 2), (ConnectionError("SECRET"), 2), (RuntimeError("SECRET"), 1)])
def test_transient_only_retry_and_sanitized_failures(case, config, error, expected_calls):
    result, model = call(case, config, error, draft())
    assert model.with_structured_output.return_value.invoke.call_count == expected_calls
    assert result["model_calls"] == expected_calls
    assert "SECRET" not in json.dumps(result)
    assert bool(result["response_draft"]) == (expected_calls == 2)


def test_repeated_transient_stops_at_two(case, config):
    result, model = call(case, config, TimeoutError("SECRET"), TimeoutError("SECRET"))
    assert model.with_structured_output.return_value.invoke.call_count == 2
    assert result["trace"][-1]["payload"]["error_code"] == "response_transient_failure"


def test_existing_retry_counter_cannot_reset(case, config):
    key = json.dumps(["turn1", "response", "original", 0, "retry"], separators=(",", ":"))
    case["technical_attempts"][key] = 1
    result, model = call(case, config, TimeoutError("SECRET"))
    assert result["model_calls"] == 1


def test_budget_exhaustion_preserves_count_and_skips_invocation(case, config):
    case["model_calls"] = 10
    result, model = call(case, config)
    assert result["model_calls"] == 10
    model.with_structured_output.return_value.invoke.assert_not_called()
    assert result["trace"][-1]["payload"]["error_code"] == "response_model_budget_exhausted"


def test_deadline_does_not_overlap_retry_or_accept_late_result(case, config):
    started, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        started.set()
        release.wait(3)
        return draft()
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = slow
    config["configurable"]["limits"]["call_timeout_seconds"] = 0.02
    try:
        result = response.draft_response(case, config, model=model)
        assert started.is_set()
        assert result["trace"][-1]["payload"]["error_code"] == "response_call_timeout"
        assert result["response_draft"] is None and result["model_calls"] == 1
        assert model.with_structured_output.return_value.invoke.call_count == 1
    finally:
        release.set()


def test_redraft_gets_new_id_and_review_reasons_without_resetting_counter(case, config):
    first, _ = call(case, config, draft())
    case.update(response_draft=first["response_draft"], original_review_attempts=1,
                review_result={"revision_reasons": ["Make the instructions clearer."]})
    result, model = call(case, config, draft())
    assert result["response_draft"]["draft_id"] != first["response_draft"]["draft_id"]
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[-1].content
    assert "Make the instructions clearer." in prompt
    assert case["original_review_attempts"] == 1


def test_exhausted_content_stage_never_calls_model(case, config):
    case["original_review_attempts"] = 2
    model = model_with(draft())
    with pytest.raises(ValueError):
        response.draft_response(case, config, model=model)
    assert not model.mock_calls


def lookup(*, account="account1", customer="customer1", turn="turn1", outcome="found", attempt=1):
    key = json.dumps(dict(tool_name="lookup_subscription", account_id=account, customer_id=customer,
                          lookup_mode=None, arguments={"customer_id": customer}), sort_keys=True)
    return dict(turn_id=turn, issue_id="issue1", lookup_key=key, tool_name="lookup_subscription", attempt=attempt,
                outcome=outcome, retryable=False, reason="Stored result", records=[dict(subscription_id="sub1", user_id=customer, tier="gold")] if outcome=="found" else [])


@pytest.mark.parametrize("entry", [lookup(account="foreign"), lookup(customer="foreign"), lookup(turn="old")])
def test_foreign_or_old_lookup_is_excluded_from_prompt(case, config, entry):
    entry["reason"] = "SECRET_FOREIGN"
    case["tool_results"] = [entry]
    result, model = call(case, config, draft())
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[-1].content
    assert "SECRET_FOREIGN" not in prompt
    assert not any(s["kind"]=="lookup" for s in result["response_draft"]["source_catalog"].values())


def test_latest_failure_cannot_reuse_earlier_found(case, config):
    case["tool_results"] = [lookup(), lookup(outcome="technical_failure", attempt=2)]
    result, _ = call(case, config, draft())
    assert not any(s["kind"]=="lookup" for s in result["response_draft"]["source_catalog"].values())


def test_not_found_is_preserved_as_a_completed_fact(case, config):
    case["tool_results"] = [lookup(outcome="not_found")]
    result, _ = call(case, config, draft())
    assert result["response_draft"]["source_catalog"]["lookup:2"]["data"]["outcome"] == "not_found"


def test_supervisor_consumes_generation_failure_without_review(case, config):
    delta, _ = call(case, config, RuntimeError("SECRET"))
    returned = {**case, **delta}
    model = Mock()
    command = supervisor.supervise_case(returned, config, model=model, classifier_model=model)
    assert command.goto == "escalation"
    assert command.update["messaging_disabled"] is True
    assert "original_review_attempts" not in command.update
    assert not model.mock_calls


def test_supervisor_budget_failure_uses_fallback_without_model(case, config):
    case["model_calls"] = 10
    delta, _ = call(case, config)
    model = Mock()
    command = supervisor.supervise_case({**case, **delta}, config, model=model, classifier_model=model)
    assert command.goto == "persist"
    assert command.update["handoff"]["handoff_source"] == "deterministic_fallback"
    assert not model.mock_calls


def test_real_review_accepts_response_contract(case, config):
    delta, _ = call(case, config, draft())
    value = delta["response_draft"]
    model = model_with(dict(draft_id=value["draft_id"], outcome="resolved", grounded=True, complete=True,
                           correct_outcome=True, appropriate_format=True, grounding_findings=["Supported"],
                           completeness_findings=["Complete"], revision_reasons=[], unsupported_sentences=[]))
    command = supervisor.supervise_case({**case, **delta}, config, model=model, classifier_model=Mock())
    assert command.goto == "persist"
    assert command.update["approved_response"]["content"] == value["body"]


@pytest.mark.parametrize("field,value", [
    ("assignment", None), ("channel", "sms"), ("turn_id", ""),
    ("messaging_disabled", True), ("classification_status", "pending"),
    ("account_context", {}), ("model_calls", True), ("technical_attempts", []),
])
def test_invalid_inputs_stop_before_model(case, config, field, value):
    case[field] = value
    model = model_with(draft())
    with pytest.raises(ValueError):
        response.draft_response(case, config, model=model)
    assert not model.mock_calls


@pytest.mark.parametrize("field,value", [
    ("max_model_calls", True), ("max_model_calls", 0),
    ("call_timeout_seconds", 0), ("call_timeout_seconds", float("nan")),
    ("call_timeout_seconds", float("inf")), ("call_timeout_seconds", 10**1000),
])
def test_invalid_limits_stop_before_model(case, config, field, value):
    config["configurable"]["limits"][field] = value
    model = model_with(draft())
    with pytest.raises(ValueError):
        response.draft_response(case, config, model=model)
    assert not model.mock_calls


def test_wrong_thread_stops_before_model(case, config):
    config["configurable"]["thread_id"] = "other"
    model = model_with(draft())
    with pytest.raises(ValueError):
        response.draft_response(case, config, model=model)
    assert not model.mock_calls


def test_combined_claim_can_cite_sources_for_two_issues(case, config):
    case["issues"].append({**case["issues"][0], "issue_id": "issue2", "article_ids": ["article2"]})
    case["assignment"]["issue_ids"].append("issue2")
    case["retrieved_articles"].append({**case["retrieved_articles"][0], "issue_id": "issue2", "article_id": "article2"})
    value = draft()
    value["draft"]["addressed_issue_ids"] = ["issue1", "issue2"]
    value["draft"]["claim_to_evidence"][0].update(issue_ids=["issue1", "issue2"], source_ids=["article:1", "article:2"])
    result, _ = call(case, config, value)
    assert result["response_draft"]["addressed_issue_ids"] == ["issue1", "issue2"]


def test_article_for_other_issue_cannot_cover_claim(case, config):
    case["issues"].append({**case["issues"][0], "issue_id": "issue2", "article_ids": ["article2"]})
    case["assignment"]["issue_ids"].append("issue2")
    case["retrieved_articles"].append({**case["retrieved_articles"][0], "issue_id": "issue2", "article_id": "article2"})
    value = draft()
    value["draft"]["addressed_issue_ids"] = ["issue1", "issue2"]
    value["draft"]["claim_to_evidence"][0].update(issue_ids=["issue1", "issue2"], source_ids=["article:1"])
    result, model = call(case, config, value, value)
    assert result["response_draft"] is None
    assert result["trace"][-1]["payload"]["detail"] == "A claim lacks evidence for one of its issues."
    assert model.with_structured_output.return_value.invoke.call_count == 2


@pytest.mark.parametrize("update", [{"excerpt": "Not stored"}, {"score": 0.1}, {"score": float("nan")}])
def test_corrupt_accepted_article_fails_before_generation(case, config, update):
    case["retrieved_articles"][0].update(update)
    result, model = call(case, config)
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_evidence"
    assert not model.mock_calls


def test_blocked_customer_may_only_receive_permitted_acknowledgement(case, config):
    value = acknowledgement(case)
    case["account_context"]["is_blocked"] = True
    result, _ = call(case, config, value)
    assert result["response_draft"]["outcome"] == "escalated"


def test_unclassified_acknowledgement_needs_no_invented_issue(case, config):
    value = acknowledgement(case)
    case["issues"] = []
    case["classification_status"] = "failed"
    case["retrieved_articles"] = []
    case["handoff"]["issue_ids"] = ["turn1:unclassified"]
    value["draft"]["addressed_issue_ids"] = []
    result, _ = call(case, config, value)
    assert result["response_draft"] is not None


def test_missing_email_framing_is_filled_by_the_application(case, config):
    case["channel"] = "email"
    value = draft()
    value["draft"].update(channel="email", subject="Subject")
    result, model = call(case, config, value)
    drafted = result["response_draft"]
    assert model.with_structured_output.return_value.invoke.call_count == 1
    assert drafted["subject"] == "Subject" and drafted["greeting"] == "Hello," and drafted["closing"] == "Best regards,"
    assert drafted["body"] == "Hello,\n\nUse the cancellation form.\n\nBest regards,"
    assert drafted["framing_defaults"] == ["greeting", "closing"]
    value["draft"].update(subject=None, greeting="Dear Bob,", closing="Kind regards,")
    result, _ = call(case, config, value)
    assert result["response_draft"]["subject"] == "Your support request"
    assert result["response_draft"]["framing_defaults"] == ["subject"]


def test_http_server_error_retries_but_authentication_error_does_not(case, config):
    import httpx
    from openai import APIStatusError
    for status, count in ((503, 2), (401, 1)):
        error = APIStatusError("SECRET", response=httpx.Response(status, request=httpx.Request("POST", "https://invalid.test")), body=None)
        result, model = call(case, config, error, draft())
        assert model.with_structured_output.return_value.invoke.call_count == count
        assert "SECRET" not in json.dumps(result)


def test_model_setup_failure_is_sanitized_without_reserving_a_call(case, config):
    model = Mock()
    model.with_structured_output.side_effect = RuntimeError("SECRET")
    result = response.draft_response(case, config, model=model)
    assert result["model_calls"] == 0
    assert "SECRET" not in json.dumps(result)


def test_ambiguous_lookup_is_question_basis_not_a_fact(case, config):
    value = clarification(case)
    key = json.dumps(dict(tool_name="lookup_reservations", account_id="account1", customer_id="customer1",
                          lookup_mode="single", arguments={"reservation_id": None}))
    case["tool_results"] = [dict(turn_id="turn1", issue_id="issue1", lookup_key=key, tool_name="lookup_reservations",
        attempt=1, outcome="ambiguous", retryable=False, reason="Several matches", records=[
            dict(reservation_id="r1", user_id="customer1"), dict(reservation_id="r2", user_id="customer1")])]
    value["draft"]["clarification_questions"][0]["basis_source_ids"] = ["lookup:1"]
    result, _ = call(case, config, value)
    assert result["response_draft"]["source_catalog"]["lookup:1"]["kind"] == "lookup_question"
    value["draft"]["claim_to_evidence"] = [dict(claim=value["draft"]["body"], issue_ids=["issue1"], source_ids=["lookup:1"])]
    result, _ = call(case, config, value)
    assert result["response_draft"] is None


def test_real_redraft_assignment_preserves_subset_and_reasons(case, config):
    value = clarification(case)
    case["issues"].append({"issue_id": "issue2", "task": "Other issue"})
    first, _ = call(case, config, value)
    d = first["response_draft"]
    review_model = model_with(dict(draft_id=d["draft_id"], outcome="information_requested", grounded=True,
        complete=True, correct_outcome=True, appropriate_format=False,
        grounding_findings=["Supported question"], completeness_findings=["Assigned issue addressed"],
        revision_reasons=["Use a shorter question."], unsupported_sentences=[]))
    command = supervisor.supervise_case({**case, **first}, config, model=review_model, classifier_model=Mock())
    assert command.goto == "response"
    returned = {**case, **first, **command.update}
    assert "issue_ids" not in returned["assignment"]
    result, model = call(returned, config, value)
    assert result["response_draft"]["addressed_issue_ids"] == ["issue1"]
    assert result["response_draft"]["draft_id"] != d["draft_id"]
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[-1].content
    assert "Use a shorter question." in prompt


def test_redraft_uses_review_history_if_live_draft_was_cleared(case, config):
    first, _ = call(case, config, draft())
    case.update(original_review_attempts=1, response_draft=None, review_result=None,
                review_history=[dict(turn_id="turn1", stage="original", reviewed_draft=first["response_draft"],
                                     revision_reasons=["Clarify the steps."])])
    result, model = call(case, config, draft())
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[-1].content
    assert "Clarify the steps." in prompt
    assert result["response_draft"] is not None
