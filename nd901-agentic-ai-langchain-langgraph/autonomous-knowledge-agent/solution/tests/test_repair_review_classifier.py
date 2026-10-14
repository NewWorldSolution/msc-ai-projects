"""Review-prompt and classifier rules from the 2026-10-12 decisions; no provider calls."""

from unittest.mock import Mock

from agentic.agents import classifier, supervisor
from test_model_budget import ScriptedModel, assessment, case, make_model  # noqa: F401
from test_supervisor_review import assessment_result, model_with, review_case  # noqa: F401
from test_supervisor_routes import completed, config  # noqa: F401


def system_prompt(model):
    return model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[0].content


def test_review_prompt_states_chat_has_no_greeting_or_closing(review_case):
    model = model_with(assessment_result())
    supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    text = system_prompt(model)
    assert "has no greeting or closing; never reject a chat draft for lacking them" in text
    assert "Do not require it to state blocked status" in text
    assert "reject it if it states one" in text
    assert "An escalation receipt that states no business fact is grounded." in text


def classification(**issue_changes):
    issue = dict(category="subscription", task="Check subscription status", missing_information=[],
                 missing_information_reasons={}, needs_operations=False, required_business_facts=["none"])
    issue.update(issue_changes)
    return classifier.TicketClassification(issues=[classifier.IssueAssessment(**issue)],
                                           urgency="normal", urgency_reason="Routine request", complexity="simple")


def test_classifier_drops_none_from_required_business_facts(monkeypatch, case):
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=classification()))
    result = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert result["classification_status"] == "completed"
    assert result["issues"][0]["required_business_facts"] == []
    kept = classification(required_business_facts=["Current subscription status"])
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=kept))
    result = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert result["issues"][0]["required_business_facts"] == ["Current subscription status"]


def test_classifier_prompt_contains_merge_rule_and_examples():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = classification(required_business_facts=[])
    classifier.assess_request("Can you renew my passport?", "{}", model=model)
    text = system_prompt(model)
    assert "Identify each distinct request the customer made" in text
    assert "The same request in two wordings" in text
    assert "is a stated_preference, never an issue" in text
    assert "exactly one unsupported issue, never unsupported plus unknown" in text
    assert "Never write 'none' as a fact." in text
    assert "exactly one unknown issue, never unsupported" in text
    assert "whenever the answer depends on the customer's own records" in text


def issue(category, task):
    return classifier.IssueAssessment(category=category, task=task, missing_information=[],
                                      missing_information_reasons={}, needs_operations=False,
                                      required_business_facts=[])


def classify(monkeypatch, case, *issues):
    result = classifier.TicketClassification(issues=list(issues), urgency="normal",
                                             urgency_reason="Routine request", complexity="simple")
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=result))
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert delta["classification_status"] == "completed"
    return [(i["category"], i["task"], i["issue_id"]) for i in delta["issues"]]


def test_classifier_drops_hedging_unknown_beside_a_classified_issue(monkeypatch, case):
    kept = classify(monkeypatch, case, issue("account_access", "reset password"),
                    issue("unknown", "request for clarification on password reset process"))
    assert kept == [("account_access", "reset password", "turn-1:issue:1")]
    kept = classify(monkeypatch, case, issue("subscription", "inquire about current subscription status"),
                    issue("unknown", "clarification needed on subscription details"))
    assert [k[0] for k in kept] == ["subscription"]


def test_classifier_collapses_identical_task_text_keeping_the_first(monkeypatch, case):
    kept = classify(monkeypatch, case, issue("unsupported", "request to renew passport"),
                    issue("unknown", "Request  to renew passport"))
    assert [k[0] for k in kept] == ["unsupported"]
    kept = classify(monkeypatch, case, issue("unknown", "Help with CultPass"),
                    issue("unsupported", "help with cultpass"))
    assert [k[0] for k in kept] == ["unknown"]


def test_classifier_keeps_distinct_classified_issues_and_lone_unknowns(monkeypatch, case):
    kept = classify(monkeypatch, case, issue("account_access", "reset password"),
                    issue("subscription", "check subscription status"))
    assert [k[0] for k in kept] == ["account_access", "subscription"]
    kept = classify(monkeypatch, case, issue("unknown", "something about the app"),
                    issue("unknown", "something about the card"))
    assert [k[0] for k in kept] == ["unknown", "unknown"]


def receipt_review_case(review_case):
    review_case.update(outcome="escalated", response_stage="acknowledgement", acknowledgement_permitted=True,
                       handoff={"handoff_source": "generated", "message_id": "turn-1:handoff", "turn_id": "turn-1"})
    review_case["response_draft"].update(outcome="escalated", body="Your request has been passed to human support.",
                                         claim_to_evidence=[], clarification_questions=[])
    return review_case


def test_receipt_review_is_grounded_and_complete_by_construction(review_case):
    case = receipt_review_case(review_case)
    model = model_with(assessment_result(
        outcome="escalated", grounded=False, complete=False,
        grounding_findings=["No evidence for blocked status."], completeness_findings=["Does not explain."],
        revision_reasons=["Add evidence for blocked status.", "Explain the refusal."]))
    result = supervisor.review_response(case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "approved"
    assert result["findings"] == {"grounded": True, "complete": True, "appropriate_format": True, "correct_outcome": True}
    assert result["revision_reasons"] == []
    assert "Receipt with no claims: grounding satisfied by construction." in result["grounding_findings"]


def test_receipt_review_still_rejects_wrong_outcome_or_format(review_case):
    case = receipt_review_case(review_case)
    model = model_with(assessment_result(
        outcome="escalated", grounded=False, correct_outcome=False,
        revision_reasons=["Promises a call back within an hour."]))
    result = supervisor.review_response(case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "rejected"
    assert result["findings"]["grounded"] is True and result["findings"]["correct_outcome"] is False
    assert result["revision_reasons"] == ["Promises a call back within an hour."]


def test_receipt_override_needs_an_empty_claim_map(review_case):
    case = receipt_review_case(review_case)
    case["response_draft"]["claim_to_evidence"] = [{"claim": "x", "issue_ids": [], "source_ids": ["handoff"]}]
    model = model_with(assessment_result(outcome="escalated", grounded=False, revision_reasons=["Unsupported claim."]))
    result = supervisor.review_response(case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "rejected" and result["findings"]["grounded"] is False


def planning_stub(monkeypatch):
    planning_model = Mock()
    planning_model.invoke.return_value = {"issues": [
        {"issue_id": "turn-1:issue:1", "reservation_mode": None, "experience_mode": None}]}
    monkeypatch.setattr(ScriptedModel, "with_structured_output", lambda self, schema, **kwargs: planning_model)
    return planning_model


def test_subscription_issue_always_gets_the_record_lookup(monkeypatch, case, assessment, config):
    assessment.issues[0].category = "subscription"
    assessment.issues[0].task = "inquire about current subscription status"
    assessment.issues[0].required_business_facts = []
    assessment.issues[0].needs_operations = False
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=assessment))
    planning_model = planning_stub(monkeypatch)
    model = make_model()
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    result = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert result.goto == "operations"
    assert result.update["assignment"]["issue_ids"] == ["turn-1:issue:1"]
    assert result.update["issues"][0]["required_business_facts"] == [supervisor.SUBSCRIPTION_RECORD_FACT]
    planning_model.invoke.assert_called_once()


def test_subscription_issue_keeps_the_classifier_facts_when_present(monkeypatch, case, assessment, config):
    assessment.issues[0].category = "subscription"
    assessment.issues[0].required_business_facts = ["Payment status"]
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=assessment))
    planning_stub(monkeypatch)
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    result = supervisor.supervise_case(case, config, model=make_model(), classifier_model=make_model())
    assert result.goto == "operations"
    # why: Unchanged issues are not part of the update; the classifier fact stands as given.
    assert "issues" not in result.update
    assert case["issues"][0]["required_business_facts"] == ["Payment status"]


def test_how_to_issue_without_facts_still_goes_to_knowledge(monkeypatch, case, assessment, config):
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=assessment))
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    result = supervisor.supervise_case(case, config, model=make_model(), classifier_model=make_model())
    assert result.goto == "knowledge"
    assert "issues" not in result.update or result.update["issues"][0]["required_business_facts"] == []


def test_category_schema_description_defines_unknown_versus_unsupported():
    description = classifier.IssueAssessment.model_json_schema()["properties"]["category"]["description"]
    assert "A request with no stated problem is unknown, never unsupported." in description


def test_preference_copied_from_memory_is_dropped_not_fatal(monkeypatch, case):
    result = classifier.TicketClassification(
        issues=[issue("account_access", "Explain login recovery")], urgency="normal",
        urgency_reason="Routine request", complexity="simple",
        stated_preferences=["keep replies concise", "sign in"])
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=result))
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert delta["classification_status"] == "completed"
    assert delta["classification"]["stated_preferences"] == ["sign in"]
    assert delta["trace"][-1]["payload"]["dropped_preferences"] == 1
    assert delta["trace"][-1]["payload"]["failure_detail"] is None


def test_classifier_failure_detail_is_recorded_and_fed_to_the_repair(monkeypatch, case):
    broken = classifier.TicketClassification(
        issues=[issue("account_access", "   ")], urgency="normal",
        urgency_reason="Routine request", complexity="simple")
    assess = Mock(return_value=broken)
    monkeypatch.setattr(classifier, "assess_request", assess)
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert delta["classification_status"] == "failed"
    assert delta["trace"][-1]["payload"]["failure_detail"] == "Incomplete classification fields."
    assert assess.call_count == 2
    assert "Failed check: Incomplete classification fields." in assess.call_args_list[1].args[1]
    assert "Failed check" not in assess.call_args_list[0].args[1]


def test_classifier_prompt_has_the_vague_request_example():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = classification(required_business_facts=[])
    classifier.assess_request("I need help", "{}", model=model)
    assert "is one unknown issue with missing_information ['Description of the problem']" in system_prompt(model)


def test_library_parser_errors_never_leak_text_into_the_detail(monkeypatch, case):
    from langchain_core.exceptions import OutputParserException
    assess = Mock(side_effect=[OutputParserException("SECRET model text"), OutputParserException("SECRET again")])
    monkeypatch.setattr(classifier, "assess_request", assess)
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert delta["classification_status"] == "failed"
    assert delta["trace"][-1]["payload"]["failure_detail"] == "Structured output could not be parsed."
    import json
    assert "SECRET" not in json.dumps(delta)
    assert "SECRET" not in assess.call_args_list[1].args[1]


def test_same_request_in_two_wordings_keeps_the_classified_reading(monkeypatch, case):
    kept = classify(monkeypatch, case, issue("account_access", "reset password"),
                    issue("unsupported", "how to reset password"))
    assert [k[0] for k in kept] == ["account_access"]
    kept = classify(monkeypatch, case, issue("unsupported", "how to reset password"),
                    issue("account_access", "reset password"))
    assert [k[0] for k in kept] == ["account_access"]
    kept = classify(monkeypatch, case, issue("subscription", "check subscription status"),
                    issue("reservation", "book a seat for the status update talk"))
    assert [k[0] for k in kept] == ["subscription", "reservation"]


def test_review_prompt_accepts_found_lookup_records_as_grounding(review_case):
    model = model_with(assessment_result())
    supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    assert "A claim mapped to a lookup source whose record was found is grounded by that record" in system_prompt(model)


def test_classifier_schema_mismatch_detail_hides_model_chosen_keys():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as caught:
        classifier.TicketClassification.model_validate({
            "issues": [{"category": "unknown", "task": "x", "missing_information": ["a"],
                        "missing_information_reasons": {"MODEL_SECRET_KEY": 5}, "needs_operations": False,
                        "required_business_facts": []}],
            "urgency": "normal", "urgency_reason": "r", "complexity": "simple"})
    detail = classifier._classification_detail(caught.value)
    assert "MODEL_SECRET_KEY" not in detail
    assert detail.startswith("Schema mismatch: issues.0.missing_information_reasons.<unexpected-field>")


def test_two_distinct_classified_requests_are_never_merged_by_containment(monkeypatch, case):
    kept = classify(monkeypatch, case, issue("account_access", "reset password"),
                    issue("account_access", "change reset password email address"))
    assert [k[1] for k in kept] == ["reset password", "change reset password email address"]


def test_unknown_versus_unsupported_on_the_same_text_gets_one_repair_then_first_stands(monkeypatch, case):
    conflict = classifier.TicketClassification(
        issues=[issue("unsupported", "Help with CultPass"), issue("unknown", "help with cultpass")],
        urgency="normal", urgency_reason="Routine request", complexity="simple")
    resolved = classifier.TicketClassification(
        issues=[issue("unknown", "Help with CultPass")],
        urgency="normal", urgency_reason="Routine request", complexity="simple")
    assess = Mock(side_effect=[conflict, resolved])
    monkeypatch.setattr(classifier, "assess_request", assess)
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert assess.call_count == 2
    assert [i["category"] for i in delta["issues"]] == ["unknown"]
    assert "labelled both unknown and unsupported" in assess.call_args_list[1].args[1]
    assess = Mock(side_effect=[conflict, conflict])
    monkeypatch.setattr(classifier, "assess_request", assess)
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    assert delta["classification_status"] == "completed"
    assert [i["category"] for i in delta["issues"]] == ["unsupported"]


def test_every_supervisor_route_emits_a_routing_event(monkeypatch, case, assessment, config):
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=assessment))
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    result = supervisor.supervise_case(case, config, model=make_model(), classifier_model=make_model())
    events = [e for e in result.update.get("trace", []) if e["event_type"] == "routing_decision"]
    assert len(events) == 1
    assert events[0]["payload"]["destination"] == result.goto == "knowledge"
    assert events[0]["payload"]["issue_ids"] == ["turn-1:issue:1"]
    assert events[0]["node"] == "supervisor" and events[0]["turn_id"] == "turn-1"


def test_review_prompt_rejects_unstated_consequences(review_case):
    model = model_with(assessment_result())
    supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    assert "Reject a consequence, cause or explanation that no article or record states" in system_prompt(model)


def test_classifier_never_keeps_a_credential_as_missing_information(monkeypatch, case):
    result = classifier.TicketClassification(
        issues=[classifier.IssueAssessment(
            category="account_access", task="reset password",
            missing_information=["current password", "preferred contact time"],
            missing_information_reasons={"current password": "to verify", "preferred contact time": "to call back"},
            needs_operations=False, required_business_facts=["email verification status"])],
        urgency="normal", urgency_reason="Routine request", complexity="simple")
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=result))
    delta = classifier.classify_ticket(dict(case), model=Mock(), reserve_model_call=Mock())
    issue_record = delta["issues"][0]
    assert issue_record["missing_information"] == ["preferred contact time"]
    assert list(issue_record["missing_information_reasons"]) == ["preferred contact time"]
    # why: "email verification status" asks about a process, not for a secret, so it stays.
    assert issue_record["required_business_facts"] == ["email verification status"]
    assert delta["trace"][-1]["payload"]["dropped_credential_requests"] == 1


def test_needs_operations_without_a_named_fact_goes_to_knowledge(monkeypatch, case, assessment, config):
    assessment.issues[0].needs_operations = True
    assessment.issues[0].required_business_facts = []
    monkeypatch.setattr(classifier, "assess_request", Mock(return_value=assessment))
    completed(case, assessment, "classifier")
    case["model_calls"] = 1
    result = supervisor.supervise_case(case, config, model=make_model(), classifier_model=make_model())
    assert result.goto == "knowledge"


def test_unplannable_lookup_sends_the_case_to_knowledge_with_facts_marked_checked(case, assessment, config):
    completed(case, assessment, "operations")
    case["issues"][0]["required_business_facts"] = ["Current email address"]
    case["assignment"]["required_business_facts"] = {"turn-1:issue:1": ["Current email address"]}
    case["trace"].append({"turn_id": "turn-1", "node": "operations", "event_type": "operations_failed",
                          "payload": {"issue_ids": ["turn-1:issue:1"], "error": {
                              "issue_id": "turn-1:issue:1", "error_code": "operations_lookup_mode_unavailable",
                              "reason": "The requested lookup has no mode authorized by Supervisor."}}})
    result = supervisor.supervise_case(case, config, model=Mock(), classifier_model=Mock())
    assert result.goto == "knowledge"
    assert result.update["assignment"]["agent"] == "knowledge"
    assert result.update["assignment"]["checked_business_facts"] == {"turn-1:issue:1": ["Current email address"]}
    assert result.update.get("outcome") != "escalated"
    case["trace"][-1]["payload"]["error"]["error_code"] = "operations_tool_failure"
    assert supervisor.supervise_case(case, config, model=Mock(), classifier_model=Mock()).goto == "escalation"


def test_credential_filter_keeps_information_about_a_credential_process():
    from agentic.text_match import mentions_credential
    assert mentions_credential("Customer's current password")
    assert mentions_credential("account verification details")
    assert mentions_credential("the verification code sent by SMS")
    assert not mentions_credential("Error message on the password reset screen")
    assert not mentions_credential("Identity verification requirement for the venue")
    assert not mentions_credential("email verification status")


def test_listed_unsupported_sentences_fail_grounding_in_code(review_case):
    review_case.update(outcome="resolved")
    review_case["response_draft"].update(outcome="resolved", body="Your plan is cancelled. Cancelled plans lose premium access.")
    review_case["issues"][0].update(coverage="supported", article_ids=["article-1"])
    review_case["retrieved_articles"] = [{"article_id": "article-1", "excerpt": "Report the stored status.", "gate_pass": True}]
    review_case["response_draft"]["addressed_issue_ids"] = [review_case["issues"][0]["issue_id"]]
    model = model_with(assessment_result(outcome="resolved", unsupported_sentences=["Cancelled plans lose premium access."]))
    result = supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "rejected" and result["findings"]["grounded"] is False
    assert any("Cancelled plans lose premium access." in reason for reason in result["revision_reasons"])
    assert "Fill unsupported_sentences with every such sentence" in system_prompt(model)


def test_unsupported_sentences_do_not_override_receipt_rules(review_case):
    case = receipt_review_case(review_case)
    model = model_with(assessment_result(outcome="escalated", unsupported_sentences=["Your request was passed to support."]))
    result = supervisor.review_response(case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "approved"


def test_unsupported_sentences_never_fail_a_clarification(review_case):
    model = model_with(assessment_result(unsupported_sentences=["Which reservation do you mean?"]))
    result = supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    assert result["verdict"] == "approved" and result["findings"]["grounded"] is True


def test_listed_sentence_inside_a_mapped_claim_is_judged_by_the_claim_not_the_list(review_case):
    review_case.update(outcome="resolved")
    review_case["response_draft"].update(
        outcome="resolved", body="Your plan is active and you are on the premium tier. Cancelled plans lose premium access.",
        claim_to_evidence=[{"claim": "Your plan is active and you are on the premium tier.", "issue_ids": [], "source_ids": ["lookup:1"]}])
    review_case["issues"][0].update(coverage="supported", article_ids=["article-1"])
    review_case["retrieved_articles"] = [{"article_id": "article-1", "excerpt": "Report the stored status.", "gate_pass": True}]
    review_case["response_draft"]["addressed_issue_ids"] = [review_case["issues"][0]["issue_id"]]
    listed_mapped = assessment_result(outcome="resolved", unsupported_sentences=["Your plan is active and you are on the premium tier."])
    result = supervisor.review_response(review_case, model=model_with(listed_mapped), reserve_model_call=Mock())
    assert result["verdict"] == "approved"
    listed_prose = assessment_result(outcome="resolved", unsupported_sentences=["Cancelled plans lose premium access."])
    result = supervisor.review_response(review_case, model=model_with(listed_prose), reserve_model_call=Mock())
    assert result["verdict"] == "rejected" and result["findings"]["grounded"] is False
