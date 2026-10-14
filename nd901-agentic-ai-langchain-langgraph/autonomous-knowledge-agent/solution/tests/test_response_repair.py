"""Repair, normalization and receipt checks for Response; fake models, no provider calls."""

from copy import deepcopy
import json

from test_response_agent import acknowledgement, call, case, config, draft, lookup  # noqa: F401

REPAIR_KEY = json.dumps(["turn1", "response", "original", 0, "repair"], separators=(",", ":"))


def variant(**changes):
    """Override keys of the shared draft helper, which cannot take duplicates."""
    value = draft()
    value["draft"].update(changes)
    return value


def invalid_claim():
    return variant(claim_to_evidence=[dict(claim="Absent wording", issue_ids=["issue1"], source_ids=["article:1"])])


def invoke_of(model):
    return model.with_structured_output.return_value.invoke


def last_feedback(model):
    return invoke_of(model).call_args.args[0].to_messages()[-1].content


def test_validation_failure_is_repaired_once_with_fixed_feedback(case, config):
    result, model = call(case, config, invalid_claim(), draft())
    assert invoke_of(model).call_count == 2
    feedback = last_feedback(model)
    assert "Claim text must be copied exactly from the draft." in feedback
    assert "['article:1', 'assignment']" in feedback
    assert "Assigned issue IDs: ['issue1']" in feedback
    assert "Absent wording" not in feedback
    first, second = (c.args[0].to_messages() for c in invoke_of(model).call_args_list)
    assert len(second) == len(first) + 1
    assert result["response_draft"]["body"] == "Use the cancellation form."
    assert result["model_calls"] == 2
    assert result["technical_attempts"][REPAIR_KEY] == 1


def test_second_validation_failure_is_terminal(case, config):
    second = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["invented"])])
    result, model = call(case, config, invalid_claim(), second)
    assert invoke_of(model).call_count == 2
    payload = result["trace"][-1]["payload"]
    assert payload["error_code"] == "invalid_response_output"
    assert payload["detail"] == "A claim cites a source key that is not in the catalog."
    assert result["response_draft"] is None and result["draft_blocker"] is None


def test_repair_counter_cannot_reset(case, config):
    case["technical_attempts"] = {REPAIR_KEY: 1}
    result, model = call(case, config, invalid_claim(), draft())
    assert invoke_of(model).call_count == 1
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_output"
    assert result["technical_attempts"][REPAIR_KEY] == 1


def test_transient_retry_and_repair_share_the_three_call_cap(case, config):
    result, model = call(case, config, ConnectionError("private"), invalid_claim(), draft())
    assert invoke_of(model).call_count == 3
    assert result["response_draft"] is not None
    result, model = call(case, config, ConnectionError("private"), invalid_claim(), invalid_claim(), draft())
    assert invoke_of(model).call_count == 3
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_output"
    assert "private" not in json.dumps(result)


def test_failure_payload_lists_available_sources_and_position(case, config):
    value = variant(claim_to_evidence=[
        dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article:1"]),
        dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["unknown:9"]),
    ])
    result, _ = call(case, config, value, value)
    payload = result["trace"][-1]["payload"]
    assert payload["available_sources"] == ["article:1", "assignment"]
    assert payload["position"] == 1
    assert payload["detail"] == "A claim cites a source key that is not in the catalog."
    assert "unknown:9" not in json.dumps(result)


def test_curly_quotes_and_double_spaces_do_not_fail_exact_text(case, config):
    body = "Tap 'Forgot Password' and  wait ten minutes."
    value = variant(body=body, claim_to_evidence=[dict(
        claim="tap ‘Forgot Password’ and wait ten minutes.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["body"] == body
    assert result["response_draft"]["claim_to_evidence"][0]["claim"].startswith("tap ‘")


def test_escalated_acknowledgement_with_claims_is_rejected(case, config):
    receipt = acknowledgement(case)
    with_claim = deepcopy(receipt)
    with_claim["draft"]["claim_to_evidence"] = [
        dict(claim="Your request requires human assistance.", issue_ids=["issue1"], source_ids=["handoff"])]
    result, model = call(case, config, with_claim, receipt)
    assert invoke_of(model).call_count == 2
    assert "An acknowledgement carries no claims or questions." in last_feedback(model)
    assert result["response_draft"]["outcome"] == "escalated"
    assert result["response_draft"]["claim_to_evidence"] == []


def test_successful_repair_records_the_repaired_check(case, config):
    result, _ = call(case, config, invalid_claim(), draft())
    assert result["trace"][-1]["event_type"] == "response_drafted"
    assert result["trace"][-1]["payload"]["repaired_check"] == "Claim text must be copied exactly from the draft."
    result, _ = call(case, config, draft())
    assert result["trace"][-1]["payload"]["repaired_check"] is None


def blocker():
    return {"draft": None, "blocker": {"issue_ids": ["issue1"], "reason": "Insufficient evidence.",
                                       "missing_evidence": ["A clear statement of the policy."]}}


def test_blocker_at_acknowledgement_stage_is_repaired_then_terminal(case, config):
    receipt = acknowledgement(case)
    result, model = call(case, config, blocker(), receipt)
    assert invoke_of(model).call_count == 2
    assert "A receipt cannot be blocked; return the acknowledgement draft." in last_feedback(model)
    assert result["response_draft"]["outcome"] == "escalated" and result["draft_blocker"] is None
    result, model = call(case, config, blocker(), blocker())
    assert invoke_of(model).call_count == 2
    assert result["draft_blocker"] is None and result["response_draft"] is None
    assert result["trace"][-1]["payload"]["error_code"] == "invalid_response_output"
    assert result["trace"][-1]["payload"]["detail"] == "A receipt cannot be blocked; return the acknowledgement draft."


def test_blocker_at_original_stage_is_still_a_blocker(case, config):
    result, model = call(case, config, blocker())
    assert invoke_of(model).call_count == 1
    assert result["draft_blocker"]["reason"] == "Insufficient evidence."


def test_claim_issue_id_failure_is_named_separately(case, config):
    value = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["other"], source_ids=["article:1"])])
    result, model = call(case, config, value, value)
    assert result["trace"][-1]["payload"]["detail"] == "Claim issue IDs must be among the assigned issue IDs."
    assert "Assigned issue IDs: ['issue1']" in last_feedback(model)


def test_article_uuid_alias_resolves_to_the_catalog_key(case, config):
    value = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article1"])])
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["claim_to_evidence"][0]["source_ids"] == ["article:1"]


def test_assignment_cited_as_evidence_is_named(case, config):
    value = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["assignment"])])
    result, _ = call(case, config, value, value)
    assert result["trace"][-1]["payload"]["detail"] == "A claim cites the assignment, which is not factual evidence."


def test_memory_reaches_the_prompt_without_ids_or_refs(case, config):
    case["customer_memory"] = [{"memory_id": "old:issue:1:resolved_issue", "memory_type": "resolved_issue",
                               "content": "Guidance provided for: password reset.", "outcome": "resolved",
                               "source_refs": ["old-turn", "old-article"], "ticket_id": "old-ticket", "turn_id": "old-turn"}]
    _, model = call(case, config, draft())
    prompt = invoke_of(model).call_args.args[0].to_messages()[-1].content
    assert "Guidance provided for: password reset." in prompt
    for leaked in ("old:issue:1:resolved_issue", "old-turn", "old-article", "old-ticket", "source_refs", "memory_id"):
        assert leaked not in prompt


def test_receipt_promise_sentences_are_removed_not_fatal(case, config):
    receipt = acknowledgement(case)
    promising = deepcopy(receipt)
    promising["draft"]["body"] = "Your request has been received. They will be in touch with you shortly."
    result, model = call(case, config, promising)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["body"] == "Your request has been received."
    assert result["response_draft"]["removed_receipt_sentences"] == 1
    only_promise = deepcopy(receipt)
    only_promise["draft"]["body"] = "They will be in touch with you shortly."
    result, model = call(case, config, only_promise, receipt)
    assert invoke_of(model).call_count == 2
    assert "nothing remained after removing such sentences" in last_feedback(model)
    assert result["response_draft"]["body"] == "Your request requires human assistance."


def test_promise_phrases_are_only_checked_on_receipts(case, config):
    body = "Use the cancellation form. The change applies soon."
    value = variant(body=body, claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["body"] == body


def test_claim_copied_from_the_article_is_diagnosed_without_leaking_text(case, config):
    value = variant(body="Please use the form to cancel.",
                    claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, model = call(case, config, value, value)
    payload = result["trace"][-1]["payload"]
    assert payload["detail"] == "Claim text must be copied exactly from the draft."
    assert payload["hint"] == "the text matches an article passage, not the draft body"
    assert "Diagnosis: the text matches an article passage, not the draft body." in last_feedback(model)
    case["customer_memory"] = [{"memory_type": "resolved_issue", "content": "Guidance provided for: cancellation."}]
    value = variant(body="Please use the form to cancel.",
                    claim_to_evidence=[dict(claim="Guidance provided for: cancellation.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, _ = call(case, config, value, value)
    assert result["trace"][-1]["payload"]["hint"] == "the text matches recalled memory, not the draft body"


def test_schema_descriptions_define_claim_and_source_fields():
    from agentic.agents import response as module
    props = module.ClaimEvidence.model_json_schema()["properties"]
    assert "copied character for character from your own draft body" in props["claim"]["description"]
    assert "article:1 or lookup:4" in props["source_ids"]["description"]


def test_schema_mismatch_detail_hides_model_chosen_keys():
    import pytest
    from pydantic import ValidationError
    from agentic.agents import response as module
    with pytest.raises(ValidationError) as caught:
        module.ResponseGeneration.model_validate({"draft": None, "blocker": None, "MODEL_SECRET_FIELD": 1})
    detail = module._validation_detail(caught.value)
    assert detail == "Schema mismatch: <unexpected-field> (extra_forbidden)"
    assert "MODEL_SECRET_FIELD" not in detail


def test_parser_exception_from_invoke_takes_the_repair_path(case, config):
    from langchain_core.exceptions import OutputParserException
    result, model = call(case, config, OutputParserException("SECRET model text"), draft())
    assert invoke_of(model).call_count == 2
    assert "Failed check: Structured output could not be parsed." in last_feedback(model)
    assert "SECRET" not in last_feedback(model) and "SECRET" not in json.dumps(result)
    assert result["response_draft"] is not None
    assert result["trace"][-1]["payload"]["repaired_check"] == "Structured output could not be parsed."


def test_successful_repair_keeps_the_hint_and_position(case, config):
    value = variant(body="Please use the form to cancel.",
                    claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article:1"])])
    good = variant(body="Please use the form to cancel.",
                   claim_to_evidence=[dict(claim="Please use the form to cancel.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, _ = call(case, config, value, good)
    payload = result["trace"][-1]["payload"]
    assert payload["repaired_check"] == "Claim text must be copied exactly from the draft."
    assert payload["repaired_hint"] == "the text matches an article passage, not the draft body"
    assert payload["repaired_position"] == 0


def test_article_alias_resolves_within_the_claims_issue_scope(case, config):
    case["issues"] = [{"issue_id": "issue1", "task": "Cancel", "coverage": "supported", "article_ids": ["article1"]},
                      {"issue_id": "issue2", "task": "Pause", "coverage": "supported", "article_ids": ["article1"]}]
    case["assignment"]["issue_ids"] = ["issue1", "issue2"]
    second = dict(case["retrieved_articles"][0], issue_id="issue2")
    case["retrieved_articles"].append(second)
    value = variant(addressed_issue_ids=["issue1", "issue2"], claim_to_evidence=[
        dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["article1"]),
        dict(claim="Use the cancellation form.", issue_ids=["issue2"], source_ids=["article1"]),
    ])
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    claims = result["response_draft"]["claim_to_evidence"]
    assert claims[0]["source_ids"] == ["article:1"] and claims[1]["source_ids"] == ["article:2"]


def test_receipt_sentence_stating_a_verified_account_value_is_removed(case, config):
    receipt = acknowledgement(case)
    case["account_context"]["is_blocked"] = True
    stating = deepcopy(receipt)
    stating["draft"]["body"] = "Your account is blocked. Your case has been passed to support."
    result, model = call(case, config, stating)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["body"] == "Your case has been passed to support."
    assert result["response_draft"]["removed_receipt_sentences"] == 1


def test_receipt_fact_guard_ignores_resolved_drafts_and_unblocked_accounts(case, config):
    receipt = acknowledgement(case)
    receipt["draft"]["body"] = "Your request about the blocked door has been passed to support."
    result, model = call(case, config, receipt)
    assert invoke_of(model).call_count == 1 and result["response_draft"] is not None


def test_paraphrased_claim_gets_an_overlap_diagnosis_without_text(case, config):
    value = variant(body="Please open the cancellation form and submit it.",
                    claim_to_evidence=[dict(claim="Open the cancellation form, then submit the form.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, model = call(case, config, value, value)
    hint = result["trace"][-1]["payload"]["hint"]
    assert hint.startswith("the text (8 words) is not in the draft body; word overlap ")
    assert "% with the draft body" in hint and "reads as a paraphrase of the draft body" in hint
    assert "cancellation" not in hint
    assert "Diagnosis: the text (8 words)" in last_feedback(model)


def test_article_backing_message_names_the_issue_and_the_fix(case, config):
    case["tool_results"] = [lookup()]
    value = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["lookup_subscription"])])
    result, model = call(case, config, value, value)
    detail = result["trace"][-1]["payload"]["detail"]
    assert detail.startswith("Every resolved issue requires an article-backed claim; for issue IDs issue1 add an article key")
    assert "add an article key (article:N)" in last_feedback(model)


def test_knowledge_reasoning_never_reaches_the_response_prompt(case, config):
    case["issues"][0].update(coverage_reason="REASONING sufficient guidance found", evidence_assessment={
        "relevant": True, "relevance_reason": "REASONING articles address the issue",
        "sufficiency_reason": "REASONING the articles contain clear steps"})
    _, model = call(case, config, draft())
    prompt = invoke_of(model).call_args.args[0].to_messages()[-1].content
    assert "REASONING" not in prompt
    assert '"task": "Explain cancellation"' in prompt and '"coverage": "supported"' in prompt


def test_claim_copied_from_the_knowledge_assessment_is_diagnosed(case, config):
    case["issues"][0]["evidence_assessment"] = {"sufficiency_reason": "The articles contain clear and actionable steps."}
    value = variant(claim_to_evidence=[dict(claim="The articles contain clear and actionable steps.", issue_ids=["issue1"], source_ids=["article:1"])])
    result, _ = call(case, config, value, value)
    assert result["trace"][-1]["payload"]["hint"] == "the text matches the knowledge assessment's reasoning, not the draft body"


def test_duplicate_memory_entries_collapse_in_the_prompt(case, config):
    note = {"memory_type": "resolved_issue", "content": "Guidance provided for: reset password.", "outcome": "resolved"}
    case["customer_memory"] = [dict(note, memory_id="a"), dict(note, memory_id="b"),
                               {"memory_type": "preference", "content": "keep replies concise", "outcome": "resolved"}]
    _, model = call(case, config, draft())
    prompt = invoke_of(model).call_args.args[0].to_messages()[-1].content
    assert prompt.count("Guidance provided for: reset password.") == 1
    assert prompt.count("keep replies concise") == 1


def test_clarification_claim_mapped_to_the_assignment_is_dropped_not_fatal(case, config):
    from test_response_agent import clarification
    value = clarification(case)
    value["draft"]["claim_to_evidence"] = [dict(claim="Thank you for reaching out.", issue_ids=["issue1"], source_ids=["assignment"])]
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    drafted = result["response_draft"]
    assert drafted["outcome"] == "information_requested"
    assert drafted["claim_to_evidence"] == [] and drafted["dropped_assignment_claims"] == 1


def test_resolved_claim_mapped_to_the_assignment_still_fails(case, config):
    value = variant(claim_to_evidence=[dict(claim="Use the cancellation form.", issue_ids=["issue1"], source_ids=["assignment"])])
    result, _ = call(case, config, value, value)
    assert result["trace"][-1]["payload"]["detail"] == "A claim cites the assignment, which is not factual evidence."


def test_response_prompt_forbids_unstated_consequences(case, config):
    _, model = call(case, config, draft())
    system = invoke_of(model).call_args.args[0].to_messages()[0].content
    assert "Do not add consequences, causes or explanations that no article states" in system


def test_shared_article_covers_both_issues_in_a_combined_claim(case, config):
    case["issues"] = [{"issue_id": "issue1", "task": "Cancel", "coverage": "supported", "article_ids": ["article1"]},
                      {"issue_id": "issue2", "task": "Pause", "coverage": "supported", "article_ids": ["article1"]}]
    case["assignment"]["issue_ids"] = ["issue1", "issue2"]
    case["retrieved_articles"].append(dict(case["retrieved_articles"][0], issue_id="issue2"))
    value = variant(addressed_issue_ids=["issue1", "issue2"], claim_to_evidence=[
        dict(claim="Use the cancellation form.", issue_ids=["issue1", "issue2"], source_ids=["article1"])])
    result, model = call(case, config, value)
    assert invoke_of(model).call_count == 1
    assert sorted(result["response_draft"]["claim_to_evidence"][0]["source_ids"]) == ["article:1", "article:2"]


def test_numeric_model_keys_are_hidden_in_schema_details():
    import pytest
    from pydantic import ValidationError
    from agentic.agents import response as module
    with pytest.raises(ValidationError) as caught:
        module.ResponseGeneration.model_validate({"draft": None, "blocker": None, "123456": 1})
    assert module._validation_detail(caught.value) == "Schema mismatch: <unexpected-field> (extra_forbidden)"


def test_receipt_subject_with_a_promise_is_replaced(case, config):
    case["channel"] = "email"
    receipt = acknowledgement(case)
    receipt["draft"].update(channel="email", subject="Your blocked account: we will contact you shortly",
                            greeting="Hello,", closing="Best regards,")
    case["account_context"]["is_blocked"] = True
    result, model = call(case, config, receipt)
    assert invoke_of(model).call_count == 1
    assert result["response_draft"]["subject"] == "Your support request"
    assert result["response_draft"]["removed_receipt_sentences"] == 1


def test_receipt_that_loses_its_acknowledgement_sentence_goes_to_repair(case, config):
    receipt = acknowledgement(case)
    case["account_context"]["is_blocked"] = True
    hollow = deepcopy(receipt)
    hollow["draft"]["body"] = "Your request about a blocked account has been received and passed to human support. Thank you for your patience."
    result, model = call(case, config, hollow, receipt)
    assert invoke_of(model).call_count == 2
    assert "must still say the request was received and passed to human support" in last_feedback(model)
    assert result["response_draft"]["body"] == "Your request requires human assistance."
