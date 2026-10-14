"""Knowledge citation check tolerates spacing and quote glyphs; fake models only."""

from test_knowledge_assessment import env, run, valid  # noqa: F401
from test_knowledge_context import case, config, lookup  # noqa: F401


def test_knowledge_citation_accepts_normalized_excerpt(env, case):
    for article in case["retrieved_articles"]:
        article["content"] = article["excerpt"] = "The customer's stored policy text."
    first = valid()
    first["citations"][0]["excerpt"] = "the customer's  stored policy text."
    second = valid("i2")
    second["citations"][0]["excerpt"] = "The customer’s stored policy text."
    result, calls = run(env, [first, second], case=case)
    assert calls == 2
    assert set(result["assessments"]) == {"i1", "i2"}
    assert not result["failures"]
    assert result["assessments"]["i2"]["citations"][0]["excerpt"] == "The customer’s stored policy text."


def test_facts_under_applies_without_a_lookup_go_to_repair(env):
    from copy import deepcopy
    from unittest.mock import Mock
    ns, original, config = env
    first = valid()
    first["required_business_facts"] = ["Customer's current subscription status and tier"]
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = [first, valid(), valid("i2")]
    delta = ns["retrieve_knowledge"](deepcopy(original), config, model=model)
    assert model.with_structured_output.return_value.invoke.call_count == 3
    repair = model.with_structured_output.return_value.invoke.call_args_list[1].args[0][-1].content
    assert "Facts listed under applicability applies are not established by a completed lookup" in repair
    coverage = [e for e in delta["trace"] if e["event_type"] == "knowledge_coverage" and e["issue_id"] == "i1"]
    assert coverage[-1]["payload"]["assessment"]["required_business_facts"] == []


def test_each_contradiction_has_its_own_message(env):
    gap = valid()
    gap["evidence_gap"] = "No cancellation policy."
    result, calls = run(env, [gap, gap, valid("i2")])
    assert calls == 3
    assert result["failures"] == {"i1": "invalid_knowledge_assessment"}
    details = [e["payload"].get("assessment_failure_detail") for e in result["working"]["trace"]
               if e.get("event_type") == "knowledge_coverage" and e.get("issue_id") == "i1"]
    assert details == ["Sufficient guidance cannot have an evidence gap."]
    needs = valid()
    needs["applicability"] = "missing_facts"
    result, _ = run(env, [needs, needs, valid("i2")])
    details = [e["payload"].get("assessment_failure_detail") for e in result["working"]["trace"]
               if e.get("event_type") == "knowledge_coverage" and e.get("issue_id") == "i1"]
    assert details == ["Applicability missing_facts requires the missing facts or details to be listed."]


def test_knowledge_repair_message_names_the_failed_check(env):
    from copy import deepcopy
    from unittest.mock import Mock
    ns, original, config = env
    gap = valid()
    gap["evidence_gap"] = "No cancellation policy."
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = [gap, valid(), valid("i2")]
    ns["retrieve_knowledge"](deepcopy(original), config, model=model)
    repair_messages = model.with_structured_output.return_value.invoke.call_args_list[1].args[0]
    assert "Failed check: Sufficient guidance cannot have an evidence gap." in repair_messages[-1].content
    assert "No cancellation policy." not in repair_messages[-1].content


def test_library_parser_errors_in_knowledge_record_only_a_fixed_detail(env):
    import json
    from langchain_core.exceptions import OutputParserException
    first = OutputParserException("Function MODEL_SECRET_FUNCTION_NAME arguments:")
    result, calls = run(env, [first, OutputParserException("MODEL_SECRET again"), valid("i2")])
    assert calls == 3
    assert result["failures"] == {"i1": "invalid_knowledge_assessment"}
    details = [e["payload"].get("assessment_failure_detail") for e in result["working"]["trace"]
               if e.get("event_type") == "knowledge_coverage" and e.get("issue_id") == "i1"]
    assert details == ["Structured output could not be parsed."]
    assert "MODEL_SECRET" not in json.dumps(result["working"]["trace"])


def test_credential_requirements_are_removed_and_guidance_applies(env):
    first = valid()
    first["applicability"] = "missing_facts"
    first["applicability_reason"] = "Needs identity verification first."
    first["required_business_facts"] = ["Customer's current password"]
    first["missing_information"] = ["account verification details"]
    first["missing_information_reasons"] = {"account verification details": "Needed to proceed"}
    result, calls = run(env, [first, valid("i2")])
    assert calls == 2 and not result["failures"]
    assessment = result["assessments"]["i1"]
    assert assessment["applicability"] == "applies"
    assert assessment["required_business_facts"] == [] and assessment["missing_information"] == []
    assert assessment["dropped_credential_requirements"] == 2
    assert result["assessments"]["i2"]["dropped_credential_requirements"] == 0


def test_only_facts_a_completed_lookup_was_asked_for_are_dropped_under_applies(env, case):
    case["issues"][0]["required_business_facts"] = ["Subscription status"]
    case["tool_results"] = [lookup(issue="i1")]
    echoed = valid()
    echoed["required_business_facts"] = ["subscription status"]
    result, calls = run(env, [echoed, valid("i2")], case=case)
    assert calls == 2 and not result["failures"]
    assert result["assessments"]["i1"]["required_business_facts"] == []
    unrelated = valid()
    unrelated["required_business_facts"] = ["Reservation status"]
    result, calls = run(env, [unrelated, unrelated, valid("i2")], case=case)
    assert calls == 3 and result["failures"] == {"i1": "invalid_knowledge_assessment"}
