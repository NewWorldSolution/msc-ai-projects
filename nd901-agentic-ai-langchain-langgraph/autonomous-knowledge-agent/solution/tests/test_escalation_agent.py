"""Escalation contracts with fake providers and in-memory persistence only."""

from copy import deepcopy
import json
import threading
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from agentic.agents import escalation, supervisor


@pytest.fixture
def case():
    return dict(ticket_id="ticket1", account_id="account1", customer_id="customer1", turn_id="turn1",
        outcome="escalated", escalation_reason="Guidance does not establish eligibility.",
        assignment={"agent":"escalation", "task":"Preserve the unresolved request."},
        preparation_result={"ticket_lookup_outcome":"success", "outcome":"ready"},
        channel="chat", messaging_disabled=False, acknowledgement_permitted=True,
        account_context={"user_id":"customer1", "is_blocked":False},
        messages=[HumanMessage(content="Can I cancel?",id="turn1")],
        classification_status="completed", classification={"urgency":"normal", "urgency_reason":"No deadline stated.","complexity":"simple"},
        issues=[dict(issue_id="issue1", task="Explain cancellation.", coverage="evidence_problem", article_ids=["a1"],
                     coverage_reason="Eligibility conditions missing.", missing_information=["Event date"],
                     required_business_facts=["Subscription tier"], initial_retrieval_attempted=True, targeted_retry_attempted=True)],
        retrieved_articles=[dict(article_id="a1",account_id="account1",turn_id="turn1",issue_id="issue1",
            title="Cancellation",content="Contact support.",excerpt="Contact support.",score=0.9,threshold=0.5,gate_pass=True)],
        tool_results=[], model_calls=0,technical_attempts={},trace=[],workflow_steps=0,
        original_review_attempts=0,acknowledgement_review_attempts=0,review_history=[])


@pytest.fixture
def config():
    return {"configurable":{"thread_id":"ticket1","limits":{"max_model_calls":8,"call_timeout_seconds":1.,"max_graph_steps":20}}}


def summary(**changes):
    return dict(highlight_refs=["request","reason","issues:0"],suggested_investigation=None) | changes


def model_with(*effects):
    model=Mock()
    model.with_structured_output.return_value.invoke.side_effect=list(effects)
    return model


def run(case,config,*effects):
    model=model_with(*effects)
    before=deepcopy(case)
    result=escalation.prepare_escalation(case,config,model=model)
    assert case==before
    assert set(result)=={"handoff","model_calls","technical_attempts","trace"}
    json.dumps(result,allow_nan=False)
    return result,model


def lookup(*,outcome="found",account="account1",customer="customer1",turn="turn1",attempt=1):
    key=json.dumps(dict(tool_name="lookup_subscription",account_id=account,customer_id=customer,
                        lookup_mode=None,arguments={"customer_id":customer}),sort_keys=True)
    return dict(tool_name="lookup_subscription",lookup_key=key,turn_id=turn,issue_id="issue1",attempt=attempt,
        outcome=outcome,retryable=outcome=="technical_failure",error_code="transient_database_error" if outcome=="technical_failure" else None,
        records=[{"subscription_id":"s1","user_id":customer,"tier":"basic"}] if outcome=="found" else [])


def test_generated_copies_facts_and_gaps_without_business_writes(case,config):
    case["tool_results"]=[lookup()]
    result,model=run(case,config,summary())
    handoff=result["handoff"]
    assert handoff["handoff_source"]=="generated"
    assert handoff["message_id"]=="turn1:handoff"
    assert handoff["verified_facts"][0]["records"][0]["subscription_id"]=="s1"
    assert {"Event date","Eligibility conditions missing."}.issubset(handoff["gaps"])
    assert "Subscription tier" not in handoff["gaps"]
    assert handoff["issues"][0]["required_business_facts"]==["Subscription tier"]
    assert handoff["classification"]["urgency_reason"]=="No deadline stated."
    assert handoff["highlights"][0]["data"]=="Can I cancel?"
    assert result["model_calls"]==1
    model.with_structured_output.assert_called_once_with(escalation.HandoffSummary)


@pytest.mark.parametrize("error",[TimeoutError("PRIVATE"),ConnectionError("PRIVATE"),RuntimeError("PRIVATE")])
def test_any_provider_failure_falls_back_without_retry(case,config,error):
    result,model=run(case,config,error,summary())
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"
    assert result["model_calls"]==1
    assert model.with_structured_output.return_value.invoke.call_count==1
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("bad",[{}, summary(highlight_refs=[]),summary(highlight_refs=["invented"]),
    summary(highlight_refs=["request","request"]),summary(suggested_investigation=" "),
    summary(ticket_id="foreign"),summary(highlight_refs="request")])
def test_invalid_output_uses_fallback_without_repair(case,config,bad):
    result,model=run(case,config,bad)
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"
    assert model.with_structured_output.return_value.invoke.call_count==1
    assert result["handoff"]["ticket_id"]=="ticket1"


def test_explicit_unavailable_model_uses_shared_helper(case,config,monkeypatch):
    helper=Mock(wraps=escalation.assemble_fallback_handoff)
    monkeypatch.setattr(escalation,"assemble_fallback_handoff",helper)
    result=escalation.prepare_escalation(case,config,model=None)
    assert result["model_calls"]==0
    assert result["handoff"]["generation_failure"]["error_code"]=="escalation_model_unavailable"
    helper.assert_called_once()


@pytest.mark.parametrize("event_type",["response_generation_failed","review_technical_failure","supervisor_coordination_failed"])
def test_known_failure_skips_generation(case,config,event_type):
    case["trace"]=[{"turn_id":"turn1","event_type":event_type,"payload":{"error_code":"model_failed"}}]
    result,model=run(case,config)
    assert result["model_calls"]==0 and not model.mock_calls
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"


def test_budget_exhaustion_has_no_extra_call(case,config):
    case["model_calls"]=8
    result,model=run(case,config)
    assert result["model_calls"]==8 and not model.mock_calls
    assert result["handoff"]["generation_failure"]["error_code"]=="escalation_model_budget_exhausted"


def test_consumed_attempt_cannot_be_reset_by_assignment(case,config):
    case["technical_attempts"]={"turn1:escalation:generation":1}
    result,model=run(case,config)
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"
    assert not model.mock_calls


@pytest.mark.parametrize("first_effect",[summary(),RuntimeError("private")])
def test_reentry_reuses_handoff_and_emits_no_duplicate_event(case,config,first_effect):
    first,_=run(case,config,first_effect)
    updated={**case,**first}
    second,model=run(updated,config)
    assert second["handoff"]==first["handoff"]
    assert second["trace"]==[] and not model.mock_calls


def test_reentry_rejects_changed_factual_base(case,config):
    first,_=run(case,config,summary())
    updated={**case,**first,"escalation_reason":"A newly established different blocker."}
    second,model=run(updated,config)
    assert second["handoff"]["escalation_reason"]==updated["escalation_reason"]
    assert second["handoff"]["handoff_source"]=="deterministic_fallback"
    assert not model.mock_calls


def test_old_lookup_history_is_ignored_without_disabling_generation(case,config):
    case["tool_results"]=[lookup(turn="old"),lookup()]
    result,model=run(case,config,summary())
    assert result["handoff"]["handoff_source"]=="generated"
    assert len(result["handoff"]["lookup_results"])==1


@pytest.mark.parametrize("entry",[lookup(account="foreign"),lookup(customer="foreign")])
def test_foreign_lookup_never_reaches_model_or_handoff(case,config,entry):
    case["tool_results"]=[entry]
    result,model=run(case,config)
    assert result["handoff"]["verified_facts"]==[]
    assert "foreign" not in json.dumps(result)
    assert not model.mock_calls


def test_newer_failure_supersedes_found_but_attempts_are_preserved(case,config):
    case["tool_results"]=[lookup(),lookup(outcome="technical_failure",attempt=2)]
    result,_=run(case,config,summary())
    assert result["handoff"]["verified_facts"]==[]
    assert len(result["handoff"]["attempted_lookups"])==2
    assert result["handoff"]["technical_errors"][0]["error_code"]=="transient_database_error"


def test_empty_lookup_is_retained_as_not_found(case,config):
    case["tool_results"]=[lookup(outcome="not_found")]
    result,_=run(case,config,summary())
    assert result["handoff"]["lookup_results"][0]["outcome"]=="not_found"
    assert result["handoff"]["technical_errors"]==[]


def test_unique_id_lookup_cannot_be_ambiguous(case,config):
    result=lookup(outcome="ambiguous")
    result["records"]=[{"subscription_id":"s1","user_id":"customer1"},{"subscription_id":"s2","user_id":"customer1"}]
    case["tool_results"]=[result]
    output,model=run(case,config)
    assert output["handoff"]["lookup_results"]==[]
    assert not model.mock_calls


def test_rejected_wording_is_excluded_from_prompt_and_handoff(case,config):
    case.update(response_draft={"body":"PRIVATE_REJECTED"},review_result={"reason":"PRIVATE_REJECTED"},
                review_history=[{"reviewed_draft":{"body":"PRIVATE_REJECTED"}}])
    result,model=run(case,config,summary())
    prompt=model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()[-1].content
    assert "PRIVATE_REJECTED" not in prompt and "PRIVATE_REJECTED" not in json.dumps(result)


def test_unclassified_case_has_stable_placeholder_and_unknown_assessment(case,config):
    case.update(issues=[],retrieved_articles=[],classification_status="failed",classification={"reason":"Unavailable"})
    result,_=run(case,config,summary(highlight_refs=["request","reason"]))
    assert result["handoff"]["issue_ids"]==["turn1:unclassified"]
    assert result["handoff"]["classification"]["urgency"] is None


def test_retrieval_gates_and_empty_scores_are_preserved(case,config):
    case["trace"]=[dict(turn_id="turn1",issue_id="issue1",node="knowledge",event_type="retrieval_gate",
        payload=dict(account_id="account1",article_id=None,score=None,threshold=0.5,gate_pass=False,reason="no_candidates"))]
    result,_=run(case,config,summary())
    gate=result["handoff"]["attempted_retrievals"][0]["payload"]
    assert gate["score"] is None and gate["gate_pass"] is False


def test_timeout_returns_fallback_and_discards_late_output(case,config):
    release=threading.Event()
    def slow(*args,**kwargs):
        release.wait(2)
        return summary()
    model=model_with()
    model.with_structured_output.return_value.invoke.side_effect=slow
    config["configurable"]["limits"]["call_timeout_seconds"]=0.02
    try:
        result=escalation.prepare_escalation(case,config,model=model)
        assert result["handoff"]["generation_failure"]["error_code"]=="escalation_call_timeout"
        assert result["model_calls"]==1
        assert model.with_structured_output.return_value.invoke.call_count==1
    finally:
        release.set()


@pytest.mark.parametrize("field,value",[("max_model_calls",True),("max_model_calls",0),("call_timeout_seconds",0),
    ("call_timeout_seconds",float("nan")),("call_timeout_seconds",10**1000)])
def test_invalid_limits_use_fallback(case,config,field,value):
    config["configurable"]["limits"][field]=value
    result,model=run(case,config)
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"
    assert not model.mock_calls


@pytest.mark.parametrize("field,value",[("outcome","resolved"),("assignment",None),("turn_id",""),
    ("issues",[{"issue_id":"i"},{"issue_id":"i"}]),("model_calls",True)])
def test_invalid_authorization_or_counters_rejected_before_call(case,config,field,value):
    case[field]=value
    model=model_with(summary())
    with pytest.raises(ValueError):
        escalation.prepare_escalation(case,config,model=model)
    assert not model.mock_calls


def test_generated_handoff_returns_to_supervisor_for_acknowledgement(case,config):
    result,_=run(case,config,summary())
    model=Mock()
    command=supervisor.supervise_case({**case,**result},config,model=model,classifier_model=model)
    assert command.goto=="response"
    assert command.update["response_stage"]=="acknowledgement"
    assert not model.mock_calls


def test_fallback_returns_to_supervisor_then_persist_without_model(case,config):
    result,_=run(case,config,RuntimeError("private"))
    model=Mock()
    command=supervisor.supervise_case({**case,**result},config,model=model,classifier_model=model)
    assert command.goto=="persist"
    assert command.update["messaging_disabled"] is True
    assert command.update["acknowledgement_permitted"] is False
    assert not model.mock_calls


@pytest.mark.parametrize("bad",[None,"not records",[{"subscription_id":"s2","user_id":"foreign"}]])
def test_invalid_newer_result_does_not_revive_prior_found(case,config,bad):
    later=lookup(attempt=2)
    later["records"]=bad
    case["tool_results"]=[lookup(),later]
    result,model=run(case,config)
    assert result["handoff"]["verified_facts"]==[]
    assert not model.mock_calls


def test_experience_lookup_does_not_require_customer(case,config):
    case["customer_id"]=None
    case["account_context"]={}
    key=json.dumps(dict(tool_name="lookup_experiences",account_id="account1",customer_id=None,
                        lookup_mode="single",arguments={"experience_id":"e1"}))
    case["tool_results"]=[dict(turn_id="turn1",issue_id="issue1",tool_name="lookup_experiences",lookup_key=key,
        attempt=1,outcome="found",retryable=False,records=[{"experience_id":"e1","title":"Museum"}])]
    result,_=run(case,config,summary())
    assert result["handoff"]["verified_facts"][0]["records"][0]["experience_id"]=="e1"


def test_nonserializable_evidence_is_excluded_without_stringifying(case,config):
    case["issues"][0]["missing_information"]=[object()]
    # `deepcopy(object())` has a different identity; compare the source field directly instead.
    original=case["issues"][0]["missing_information"][0]
    model=model_with()
    result=escalation.prepare_escalation(case,config,model=model)
    assert case["issues"][0]["missing_information"][0] is original
    assert result["handoff"]["handoff_source"]=="deterministic_fallback"
    assert not model.mock_calls
    json.dumps(result,allow_nan=False)


def test_suggestion_remains_separate_from_verified_facts(case,config):
    result,_=run(case,config,summary(suggested_investigation="Check the eligibility policy with the support lead."))
    assert result["handoff"]["suggested_investigation"].startswith("Check")
    assert result["handoff"]["verified_facts"]==[]


def test_model_factory_failure_consumes_no_invoke_but_no_repeat(case,config):
    model=Mock()
    model.with_structured_output.side_effect=RuntimeError("PRIVATE")
    result=escalation.prepare_escalation(case,config,model=model)
    assert result["model_calls"]==0
    assert result["technical_attempts"]["turn1:escalation:generation"]==1
    assert "PRIVATE" not in json.dumps(result)


def test_wrong_thread_is_rejected_before_model(case,config):
    config["configurable"]["thread_id"]="other"
    model=model_with(summary())
    with pytest.raises(ValueError):
        escalation.prepare_escalation(case,config,model=model)
    assert not model.mock_calls


def test_missing_request_is_explicit(case,config):
    case["messages"]=[]
    result,_=run(case,config,summary(highlight_refs=["reason"]))
    assert result["handoff"]["request"]==""
    assert result["handoff"]["request_status"]=="not_obtained"


def test_handoff_preserves_all_issues_even_with_subset_assignment(case,config):
    case["assignment"]["issue_ids"]=["issue1"]
    case["issues"].append(dict(issue_id="issue2",task="Check a second issue.",missing_information=["Booking date"]))
    result,_=run(case,config,summary())
    assert result["handoff"]["issue_ids"]==["issue1","issue2"]
    assert "Booking date" in result["handoff"]["gaps"]


@pytest.mark.parametrize("effect",[summary(),RuntimeError("private")])
def test_supervisor_and_real_persist_save_handoff_once(case,config,effect):
    from langgraph.store.memory import InMemoryStore
    from sqlalchemy import create_engine
    from data.models import udahub
    from agentic.persist import persist_case
    from utils import get_session

    engine=create_engine("sqlite://")
    udahub.Base.metadata.create_all(engine)
    try:
        with get_session(engine) as session:
            session.add(udahub.Account(account_id="account1",account_name="CultPass"))
            session.add(udahub.User(user_id="support1",account_id="account1",external_user_id="customer1",user_name="Ada"))
            session.add(udahub.Ticket(ticket_id="ticket1",account_id="account1",user_id="support1",channel="chat"))
            session.add(udahub.TicketMetadata(ticket_id="ticket1",status="open"))
        case["acknowledgement_permitted"]=False
        update,_=run(case,config,effect)
        current={**case,**update}
        model=Mock()
        command=supervisor.supervise_case(current,config,model=model,classifier_model=model)
        assert command.goto=="persist" and not model.mock_calls
        current.update(command.update)
        store=InMemoryStore()
        for _ in range(2):
            saved=persist_case(current,config,udahub_engine=engine,store=store)
            assert saved["persistence_result"]["saved"] is True
            assert "messages" not in saved
        with get_session(engine) as session:
            messages=session.query(udahub.TicketMessage).all()
            assert len(messages)==2
            handoff=session.get(udahub.TicketMessage,"turn1:handoff")
            assert handoff.role==udahub.RoleEnum.system
            assert json.loads(handoff.content)["message_type"]=="human_handoff"
            assert session.get(udahub.TicketMetadata,"ticket1").status=="escalated"
    finally:
        engine.dispose()


def test_unresolved_business_facts_are_reported_when_coverage_requires_them(case,config):
    case["issues"][0]["coverage"]="awaiting_business_facts"
    result,_=run(case,config,summary())
    assert "Subscription tier" in result["handoff"]["gaps"]


def test_supported_assessment_reason_is_not_reported_as_a_gap(case,config):
    case["issues"][0].update(coverage="supported",coverage_reason="Guidance is complete.",missing_information=[])
    result,_=run(case,config,summary())
    assert "Guidance is complete." not in result["handoff"]["gaps"]
