"""Check scoped evidence in prompts produced by the real Knowledge function."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from agentic.agents import knowledge


@pytest.fixture
def prepare_contexts(monkeypatch):
    tools = {"search_knowledge": Mock(side_effect=AssertionError("Reassessment must not search"))}

    def prepare(state, config, *, model):
        contexts = {}

        def assess(messages, **kwargs):
            # Capture the real serialized prompt, not function-local variables.
            context = json.loads(messages[1].content.split("\n", 1)[1])
            contexts[context["issue_id"]] = context
            return dict(
                issue_id=context["issue_id"], relevant=False,
                relevance_reason="No relevant guidance in this test assessment.",
                sufficient=False, sufficiency_reason="No sufficient guidance.",
                applicability="not_applicable", applicability_reason="No applicable guidance.",
                consistent=True, consistency_reason="No conflict assessed.", citations=[],
                required_business_facts=[], missing_information=[],
                missing_information_reasons={}, evidence_gap="",
            )

        monkeypatch.setattr(knowledge, "search_knowledge", tools["search_knowledge"])
        model.with_structured_output.return_value.invoke.side_effect = assess
        delta = knowledge.retrieve_knowledge(state, config, model=model)
        return {"contexts": contexts, "delta": delta}

    tools["retrieve_knowledge"] = prepare
    return tools


@pytest.fixture
def case():
    return dict(assignment={"agent":"knowledge","task":"Assess each issue.","issue_ids":["i1","i2"]},
        issues=[dict(issue_id=i,task=f"Task {i}",initial_retrieval_attempted=True,targeted_retry_attempted=False) for i in ("i1","i2")],
        preparation_result={"outcome":"ready"},classification_status="completed",messaging_disabled=False,
        account_context={"is_blocked":False,"user_id":"c1"},account_id="a1",customer_id="c1",turn_id="t1",
        messages=[HumanMessage(content="Old request",id="old"),HumanMessage(content="Current request",id="t1")],
        model_calls=0,technical_attempts={},retrieved_articles=[dict(article_id=f"article-{i}",account_id="a1",turn_id="t1",issue_id=i,
            title="Policy",content="Stored policy text.",excerpt="Stored policy text.",score=0.9,threshold=0.5,
            gate_pass=True,embedding_model="test",score_convention="higher_is_more_relevant") for i in ("i1","i2")],
        tool_results=[],trace=[])


@pytest.fixture
def config():
    engine=create_engine("sqlite://", poolclass=StaticPool,
                         connect_args={"check_same_thread": False})
    yield {"configurable":dict(knowledge_store=Mock(),udahub_engine=engine,retrieval_settings=dict(threshold=0.5,
        candidate_count=3,embedding_model="test",score_convention="higher_is_more_relevant"),
        limits=dict(max_model_calls=5,max_graph_steps=10,call_timeout_seconds=1.))}
    engine.dispose()


def lookup(*,account="a1",customer="c1",issue="i1",turn="t1",attempt=1,outcome="found",tool="lookup_subscription",mode=None):
    import json
    args={"customer_id":customer} if tool=="lookup_subscription" else {"experience_id":"e1"}
    key=json.dumps(dict(account_id=account,customer_id=customer,tool_name=tool,lookup_mode=mode,arguments=args),sort_keys=True)
    row={"subscription_id":"s1","user_id":customer,"tier":"basic"} if tool=="lookup_subscription" else {"experience_id":"e1","title":"Museum"}
    return dict(tool_name=tool,lookup_key=key,turn_id=turn,issue_id=issue,attempt=attempt,outcome=outcome,
        records=[row] if outcome=="found" else [],retryable=outcome=="technical_failure",
        error_code="transient_database_error" if outcome=="technical_failure" else None)


def run(prepare_contexts,case,config):
    before=deepcopy(case)
    model=Mock()
    result=prepare_contexts["retrieve_knowledge"](case,config,model=model)
    assert case==before
    assert model.with_structured_output.return_value.invoke.call_count == len(result["contexts"])
    assert result["delta"]["model_calls"] == case["model_calls"] + len(result["contexts"])
    return result["contexts"]


def test_each_issue_keeps_its_own_request_task_and_sources(prepare_contexts,case,config):
    case["tool_results"]=[lookup(issue="i1"),lookup(issue="i2")]
    contexts=run(prepare_contexts,case,config)
    assert set(contexts)=={"i1","i2"}
    for issue_id,context in contexts.items():
        assert context["customer_request"]=="Current request"
        assert context["issue"]["task"]==f"Task {issue_id}"
        assert context["assignment"]["issue_ids"]==[issue_id]
        assert {a["issue_id"] for a in context["articles"]}=={issue_id}
        assert {r["issue_id"] for r in context["verified_lookup_results"]}=={issue_id}
    contexts["i1"]["account_context"]["user_id"]="changed"
    assert contexts["i2"]["account_context"]["user_id"]=="c1"
    assert case["account_context"]["user_id"]=="c1"


@pytest.mark.parametrize("entry",[lookup(account="foreign"),lookup(customer="foreign"),lookup(turn="old"),lookup(issue="other")])
def test_foreign_old_or_unassigned_lookup_cannot_enter_context(prepare_contexts,case,config,entry):
    case["tool_results"]=[entry]
    contexts=run(prepare_contexts,case,config)
    assert not contexts["i1"]["verified_lookup_results"]
    assert not contexts["i1"]["unresolved_lookup_results"]


@pytest.mark.parametrize("reverse",[False,True])
def test_newer_failure_removes_old_fact_regardless_of_list_order(prepare_contexts,case,config,reverse):
    values=[lookup(),lookup(outcome="technical_failure",attempt=2)]
    case["tool_results"]=values[::-1] if reverse else values
    contexts=run(prepare_contexts,case,config)
    assert contexts["i1"]["verified_lookup_results"]==[]
    assert contexts["i1"]["unresolved_lookup_results"][0]["outcome"]=="technical_failure"


def test_not_found_remains_a_completed_lookup(prepare_contexts,case,config):
    case["tool_results"]=[lookup(outcome="not_found")]
    contexts=run(prepare_contexts,case,config)
    assert contexts["i1"]["verified_lookup_results"][0]["outcome"]=="not_found"
    assert contexts["i1"]["unresolved_lookup_results"]==[]


def test_experience_evidence_needs_no_customer_id(prepare_contexts,case,config):
    case["customer_id"]=None
    case["account_context"]={"is_blocked":False}
    case["tool_results"]=[lookup(tool="lookup_experiences",customer=None,mode="single")]
    contexts=run(prepare_contexts,case,config)
    assert contexts["i1"]["verified_lookup_results"][0]["records"][0]["experience_id"]=="e1"


@pytest.mark.parametrize("update",[{"records":"bad"},{"attempt":True},{"retryable":True},{"records":[]},
    {"records":[{"subscription_id":"s1","user_id":"foreign"}]},{"records":[{"user_id":"c1"}]}])
def test_invalid_latest_lookup_is_rejected(prepare_contexts,case,config,update):
    later=lookup(attempt=2)
    later.update(update)
    case["tool_results"]=[lookup(),later]
    with pytest.raises(ValueError):
        run(prepare_contexts,case,config)


@pytest.mark.parametrize("update",[{"excerpt":"Fabricated"},{"score":float("nan")},{"score":0.1},
    {"threshold":0.7},{"embedding_model":"other"},{"score_convention":"wrong"}])
def test_reused_articles_are_revalidated(prepare_contexts,case,config,update):
    case["retrieved_articles"][0].update(update)
    with pytest.raises(ValueError):
        run(prepare_contexts,case,config)


def test_terminal_search_failure_skips_only_affected_issue(prepare_contexts,case,config):
    case["issues"][0]["initial_retrieval_attempted"]=False
    prepare_contexts["search_knowledge"]=Mock(return_value=dict(status="technical_failure",account_id="a1",turn_id="t1",
        issue_id="i1",retryable=False,error_code="retrieval_error",articles=[],gate_records=[]))
    contexts=run(prepare_contexts,case,config)
    assert set(contexts)=={"i2"}


def test_result_extras_are_not_exposed(prepare_contexts,case,config):
    entry=lookup()
    entry["raw_exception"]="PRIVATE"
    case["tool_results"]=[entry]
    contexts=run(prepare_contexts,case,config)
    assert "raw_exception" not in contexts["i1"]["verified_lookup_results"][0]


def test_nonfinite_record_values_are_rejected(prepare_contexts,case,config):
    entry=lookup()
    entry["records"][0]["quota"]=float("nan")
    case["tool_results"]=[entry]
    with pytest.raises(ValueError):
        run(prepare_contexts,case,config)
