"""Assessment and retry tests using the real Knowledge function and returned delta."""
from copy import deepcopy
from unittest.mock import Mock
from threading import Event
import pytest
from agentic.agents import knowledge

from test_knowledge_context import case, config

@pytest.fixture
def env(case, config, monkeypatch):
    monkeypatch.setattr(knowledge, "search_knowledge", Mock(side_effect=AssertionError("No search allowed")))
    return {
        "retrieve_knowledge": knowledge.retrieve_knowledge,
        "invoke_knowledge_assessment": knowledge.invoke_knowledge_assessment,
        "KnowledgeCallTimedOut": knowledge.KnowledgeCallTimedOut,
    }, case, config


def observe(state, delta):
    """Derive assertions from public state updates and coverage trace events."""
    working = deepcopy(state)
    working.update(delta)
    working["trace"] = state.get("trace", []) + delta["trace"]
    assessments = {}
    failures = {}
    for event in delta["trace"]:
        if event["event_type"] != "knowledge_coverage":
            continue
        payload = event["payload"]
        if payload["assessment"] is not None:
            assessments[event["issue_id"]] = payload["assessment"]
        if payload["assessment_failure"] is not None:
            failures[event["issue_id"]] = payload["assessment_failure"]
    return {"assessments": assessments, "failures": failures, "working": working}

def valid(issue='i1'):
    return dict(issue_id=issue, relevant=True, relevance_reason='Policy applies', sufficient=True,
        sufficiency_reason='Complete guidance', applicability='applies', applicability_reason='Facts established',
        consistent=True, consistency_reason='No conflict', citations=[dict(article_id='article-'+issue,
        excerpt='Stored policy text.',reason='Supports issue')],required_business_facts=[],missing_information=[],
        missing_information_reasons={},evidence_gap='')

def run(env, effects, case=None):
    ns, original, config = env
    state = deepcopy(original if case is None else case)
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = effects
    delta = ns['retrieve_knowledge'](state,config,model=model)
    result = observe(state, delta)
    assert state == (original if case is None else case)
    return result, model.with_structured_output.return_value.invoke.call_count

@pytest.mark.parametrize('first', [[],[ConnectionError('private')],[{}],[ConnectionError('private'),{}],[{},ConnectionError('private')]])
def test_success_retry_and_repair(env,first):
    result,calls=run(env,first+[valid(),valid('i2')])
    assert set(result['assessments']) == {'i1','i2'}
    assert not result['failures']
    assert calls == len(first)+2
    assert result['working']['model_calls'] == calls

def test_terminal_error_sanitized_stops(env):
    result,calls=run(env,[ValueError('secret database URL')])
    assert calls==1
    assert result['failures']==dict(i1='knowledge_assessment_call_failed',i2='knowledge_assessment_call_failed')
    assert 'secret' not in str(result)

def test_two_bad_outputs_stop_issue(env):
    result,calls=run(env,[{}, {}, valid('i2')])
    assert calls==3
    assert result['failures']=={'i1':'invalid_knowledge_assessment'}

def test_budget_stops_next_issue(env):
    env[2]['configurable']['limits']['max_model_calls']=1
    result,calls=run(env,[valid()])
    assert calls==1
    assert result['failures']=={'i2':'knowledge_model_budget_exhausted'}

def test_bad_citation_repaired(env):
    wrong=valid()
    wrong['citations'][0]['excerpt']='invented'
    result,calls=run(env,[wrong,valid(),valid('i2')])
    assert calls==3 and len(result['assessments'])==2

def test_timeout_does_not_retry(env):
    result,calls=run(env,[env[0]['KnowledgeCallTimedOut']('timeout')])
    assert calls==1
    assert set(result['failures'].values())=={'knowledge_assessment_timeout'}

def test_real_wait_timeout_discards_late_result(env):
    release=Event()
    entered=Event()
    model=Mock()
    def blocked(*args,**kwargs):
        entered.set()
        release.wait(1)
        return valid()
    model.invoke.side_effect=blocked
    try:
        with pytest.raises(env[0]['KnowledgeCallTimedOut']):
            env[0]['invoke_knowledge_assessment'](model,[],{},0.01)
        assert entered.is_set() and model.invoke.call_count==1
    finally:
        release.set()

def test_non_retryable_failure_cannot_be_retried_by_reassignment(env):
    result,calls=run(env,[ValueError('permanent')])
    again,calls=run(env,[valid(),valid('i2')],case=result['working'])
    assert calls == 0, 'Reassigning identical evidence retried a terminal failure'

@pytest.mark.parametrize("effects,error", [
    ([ValueError("private")], "knowledge_assessment_call_failed"),
    ([ConnectionError("private"), ConnectionError("private")], "knowledge_assessment_call_failed"),
    ([{}, {}], "invalid_knowledge_assessment"),
])
def test_terminal_operation_survives_reassignment(env, effects, error):
    env[1]["assignment"]["issue_ids"] = ["i1"]
    first, first_calls = run(env, effects)
    assert first["failures"] == {"i1": error}
    first["working"]["assignment"]["task"] = "Reworded assignment, identical evidence"
    second, calls = run(env, [valid()], case=first["working"])
    assert calls == 0
    assert second["failures"] == first["failures"]
    assert second["working"]["model_calls"] == first_calls


def test_timeout_stays_stopped_even_with_changed_evidence(env):
    first, _ = run(env, [env[0]["KnowledgeCallTimedOut"]("timeout")])
    first["working"]["account_context"]["tier"] = "updated"
    second, calls = run(env, [valid(), valid("i2")], case=first["working"])
    assert calls == 0
    assert set(second["failures"].values()) == {"knowledge_assessment_timeout"}


def test_new_evidence_allows_reassessment_after_invalid_output(env):
    env[1]["assignment"]["issue_ids"] = ["i1"]
    first, _ = run(env, [{}, {}])
    first["working"]["account_context"]["tier"] = "verified premium"
    second, calls = run(env, [valid()], case=first["working"])
    assert calls == 1
    assert set(second["assessments"]) == {"i1"}


def test_terminal_marker_is_scoped_to_turn(env):
    from langchain_core.messages import HumanMessage
    first, _ = run(env, [ValueError("private")])
    state = first["working"]
    state["turn_id"] = "t2"
    state["messages"].append(HumanMessage(content="Next request", id="t2"))
    state["model_calls"] = 0
    for article in state["retrieved_articles"]:
        article["turn_id"] = "t2"
    second, calls = run(env, [valid(), valid("i2")], case=state)
    assert calls == 2
    assert not second["failures"]


def test_provider_timeout_can_retry_after_request_has_finished(env):
    result, calls = run(env, [TimeoutError("provider finished"), valid(), valid("i2")])
    assert calls == 3
    assert len(result["assessments"]) == 2


def test_setup_failure_is_terminal_without_spending_model_budget(env):
    ns, state, config = env
    model = Mock()
    model.with_structured_output.side_effect = ValueError("unsupported schema")
    first = observe(state, ns["retrieve_knowledge"](state, config, model=model))
    assert first["working"]["model_calls"] == 0
    second, calls = run(env, [valid(), valid("i2")], case=first["working"])
    assert calls == 0
    assert second["failures"] == first["failures"]
