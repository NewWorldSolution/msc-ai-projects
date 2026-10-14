"""Knowledge coverage, retrieval, source validation and Supervisor integration tests."""
import json
import time
from copy import deepcopy
from threading import Event
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from sqlalchemy import event
from sqlalchemy.orm import Session
from agentic.agents import knowledge
from agentic.agents.supervisor import _continue_specialist_work
from data.models import udahub
from test_knowledge_context import case, config, lookup
from test_knowledge_assessment import valid

@pytest.fixture
def env(case,config,monkeypatch):
    search=Mock(side_effect=AssertionError('Reassessment should not search'))
    monkeypatch.setattr(knowledge,'search_knowledge',search)
    return case,config,search

def run(env,effects,state=None):
    original,config,_=env
    state=original if state is None else state
    before=deepcopy(state)
    model=Mock()
    model.with_structured_output.return_value.invoke.side_effect=effects
    delta=knowledge.retrieve_knowledge(state,config,model=model)
    assert state==before
    assert set(delta)=={'issues','retrieved_articles','technical_attempts','model_calls','trace'}
    assert all(type(v) is int and v>=0 for v in delta['technical_attempts'].values())
    assert len({e['event_id'] for e in delta['trace']})==len(delta['trace'])
    json.dumps(delta,allow_nan=False)
    assert not ({'messages','outcome','assignment','handoff','approved_response'} & set(delta))
    return delta,model

def merge(state,delta):
    result=deepcopy(state)
    result.update(delta)
    result['trace']=state.get('trace',[])+delta['trace']
    return result

def article_result(issue='i1',score=0.9,empty=False,account='a1'):
    gate=dict(account_id=account,turn_id='t1',issue_id=issue,article_id=None if empty else 'article-'+issue,
        score=None if empty else score,threshold=0.5,gate_pass=False if empty else score>=0.5,
        reason='no_candidates' if empty else ('threshold_met' if score>=0.5 else 'below_threshold'),
        embedding_model='test',score_convention='higher_is_more_relevant')
    article=dict(article_id='article-'+issue,title='Policy',content='Stored policy text.',excerpt='Stored policy text.',
        score=score,threshold=0.5,gate_pass=score>=0.5,embedding_model='test',score_convention='higher_is_more_relevant')
    return dict(status='completed',account_id=account,turn_id='t1',issue_id=issue,
        articles=[] if empty else [article],gate_records=[gate],retryable=False,error_code=None)

def failed_search(issue='i1',retryable=False):
    return dict(status='technical_failure',account_id='a1',turn_id='t1',issue_id=issue,articles=[],gate_records=[],
        error_code='transient_retrieval_error' if retryable else 'source_content_mismatch',retryable=retryable)

def initial(env,ids=('i1',)):
    state,_,_=env
    state['assignment']['issue_ids']=list(ids)
    state['retrieved_articles']=[]
    for issue in state['issues']:
        issue['initial_retrieval_attempted']=False

@pytest.mark.parametrize('changes,targeted,expected,route',[
    ({},False,'supported','response'),
    ({'relevant':False,'sufficient':False,'citations':[]},False,'evidence_problem','escalation'),
    ({'consistent':False},False,'evidence_problem','escalation'),
    ({'applicability':'not_applicable'},False,'evidence_problem','escalation'),
    ({'applicability':'missing_facts','required_business_facts':['tier']},False,'awaiting_business_facts','operations'),
    ({'applicability':'missing_facts','missing_information':['date'],'missing_information_reasons':{'date':'Needed'}},False,'awaiting_business_facts','response'),
    ({'sufficient':False,'evidence_gap':'Exceptions'},False,'targeted_retry_needed','knowledge'),
    ({'sufficient':False,'evidence_gap':'Exceptions'},True,'evidence_problem','escalation'),
])
def test_coverage_and_supervisor_routes(env,changes,targeted,expected,route):
    env[0]['issues'][0]['targeted_retry_attempted']=targeted
    delta,model=run(env,[valid()|changes,valid('i2')])
    assert delta['issues'][0]['coverage']==expected
    assert delta['issues'][1]['coverage']=='supported'
    assert _continue_specialist_work(merge(env[0],delta))==route
    assert model.with_structured_output.return_value.invoke.call_count==2

@pytest.mark.parametrize('changes',[
    {'citations':[]},{'relevant':False},{'evidence_gap':'Missing'}, {'sufficient':False},
    {'applicability':'missing_facts'}, {'required_business_facts':['tier']},
    {'missing_information':['date'],'applicability':'missing_facts'},
    {'required_business_facts':[' '],'applicability':'missing_facts'},
    {'issue_id':'wrong'}, {'relevance_reason':' '},
    {'citations':[dict(article_id='article-i2',excerpt='Stored policy text.',reason='Wrong issue')]},
    {'citations':[dict(article_id='article-i1',excerpt='invented',reason='False')]},
])
def test_invalid_output_repaired_once(env,changes):
    delta,model=run(env,[valid()|changes,valid(),valid('i2')])
    assert delta['issues'][0]['coverage']=='supported'
    assert delta['model_calls']==3
    assert model.with_structured_output.return_value.invoke.call_count==3

def test_invalid_output_twice_is_failure(env):
    delta,model=run(env,[{}, {},valid('i2')])
    assert delta['issues'][0]['coverage']=='evidence_problem'
    assert delta['trace'][0]['payload']['assessment_failure']=='invalid_knowledge_assessment'
    assert delta['model_calls']==3

@pytest.mark.parametrize('effects,count',[
    ([ConnectionError('secret'),valid(),valid('i2')],3),
    ([TimeoutError('secret'),valid(),valid('i2')],3),
    ([ConnectionError('secret'),{},valid(),valid('i2')],4),
    ([ValueError('secret')],1),
    ([ConnectionError('secret'),ConnectionError('secret')],2),
])
def test_retry_budget_and_sanitization(env,effects,count):
    delta,model=run(env,effects)
    assert delta['model_calls']==count
    assert 'secret' not in json.dumps(delta)
    assert model.with_structured_output.return_value.invoke.call_count==count

def test_exhausted_model_budget(env):
    env[0]['model_calls']=env[1]['configurable']['limits']['max_model_calls']
    delta,model=run(env,[])
    model.with_structured_output.return_value.invoke.assert_not_called()
    assert all(i['coverage']=='evidence_problem' for i in delta['issues'])

def test_model_deadline_does_not_launch_overlapping_requests(env):
    release=Event()
    env[1]['configurable']['limits']['call_timeout_seconds']=0.01
    def slow(*args,**kwargs):
        release.wait(1)
        return valid()
    try:
        delta,model=run(env,slow)
        assert delta['model_calls']==1
        assert model.with_structured_output.return_value.invoke.call_count==1
        assert all(e['payload']['assessment_failure']=='knowledge_assessment_timeout' for e in delta['trace'])
        again,_=run(env,[],merge(env[0],delta))
        assert again['model_calls']==1
    finally:
        release.set()

@pytest.mark.parametrize('field,value',[
    ('assignment',None),('messages',[]),('account_id',''),('turn_id',''),('model_calls',True),
    ('technical_attempts',{'k':True}),('messaging_disabled',True),('classification_status','pending'),
    ('account_context',{'is_blocked':True}),
])
def test_invalid_state_before_search(env,field,value):
    env[0][field]=value
    model=Mock()
    with pytest.raises(ValueError):knowledge.retrieve_knowledge(env[0],env[1],model=model)
    env[2].assert_not_called()
    assert not model.mock_calls

@pytest.mark.parametrize('key,value',[('threshold',float('nan')),('candidate_count',True),('embedding_model',''),('score_convention','distance')])
def test_invalid_settings_before_search(env,key,value):
    env[1]['configurable']['retrieval_settings'][key]=value
    with pytest.raises(ValueError):run(env,[])
    env[2].assert_not_called()

def test_initial_search_scopes_and_traces(env):
    initial(env,('i1','i2'))
    env[0]['trace']=[{'event_id':'old'}]
    env[2].side_effect=[article_result(),article_result('i2')]
    delta,model=run(env,[valid(),valid('i2')])
    assert len(delta['retrieved_articles'])==2
    assert len(delta['trace'])==6
    assert all(e['event_id']!='old' for e in delta['trace'])
    for i,call in enumerate(env[2].call_args_list,1):
        ctx=call.kwargs['config']['configurable']
        assert (ctx['trusted_account_id'],ctx['turn_id'],ctx['issue_id'])==('a1','t1',f'i{i}')
    assert env[1]['configurable'].get('issue_id') is None

@pytest.mark.parametrize('empty,score',[(True,0.9),(False,0.1)])
def test_empty_and_below_threshold_are_not_technical_failure(env,empty,score):
    initial(env)
    env[2].side_effect=[article_result(empty=empty,score=score)]
    delta,_=run(env,[valid()|dict(relevant=False,sufficient=False,citations=[])])
    assert delta['retrieved_articles']==[]
    assert delta['issues'][0]['coverage']=='evidence_problem'
    assert delta['trace'][-1]['payload']['retrieval_failure'] is None
    assert delta['trace'][-1]['payload']['assessment_failure'] is None

@pytest.mark.parametrize('retryable', [False,True])
def test_retrieval_failure_allowance_and_continuation(env,retryable):
    initial(env,('i1','i2'))
    env[2].side_effect=([failed_search(retryable=retryable)]*(2 if retryable else 1))+[article_result('i2')]
    delta,model=run(env,[valid('i2')])
    assert env[2].call_count==(3 if retryable else 2)
    assert delta['issues'][0]['coverage']=='evidence_problem'
    assert delta['issues'][1]['coverage']=='supported'
    assert delta['model_calls']==1

def test_transient_retrieval_retry_succeeds(env):
    initial(env)
    env[2].side_effect=[failed_search(retryable=True),article_result()]
    delta,_=run(env,[valid()])
    assert env[2].call_count==2
    assert delta['issues'][0]['coverage']=='supported'

def test_targeted_then_reassessment_preserves_other_issue(env):
    env[0]['assignment'].update(issue_ids=['i1'],retrieval_mode='targeted')
    env[0]['issues'][0]['coverage_reason']='Missing exception'
    untouched=deepcopy(env[0]['issues'][1])
    env[2].side_effect=[article_result()]
    delta,_=run(env,[valid()])
    assert delta['issues'][0]['targeted_retry_attempted'] is True
    assert delta['issues'][1]==untouched
    assert len(delta['retrieved_articles'])==2
    again,_=run(env,[valid()],merge(env[0],delta))
    assert env[2].call_count==1
    assert again['issues'][0]['coverage']=='supported'

@pytest.mark.parametrize('change',[{'account_id':'foreign'},{'gate_pass':False},{'turn_id':'old'},{'excerpt':'invented'}])
def test_reused_evidence_rejected(env,change):
    env[0]['retrieved_articles'][0].update(change)
    with pytest.raises(ValueError):run(env,[])

def test_lookup_scope_and_latest_failure_in_actual_prompt(env):
    env[0]['tool_results']=[lookup(),lookup(attempt=2,outcome='technical_failure'),lookup(account='foreign'),lookup(turn='old')]
    seen=[]
    def invoke(messages,**kwargs):
        context=json.loads(messages[1].content.split('\n',1)[1]);seen.append(context)
        return valid(context['issue_id'])
    run(env,invoke)
    assert seen[0]['verified_lookup_results']==[]
    assert len(seen[0]['unresolved_lookup_results'])==1
    assert seen[0]['unresolved_lookup_results'][0]['attempt']==2

@pytest.fixture
def real_search(case,config):
    engine=config['configurable']['udahub_engine']
    udahub.Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(udahub.Knowledge(article_id='article-i1',account_id='a1',title='Policy',content='Stored policy text.',tags='test'))
        s.add(udahub.Knowledge(article_id='foreign',account_id='a2',title='Private',content='Secret.',tags='test'))
        s.commit()
    case['assignment']['issue_ids']=['i1']
    case['issues'][0]['initial_retrieval_attempted']=False
    case['retrieved_articles']=[]
    store=Mock()
    store.similarity_search_with_relevance_scores.return_value=[(Document(page_content='Stored policy text.',metadata={'article_id':'article-i1','account_id':'a1'}),0.9)]
    config['configurable']['knowledge_store']=store
    return case,config,store

def test_real_tool_reads_authoritative_sql_rows_only(real_search):
    state,config,store=real_search
    statements=[]
    def collect(conn,cursor,statement,parameters,context,executemany):statements.append((statement,parameters))
    event.listen(config['configurable']['udahub_engine'],'before_cursor_execute',collect)
    model=Mock();model.with_structured_output.return_value.invoke.return_value=valid()
    delta=knowledge.retrieve_knowledge(state,config,model=model)
    assert delta['issues'][0]['coverage']=='supported'
    assert delta['retrieved_articles'][0]['content']=='Stored policy text.'
    assert all(sql.lstrip().upper().startswith('SELECT') for sql,_ in statements)
    assert any('a1' in args for _,args in statements)
    assert store.similarity_search_with_relevance_scores.call_args.kwargs['filter']=={'account_id':'a1'}

@pytest.mark.parametrize('metadata,text',[
    ({'article_id':'foreign','account_id':'a2'},'Secret.'),
    ({'article_id':'foreign','account_id':'a1'},'Secret.'),
    ({'article_id':'article-i1','account_id':'a1'},'Fabricated.'),
])
def test_real_source_mismatch_is_failure_without_model(real_search,metadata,text):
    state,config,store=real_search
    store.similarity_search_with_relevance_scores.return_value=[(Document(page_content=text,metadata=metadata),0.9)]
    model=Mock()
    delta=knowledge.retrieve_knowledge(state,config,model=model)
    assert delta['issues'][0]['coverage']=='evidence_problem'
    assert delta['retrieved_articles']==[]
    assert not model.mock_calls

# Regression probes: expected behavior from the existing retry/timeout contract.
def test_terminal_source_failure_cannot_become_supported_on_reassignment(env):
    env[0]['assignment'].update(issue_ids=['i1'],retrieval_mode='targeted')
    env[0]['issues'][0]['coverage_reason']='Missing exception'
    env[2].side_effect=[failed_search()]
    first,_=run(env,[])
    assert first['issues'][0]['coverage']=='evidence_problem'
    second,model=run(env,[valid()],merge(env[0],first))
    assert second['issues'][0]['coverage']=='evidence_problem'
    assert not model.mock_calls


def test_configured_deadline_also_bounds_retrieval(real_search, monkeypatch):
    state, config, store = real_search
    config["configurable"]["limits"]["call_timeout_seconds"] = 0.01
    candidates = store.similarity_search_with_relevance_scores.return_value
    release, finished = Event(), Event()
    original_search = knowledge.search_knowledge

    def monitored_search(*args, **kwargs):
        try:
            return original_search(*args, **kwargs)
        finally:
            finished.set()

    def slow(*args, **kwargs):
        release.wait(0.3)
        return candidates

    monkeypatch.setattr(knowledge, "search_knowledge", monitored_search)
    store.similarity_search_with_relevance_scores.side_effect = slow
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = valid()
    try:
        start = time.monotonic()
        delta = knowledge.retrieve_knowledge(state, config, model=model)
        elapsed = time.monotonic() - start
        assert elapsed < 0.25, f"Retrieval ignored its configured deadline: {elapsed:.3f}s"
        assert delta["issues"][0]["coverage"] == "evidence_problem"
        assert delta["retrieved_articles"] == []
        assert not model.mock_calls
    finally:
        release.set()
        # Keep the in-memory database alive until the discarded worker finishes.
        assert finished.wait(1)


def test_retrieval_timeout_blocks_other_issues_and_reassignment(env):
    initial(env, ("i1", "i2"))
    env[1]["configurable"]["limits"]["call_timeout_seconds"] = 0.01
    release, finished = Event(), Event()

    def blocked_search(*args, **kwargs):
        try:
            release.wait(1)
            return article_result()
        finally:
            finished.set()

    env[2].side_effect = blocked_search
    try:
        first, model = run(env, [])
        assert env[2].call_count == 1
        assert not model.mock_calls
        assert first["model_calls"] == 0
        assert all(i["coverage"] == "evidence_problem" for i in first["issues"])
        second, model = run(env, [], merge(env[0], first))
        assert env[2].call_count == 1
        assert not model.mock_calls
        assert second["retrieved_articles"] == []
        # Both returned states remain unchanged when the late result arrives.
        before = deepcopy((first, second))
        release.set()
        assert finished.wait(1)
        assert (first, second) == before
    finally:
        release.set()
        assert finished.wait(1)


def test_retrieval_timeout_prevents_assessment_of_earlier_success(env):
    initial(env, ("i1", "i2"))
    env[2].side_effect = [article_result(), knowledge.KnowledgeCallTimedOut("timeout")]
    delta, model = run(env, [])
    assert env[2].call_count == 2
    assert not model.mock_calls
    assert all(i["coverage"] == "evidence_problem" for i in delta["issues"])
    assert delta["trace"][-2]["payload"]["assessment_failure"] == "knowledge_retrieval_timeout"


@pytest.mark.parametrize("status,retryable", [("technical_failure", False), ("technical_failure", True), ("missing_input", False)])
def test_initial_search_terminal_failure_survives_changed_assignment(env, status, retryable):
    initial(env)
    failure = failed_search(retryable=retryable)
    failure["status"] = status
    env[2].side_effect = [failure, failure] if retryable else [failure]
    first, _ = run(env, [])
    called = env[2].call_count
    state = merge(env[0], first)
    state["assignment"].update(task="Renamed work", retrieval_mode="targeted")
    second, model = run(env, [valid()], state)
    assert env[2].call_count == called
    assert not model.mock_calls
    assert second["issues"][0]["coverage"] == "evidence_problem"


def test_search_failure_marker_does_not_block_new_turn(env):
    from langchain_core.messages import HumanMessage
    initial(env)
    env[2].side_effect = [failed_search()]
    first, _ = run(env, [])
    state = merge(env[0], first)
    state["turn_id"] = "t2"
    state["messages"].append(HumanMessage(content="New request", id="t2"))
    state["issues"][0]["initial_retrieval_attempted"] = False
    result = article_result()
    result["turn_id"] = "t2"
    result["gate_records"][0]["turn_id"] = "t2"
    env[2].side_effect = [result]
    second, _ = run(env, [valid()], state)
    assert second["issues"][0]["coverage"] == "supported"


def test_local_timeout_race_is_not_treated_as_provider_failure(monkeypatch):
    # The Future completes successfully between result(timeout) raising and done().
    future = Mock()
    future.result.side_effect = TimeoutError()
    future.done.return_value = True
    future.exception.return_value = None
    executor = Mock()
    executor.submit.return_value = future
    monkeypatch.setattr(knowledge, "ThreadPoolExecutor", Mock(return_value=executor))
    with pytest.raises(knowledge.KnowledgeCallTimedOut):
        knowledge.invoke_knowledge_search("query", {}, 0.01)
    with pytest.raises(knowledge.KnowledgeCallTimedOut):
        knowledge.invoke_knowledge_assessment(Mock(), [], {}, 0.01)
