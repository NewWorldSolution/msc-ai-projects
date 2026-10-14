"""Actual Classifier node routing and shared reservations without providers."""
from copy import deepcopy
from unittest.mock import Mock
import pytest
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine
from agentic import workflow
from agentic.agents import classifier
from test_model_budget import case, assessment, ScriptedModel


class UnusedVectorStore(VectorStore):
    def add_texts(self, texts, metadatas=None, **kwargs):
        raise AssertionError('Unexpected retrieval')
    def similarity_search(self, query, k=4, **kwargs):
        raise AssertionError('Unexpected retrieval')
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        return cls()


@pytest.fixture
def factory(monkeypatch, case):
    engine = create_engine('sqlite://')
    model = ScriptedModel(responses=[AIMessage(content='unused')])
    monkeypatch.setattr(workflow, 'prepare_case', lambda state, config, **kwargs: deepcopy(case))
    saved = Mock(return_value={'persistence_result': {'saved': True}})
    monkeypatch.setattr(workflow, 'persist_case', saved)
    def build(max_steps=20, max_calls=10, call_timeout=1.0):
        return workflow.build_workflow(
            cultpass_engine=engine, udahub_engine=engine,
            models={role: model for role in workflow.AgentModels.__annotations__},
            knowledge_store=UnusedVectorStore(),
            retrieval_settings={'threshold': 0.5, 'candidate_count': 3,
                                'embedding_model': 'fake', 'score_convention': 'higher_is_more_relevant'},
            limits={'max_graph_steps': max_steps, 'max_model_calls': max_calls, 'call_timeout_seconds': call_timeout},
            checkpointer=InMemorySaver(), store=InMemoryStore())
    yield build, model, saved
    engine.dispose()


def test_classifier_executes_once_as_registered_node(monkeypatch, case, assessment, factory):
    assess = Mock(return_value=assessment)
    monkeypatch.setattr(classifier, 'assess_request', assess)
    build, model, _ = factory
    graph = build()
    iterator = graph.stream({'messages': case['messages']},
                            {'configurable': {'thread_id': 'ticket-1'}}, stream_mode='updates')
    updates = [next(iterator) for _ in range(4)]
    iterator.close()
    assert [next(iter(item)) for item in updates] == ['intake', 'supervisor', 'classifier', 'supervisor']
    assert updates[1]['supervisor']['assignment']['agent'] == 'classifier'
    assert updates[2]['classifier']['classification_status'] == 'completed'
    assert updates[2]['classifier']['model_calls'] == 1
    assert updates[3]['supervisor']['assignment']['agent'] == 'knowledge'
    assert 'messages' not in updates[1]['supervisor'] and 'messages' not in updates[2]['classifier']
    assess.assert_called_once()
    assert model.invocations == 0


def test_exhausted_budget_saves_without_classification(monkeypatch, case, factory):
    case['model_calls'] = 2
    assess = Mock()
    monkeypatch.setattr(classifier, 'assess_request', assess)
    build, model, saved = factory
    updates = list(build(max_calls=2).stream({'messages': case['messages']},
                   {'configurable': {'thread_id': 'ticket-1'}}, stream_mode='updates'))
    assert [next(iter(item)) for item in updates] == ['intake', 'supervisor', 'persist']
    update = updates[1]['supervisor']
    assert update['handoff']['handoff_source'] == 'deterministic_fallback'
    assert update['classification_status'] == 'skipped' and update['approved_response'] is None
    assess.assert_not_called()
    saved.assert_called_once()
    assert model.invocations == 0


def test_denied_classifier_retry_returns_through_supervisor(monkeypatch, case, factory):
    assess = Mock(side_effect=TimeoutError('PRIVATE EXCEPTION'))
    monkeypatch.setattr(classifier, 'assess_request', assess)
    build, model, saved = factory
    updates = list(build(max_calls=1).stream({'messages': case['messages']},
                   {'configurable': {'thread_id': 'ticket-1'}}, stream_mode='updates'))
    assert [next(iter(item)) for item in updates] == ['intake', 'supervisor', 'classifier', 'supervisor', 'persist']
    assert updates[2]['classifier']['model_calls'] == 1
    assert updates[2]['classifier']['classification_status'] == 'failed'
    assert updates[3]['supervisor']['handoff']['handoff_source'] == 'deterministic_fallback'
    assess.assert_called_once()
    saved.assert_called_once()
    assert model.invocations == 0


def test_classifier_deadline_preserves_handoff_without_retry(monkeypatch, case, assessment, factory):
    from threading import Event
    release, completed = Event(), Event()
    def slow(*args, **kwargs):
        release.wait(2)
        completed.set()
        return assessment
    assess = Mock(side_effect=slow)
    monkeypatch.setattr(classifier, 'assess_request', assess)
    build, model, saved = factory
    try:
        updates = list(build(call_timeout=0.01).stream({'messages': case['messages']},
                       {'configurable': {'thread_id': 'ticket-1'}}, stream_mode='updates'))
        assert [next(iter(item)) for item in updates] == ['intake', 'supervisor', 'classifier', 'supervisor', 'persist']
        assert updates[2]['classifier']['model_calls'] == 1
        assert updates[2]['classifier']['trace'][0]['payload']['error_code'] == 'classification_call_timeout'
        assert updates[3]['supervisor']['handoff']['handoff_source'] == 'deterministic_fallback'
        assess.assert_called_once()
        saved.assert_called_once()
        assert model.invocations == 0
    finally:
        release.set()
        assert completed.wait(1)


def test_completed_provider_timeout_keeps_single_transient_retry(monkeypatch, case, assessment, factory):
    assess = Mock(side_effect=[TimeoutError('PRIVATE EXCEPTION'), assessment])
    monkeypatch.setattr(classifier, 'assess_request', assess)
    build, model, _ = factory
    iterator = build().stream({'messages': case['messages']},
                             {'configurable': {'thread_id': 'ticket-1'}}, stream_mode='updates')
    updates = [next(iterator) for _ in range(4)]
    iterator.close()
    assert updates[2]['classifier']['classification_status'] == 'completed'
    assert updates[2]['classifier']['model_calls'] == 2
    assert assess.call_count == 2 and model.invocations == 0
