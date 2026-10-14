"""Release only saved, current-turn approved output at the chat boundary."""
import logging
from unittest.mock import Mock
import pytest
from langchain_core.messages import HumanMessage
from utils import chat_interface


def run_chat(monkeypatch, builder):
    inputs = iter(['Help me sign in', 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    graph = Mock()
    graph.invoke.side_effect = builder
    chat_interface(graph, 'ticket-1')
    return graph


def outcome(input, config, saved=True, reply=True):
    turn = input['messages'][0].id
    return {'turn_id': turn, 'messages': [HumanMessage(content='Old text')],
            'persistence_result': {'saved': saved, 'ticket_id': 'ticket-1', 'turn_id': turn},
            'approved_response': {'message_id': f'{turn}:reply', 'content': 'Approved answer'} if reply else None}


def test_approved_output_and_stable_turn_id(monkeypatch, capsys, caplog):
    with caplog.at_level(logging.INFO):
        graph = run_chat(monkeypatch, outcome)
    output = capsys.readouterr().out
    assert 'Assistant: Approved answer' in output and 'Assistant: Old text' not in output
    assert graph.invoke.call_args.kwargs['input']['messages'][0].id
    assert graph.invoke.call_args.kwargs['config']['configurable']['thread_id'] == 'ticket-1'
    graph.update_state.assert_not_called()
    assert 'presentation_succeeded' in caplog.text


@pytest.mark.parametrize('saved,reply', [(False, True), (False, False), (True, False)])
def test_unsaved_or_handoff_only_output(monkeypatch, capsys, caplog, saved, reply):
    run_chat(monkeypatch, lambda input, config: outcome(input, config, saved, reply))
    output = capsys.readouterr().out
    assert 'Assistant: Approved answer' not in output and 'Assistant: Old text' not in output
    assert ('turn_not_saved' if not saved else 'handoff_only') in caplog.text


@pytest.mark.parametrize('field', ['ticket_id', 'turn_id'])
def test_stale_save_is_not_released(monkeypatch, capsys, caplog, field):
    def stale(input, config):
        result = outcome(input, config)
        result['persistence_result'][field] = 'unrelated'
        return result
    run_chat(monkeypatch, stale)
    assert 'Assistant: Approved answer' not in capsys.readouterr().out
    assert 'turn_not_saved' in caplog.text


def test_stale_approval_is_not_released(monkeypatch, capsys, caplog):
    def stale(input, config):
        result = outcome(input, config)
        result['approved_response']['message_id'] = 'old-turn:reply'
        return result
    run_chat(monkeypatch, stale)
    assert 'Assistant: Approved answer' not in capsys.readouterr().out
    assert 'invalid_approved_response' in caplog.text


def test_graph_failure_is_reported_safely(monkeypatch, capsys, caplog):
    def failed(**kwargs):
        raise RuntimeError('PRIVATE EXCEPTION')
    run_chat(monkeypatch, failed)
    assert 'Assistant:' not in capsys.readouterr().out
    assert 'workflow_failed' in caplog.text and 'PRIVATE EXCEPTION' not in caplog.text
