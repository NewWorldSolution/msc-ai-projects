"""Checks for the proposed runner using temporary databases and no providers."""
from datetime import datetime, timedelta
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from data.models import udahub
from utils import get_session
from test_intake import seeded_resources
from test_model_budget import ScriptedModel
from test_workflow_integration import UnusedVectorStore
from agentic.workflow import AgentModels, build_workflow

import utils as batch


@pytest.fixture
def core():
    core, business = seeded_resources()
    base = datetime(2026, 10, 10, 12)
    with get_session(core) as session:
        session.get(udahub.Ticket, "ticket-1").channel = "email"
        for i, ticket in enumerate(["ticket-1", "ticket-2"]):
            if i:
                session.add(udahub.Ticket(ticket_id=ticket, account_id="account-1",
                                         user_id="support-user-1", channel="email",
                                         created_at=base + timedelta(minutes=i)))
                session.add(udahub.TicketMetadata(ticket_id=ticket, status="open"))
            session.add(udahub.TicketMessage(message_id=f"message-{i+1}",
                                            ticket_id=ticket, role="user",
                                            content=f"Customer request {i+1}",
                                            created_at=base + timedelta(minutes=i)))
    yield core
    core.dispose()
    business.dispose()


def success(input, config):
    message = input["messages"][0]
    ticket = config["configurable"]["thread_id"]
    return {"turn_id": message.id, "outcome": "resolved",
            "persistence_result": {"saved": True, "ticket_id": ticket, "turn_id": message.id},
            "approved_response": {"message_id": f"{message.id}:reply", "content": "Subject: Help\n\nApproved email"}}


def test_outputs_arrive_one_at_a_time_and_reuse_source_ids(core, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected input prompt"))
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    def invoke(input, config):
        if config["configurable"]["thread_id"] == "ticket-2":
            # The first email must be displayed before invoking the second ticket.
            assert "Approved email" in capsys.readouterr().out
        return success(input, config)
    graph.invoke.side_effect = invoke
    first, second = batch.batch_interface(graph, core, account_id="account-1")
    assert graph.invoke.call_count == 2
    assert first["turn_id"] == "message-1" and first["channel"] == "email"
    assert first["reply"].startswith("Subject:") and second["turn_id"] == "message-2"
    assert graph.invoke.call_args_list[0].kwargs["input"]["messages"][0].content == "Customer request 1"
    assert graph.invoke.call_args_list[0].kwargs["config"]["configurable"]["thread_id"] == "ticket-1"


def test_completed_open_turns_and_human_messages_only(core):
    with get_session(core) as session:
        session.add(udahub.TraceEvent(event_id="message-1:persist:outcome", ticket_id="ticket-1",
                                     turn_id="message-1", node="persist", event_type="final_outcome"))
        session.add(udahub.TicketMessage(message_id="agent-message", ticket_id="ticket-2",
                                        role="ai", content="Do not treat me as a request"))
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = success
    rows = list(batch.batch_interface(graph, core, account_id="account-1"))
    assert [row["turn_id"] for row in rows] == ["message-2"]


@pytest.mark.parametrize("status", ["resolved", "escalated"])
def test_closed_tickets_are_not_automated(core, status):
    with get_session(core) as session:
        session.get(udahub.TicketMetadata, "ticket-1").status = status
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = success
    assert [row["ticket_id"] for row in batch.batch_interface(graph, core, account_id="account-1")] == ["ticket-2"]


def test_account_scope(core):
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    assert list(batch.batch_interface(graph, core, account_id="other-account")) == []
    graph.invoke.assert_not_called()


@pytest.mark.parametrize("failed", [False, True])
def test_later_messages_do_not_restart_a_closed_or_failed_ticket(core, failed):
    with get_session(core) as session:
        session.add(udahub.TicketMessage(message_id="message-later", ticket_id="ticket-1",
                                        role="user", content="Another queued request",
                                        created_at=datetime(2026, 10, 10, 13)))
    def invoke(input, config):
        if config["configurable"]["thread_id"] == "ticket-1":
            if failed:
                raise RuntimeError("Failure")
            with get_session(core) as session:
                session.get(udahub.TicketMetadata, "ticket-1").status = "resolved"
        return success(input, config)
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = invoke
    rows = list(batch.batch_interface(graph, core, account_id="account-1"))
    assert [row["turn_id"] for row in rows] == ["message-1", "message-2", "message-later"]
    assert rows[-1]["status"] == "skipped"
    assert graph.invoke.call_count == 2


@pytest.mark.parametrize("fault", ["unsaved", "wrong_ticket", "wrong_turn", "stale_approval", "missing_reply", "exception"])
def test_bad_results_are_not_presented_and_next_ticket_continues(core, fault):
    def invoke(input, config):
        if config["configurable"]["thread_id"] == "ticket-2":
            return success(input, config)
        if fault == "exception":
            raise RuntimeError("PRIVATE PROVIDER DETAILS")
        result = success(input, config)
        if fault == "unsaved": result["persistence_result"]["saved"] = False
        if fault == "wrong_ticket": result["persistence_result"]["ticket_id"] = "other-ticket"
        if fault == "wrong_turn": result["turn_id"] = "other-turn"
        if fault == "stale_approval": result["approved_response"]["message_id"] = "old:reply"
        if fault == "missing_reply": result["approved_response"] = None
        return result
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = invoke
    first, second = list(batch.batch_interface(graph, core, account_id="account-1"))
    assert first["status"] == "failed" and first["reply"] is None
    assert second["status"] == "processed"
    assert "PRIVATE" not in str(first)


def test_saved_handoff_without_reply_is_reported(core):
    def invoke(input, config):
        result = success(input, config)
        result.update(outcome="escalated", approved_response=None)
        result["persistence_result"]["handoff_saved"] = True
        return result
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = invoke
    assert all(row["status"] == "handoff_only" and row["reply"] is None
               for row in batch.batch_interface(graph, core, account_id="account-1"))


def test_real_graph_saves_existing_message_and_second_run_is_empty():
    core, business = seeded_resources()
    try:
        with get_session(core) as session:
            session.get(udahub.Ticket, "ticket-1").channel = "email"
            session.add(udahub.TicketMessage(message_id="stored-id", ticket_id="ticket-1",
                                            role="user", content="Please help me"))
        model = ScriptedModel(responses=[AIMessage(content="Must not be called")])
        graph = build_workflow(
            cultpass_engine=business, udahub_engine=core,
            models={role: model for role in AgentModels.__annotations__},
            knowledge_store=UnusedVectorStore(),
            retrieval_settings={"threshold": 0.15, "candidate_count": 5,
                                "embedding_model": "fake", "score_convention": "higher_is_more_relevant"},
            limits={"max_graph_steps": 3, "max_model_calls": 10, "call_timeout_seconds": 1},
            checkpointer=InMemorySaver(), store=InMemoryStore(),
        )
        results = list(batch.batch_interface(graph, core, account_id="account-1"))
        assert results[0]["status"] == "handoff_only"
        assert results[0]["persistence_result"]["saved"] is True
        assert model.invocations == 0
        with get_session(core) as session:
            assert session.query(udahub.TicketMessage).filter_by(message_id="stored-id").count() == 1
            assert session.query(udahub.TraceEvent).filter_by(turn_id="stored-id", node="persist", event_type="final_outcome").count() == 1
        assert list(batch.batch_interface(graph, core, account_id="account-1")) == []
    finally:
        core.dispose(); business.dispose()


@pytest.mark.parametrize("checkpoint", ["pending_node", "same_turn_without_commit", "read_error"])
def test_incomplete_checkpoint_is_reported_without_invocation(core, checkpoint):
    graph = Mock()
    empty = SimpleNamespace(next=(), values={})
    if checkpoint == "read_error":
        graph.get_state.side_effect = [RuntimeError("PRIVATE CHECKPOINT DETAIL"), empty]
    else:
        first = SimpleNamespace(
            next=("knowledge",) if checkpoint == "pending_node" else (),
            values={"turn_id": "message-1"} if checkpoint == "same_turn_without_commit" else {},
        )
        graph.get_state.side_effect = [first, empty]
    graph.invoke.side_effect = success
    rows = batch.batch_interface(graph, core, account_id="account-1")
    assert rows[0]["error_code"] == (
        "checkpoint_read_failed" if checkpoint == "read_error" else "incomplete_previous_turn")
    assert rows[1]["status"] == "processed"
    assert graph.invoke.call_count == 1


def test_error_and_presentation_logs_include_ids_without_exception_text(core, caplog):
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    def invoke(input, config):
        if config["configurable"]["thread_id"] == "ticket-1":
            raise RuntimeError("PRIVATE PROVIDER DETAIL")
        return success(input, config)
    graph.invoke.side_effect = invoke
    with caplog.at_level("INFO", logger="utils"):
        rows = batch.batch_interface(graph, core, account_id="account-1")
    assert rows[0]["status"] == "failed"
    assert "workflow_failed ticket=ticket-1 turn=message-1 exception_type=RuntimeError" in caplog.text
    assert "presentation_succeeded ticket=ticket-2 turn=message-2" in caplog.text
    assert "PRIVATE" not in caplog.text


def test_presentation_failure_is_logged_and_stops_batch(core, monkeypatch, caplog):
    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(next=(), values={})
    graph.invoke.side_effect = success
    def broken_print(*args, **kwargs):
        if args and str(args[0]).startswith("Subject:"):
            raise OSError("PRIVATE OUTPUT DETAIL")
    monkeypatch.setattr("builtins.print", broken_print)
    with caplog.at_level("INFO", logger="utils"), pytest.raises(OSError):
        batch.batch_interface(graph, core, account_id="account-1")
    assert graph.invoke.call_count == 1
    assert "presentation_failed ticket=ticket-1 turn=message-1 exception_type=OSError" in caplog.text
    assert "presentation_succeeded" not in caplog.text
    assert "PRIVATE" not in caplog.text


def real_batch_graph(core, business, *, max_steps=20, store=None):
    model = ScriptedModel(responses=[AIMessage(content="No provider calls")])
    return build_workflow(
        cultpass_engine=business, udahub_engine=core,
        models={role: model for role in AgentModels.__annotations__},
        knowledge_store=UnusedVectorStore(),
        retrieval_settings={"threshold": 0.15, "candidate_count": 5,
                            "embedding_model": "fake", "score_convention": "higher_is_more_relevant"},
        limits={"max_graph_steps": max_steps, "max_model_calls": 10, "call_timeout_seconds": 1},
        checkpointer=InMemorySaver(), store=store or InMemoryStore(),
    )


def test_real_specialist_crash_is_not_reentered_as_a_new_turn(monkeypatch):
    from agentic import workflow
    from agentic.agents import classifier
    core, business = seeded_resources()
    try:
        with get_session(core) as session:
            session.add(udahub.TicketMessage(message_id="stored-1", ticket_id="ticket-1",
                                            role="user", content="I cannot sign in."))
        assess = Mock(return_value=classifier.TicketClassification(
            issues=[classifier.IssueAssessment(category="account_access", task="Explain login recovery",
                missing_information=[], missing_information_reasons={},
                needs_operations=False, required_business_facts=[])],
            urgency="normal", urgency_reason="No deadline", complexity="simple"))
        monkeypatch.setattr(classifier, "assess_request", assess)
        crash = Mock(side_effect=RuntimeError("Knowledge failed unexpectedly"))
        monkeypatch.setattr(workflow, "retrieve_knowledge", crash)
        graph = real_batch_graph(core, business)
        first = batch.batch_interface(graph, core, account_id="account-1")
        second = batch.batch_interface(graph, core, account_id="account-1")
        assert first[0]["error_code"] == "workflow_failed"
        assert second[0]["error_code"] == "incomplete_previous_turn"
        assert assess.call_count == crash.call_count == 1
        snapshot = graph.get_state({"configurable": {"thread_id": "ticket-1"}})
        assert snapshot.next == ("knowledge",)
        with get_session(core) as session:
            assert session.get(udahub.TicketMetadata, "ticket-1").status == "open"
            assert session.query(udahub.TraceEvent).filter_by(event_type="final_outcome").count() == 0
    finally:
        core.dispose()
        business.dispose()


def test_real_graph_follow_up_keeps_history_and_persists_approved_email(monkeypatch, capsys):
    """Script approval only; use real graph, Intake, Persist, SQLite and checkpoints."""
    import json
    from pathlib import Path
    from langgraph.types import Command
    from agentic import workflow
    core, business = seeded_resources()
    store = InMemoryStore()
    observed = []
    try:
        with get_session(core) as session:
            session.get(udahub.Ticket, "ticket-1").channel = "email"
            for index, content in enumerate(["I need help.", "How do I reset my password?"]):
                session.add(udahub.TicketMessage(
                    message_id=f"turn-{index+1}", ticket_id="ticket-1", role="user", content=content,
                    created_at=datetime(2026, 1, 1, 12) + timedelta(minutes=index)))
        def scripted_approval(state, config, *, model):
            observed.append([message.id for message in state["messages"]])
            turn = state["turn_id"]
            outcome = "information_requested" if turn == "turn-1" else "resolved"
            text = "What is the problem?" if turn == "turn-1" else "Use the verified reset steps."
            return Command(goto="persist", update={
                "outcome": outcome, "memory_updates": [],
                "approved_response": {"message_id": f"{turn}:reply", "draft_id": f"{turn}:draft",
                                      "outcome": outcome, "content": "Subject: CultPass help\n\n" + text},
                "review_result": {"draft_id": f"{turn}:draft", "verdict": "approved"},
            })
        monkeypatch.setattr(workflow, "supervise_case", scripted_approval)
        graph = real_batch_graph(core, business, store=store)
        rows = batch.batch_interface(graph, core, account_id="account-1")
        assert [r["outcome"] for r in rows] == ["information_requested", "resolved"]
        assert all(r["status"] == "processed" for r in rows)
        assert observed == [["turn-1"], ["turn-1", "turn-1:reply", "turn-2"]]
        snapshot = graph.get_state({"configurable": {"thread_id": "ticket-1"}})
        assert [m.id for m in snapshot.values["messages"]] == ["turn-1", "turn-1:reply", "turn-2", "turn-2:reply"]
        with get_session(core) as session:
            assert session.get(udahub.TicketMetadata, "ticket-1").status == "resolved"
            assert session.query(udahub.TicketMessage).count() == 4
            assert session.get(udahub.TicketMessage, "turn-2:reply").content == rows[1]["reply"]
            assert session.query(udahub.TraceEvent).filter_by(event_type="final_outcome").count() == 2
        assert batch.batch_interface(graph, core, account_id="account-1") == []
        notebook = json.loads((Path(__file__).parents[1]/'03_agentic_app.ipynb').read_text())
        inspection = next(''.join(c['source']) for c in notebook['cells']
                          if ''.join(c['source']).startswith('# Inspect every attempted'))
        # The inspection cell calls utils.routing_path; the notebook imports utils in its first cell.
        scope = {"orchestrator": graph, "store": store, "demo_runs": rows, "json": json, "utils": batch}
        exec(compile(inspection, '<inspection cell>', 'exec'), scope)
        report = scope["demo_inspections"][0]
        assert report["checkpoint_count"] > 2
        assert report["routing"] == ["email -> intake -> supervisor"] * 2
        assert report["pending_nodes"] == []
        assert len(report["messages"]) == 4 and len(report["turns"]) == 2
        assert any(event["node"] == "intake" for event in report["trace"])
        assert '"checkpoint_count"' in capsys.readouterr().out
    finally:
        core.dispose()
        business.dispose()


def test_notebook_seed_uses_timestamps_before_uuid_tiebreakers():
    """Execute the actual proposed seed cell with UUIDs ordered against scenario order."""
    import json
    from pathlib import Path
    core, business = seeded_resources()
    try:
        notebook = json.loads((Path(__file__).parents[1]/'02_core_db_setup.ipynb').read_text())
        seed = next(''.join(c['source']) for c in notebook['cells']
                    if ''.join(c['source']).startswith('# Additional email cases:'))
        identifiers = iter(['f-resolution', 'm-resolution', '000-returning', 'm-returning',
                            'c-tools', 'm-tools', 'd-unsupported', 'm-unsupported',
                            'e-follow-up', 'm-follow-up', 'm-follow-up-2',
                            'g-no-article', 'm-no-article'])
        scope = {"get_session": get_session, "engine": core, "account_id": "account-1",
                 "cultpass_users": [{"id": "customer-1", "name": "Ada", "is_blocked": False}],
                 "udahub": udahub, "json": json,
                 "uuid": SimpleNamespace(uuid4=lambda: next(identifiers))}
        exec(compile(seed, '<seed cell>', 'exec'), scope)
        with get_session(core) as session:
            seeded = session.query(udahub.TicketMessage).order_by(udahub.TicketMessage.created_at).all()
            assert len({m.created_at for m in seeded}) == 7
            assert [m.ticket_id for m in seeded[:2]] == ['f-resolution', '000-returning']
            assert session.get(udahub.Ticket, 'ticket-1').channel == 'chat'
            assert all(session.get(udahub.Ticket, m.ticket_id).channel == 'email' for m in seeded)
        graph = Mock()
        graph.get_state.return_value = SimpleNamespace(next=(), values={})
        graph.invoke.side_effect = success
        rows = batch.batch_interface(graph, core, account_id='account-1')
        assert [r['turn_id'] for r in rows] == ['m-resolution', 'm-returning', 'm-tools',
                                               'm-unsupported', 'm-follow-up', 'm-follow-up-2',
                                               'm-no-article']
    finally:
        core.dispose()
        business.dispose()
