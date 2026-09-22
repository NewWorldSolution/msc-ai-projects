"""Executable tests for the Persist node's release and retry contracts."""

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import agentic.persist as persist_module
from agentic.persist import persist_case
from agentic.tools.memory_tools import search_customer_memory
from data.models import udahub
from utils import get_session


def resolved_state() -> dict:
    """Return one Supervisor-authorized resolved turn."""
    return {
        "ticket_id": "ticket-1",
        "account_id": "account-1",
        "customer_id": "customer-1",
        "turn_id": "turn-1",
        "outcome": "resolved",
        "messages": [
            HumanMessage(content="I cannot sign in.", id="turn-1"),
        ],
        "approved_response": {
            "message_id": "reply-1",
            "draft_id": "draft-1",
            "outcome": "resolved",
            "content": "Use the verified password reset steps.",
        },
        "review_result": {
            "draft_id": "draft-1",
            "verdict": "approved",
        },
        "handoff": None,
        "trace": [
            {
                "event_id": "turn-1:supervisor:approved",
                "turn_id": "turn-1",
                "issue_id": "issue-1",
                "node": "supervisor",
                "event_type": "draft_review",
                "outcome": "resolved",
                "reason": "The exact grounded draft was approved.",
                "payload": {"source_ids": ["article-1"]},
            }
        ],
        "memory_updates": [
            {
                "memory_id": "turn-1:issue-1",
                "memory_type": "resolved_issue",
                "content": "Password reset guidance resolved the sign-in issue.",
                "source_refs": ["turn-1", "article-1"],
            }
        ],
    }


def successful_core_result(payload: dict) -> dict:
    """Return the transaction helper's confirmed success shape."""
    return {
        "saved": True,
        "ticket_id": "ticket-1",
        "turn_id": payload["turn_id"],
        "outcome": payload["outcome"],
        "ticket_status": "resolved",
        "approved_response": payload["approved_response"],
        "handoff_saved": payload["handoff"] is not None,
        "trace_event_count": len(payload["trace_events"]),
        "error_code": None,
        "retryable": False,
        "reason": "The support outcome was committed.",
    }


def test_success_releases_only_persisted_reply_and_writes_memory(monkeypatch) -> None:
    state = resolved_state()
    calls = []

    def fake_save(ticket_id, payload, engine):
        calls.append((ticket_id, payload, engine))
        return successful_core_result(payload)

    monkeypatch.setattr(persist_module, "save_ticket_outcome", fake_save)
    store = InMemoryStore()

    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=object(),
        store=store,
    )

    assert update["persistence_result"]["saved"] is True
    assert update["persistence_result"]["attempts"] == 1
    assert update["persistence_result"]["memory"]["status"] == "succeeded"
    assert [message.id for message in update["messages"]] == ["reply-1"]
    assert update["messages"][0].content == state["approved_response"]["content"]
    assert calls[0][1]["customer_message"] == {
        "message_id": "turn-1",
        "content": "I cannot sign in.",
    }
    assert calls[0][1]["trace_events"][-1]["event_id"] == "turn-1:persist:outcome"

    recalled = search_customer_memory("account-1", "customer-1", store=store)
    assert recalled[0]["memory_id"] == "turn-1:issue-1"
    assert recalled[0]["ticket_id"] == "ticket-1"

    history = add_messages(state["messages"], update["messages"])
    history = add_messages(history, update["messages"])
    assert [message.id for message in history].count("reply-1") == 1


def test_transient_failure_retries_the_exact_same_batch_once(monkeypatch) -> None:
    state = resolved_state()
    state["memory_updates"] = []
    calls = []

    def transient_then_success(ticket_id, payload, engine):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "saved": False,
                "error_code": "transient_database_error",
                "retryable": True,
                "reason": "The support outcome could not be saved.",
            }
        return successful_core_result(payload)

    monkeypatch.setattr(
        persist_module,
        "save_ticket_outcome",
        transient_then_success,
    )

    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=object(),
        store=InMemoryStore(),
    )

    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert update["persistence_result"]["saved"] is True
    assert update["persistence_result"]["attempts"] == 2
    assert update["messages"][0].id == "reply-1"


def test_terminal_core_failure_returns_no_reply_or_memory_write(monkeypatch) -> None:
    state = resolved_state()

    def failed_save(ticket_id, payload, engine):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": payload["turn_id"],
            "error_code": "message_id_conflict",
            "retryable": False,
            "reason": "The stable message ID conflicts.",
        }

    monkeypatch.setattr(persist_module, "save_ticket_outcome", failed_save)
    store = InMemoryStore()

    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=object(),
        store=store,
    )

    assert update["persistence_result"]["saved"] is False
    assert update["persistence_result"]["attempts"] == 1
    assert "messages" not in update
    assert store.search(("customer_memory", "account-1", "customer-1")) == []
    assert update["trace"][0]["event_type"] == "persistence_failed"


def test_escalation_without_handoff_uses_fallback_and_returns_no_reply(
    monkeypatch,
) -> None:
    state = resolved_state()
    state.update(
        {
            "outcome": "escalated",
            "approved_response": None,
            "review_result": None,
            "handoff": None,
            "escalation_reason": "The account is blocked.",
            "classification_status": "completed",
            "classification": {
                "urgency": "urgent",
                "urgency_reason": "Paid service is inaccessible.",
                "complexity": "simple",
            },
            "issues": [
                {
                    "issue_id": "issue-1",
                    "task": "Restore account access.",
                    "category": "account_access",
                    "status": "unresolved",
                }
            ],
            "response_draft": {
                "content": "This rejected draft must not enter the handoff."
            },
            "memory_updates": [],
        }
    )
    captured = {}

    def successful_escalation(ticket_id, payload, engine):
        captured.update(payload)
        result = successful_core_result(payload)
        result["outcome"] = "escalated"
        result["ticket_status"] = "escalated"
        return result

    monkeypatch.setattr(
        persist_module,
        "save_ticket_outcome",
        successful_escalation,
    )

    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=object(),
        store=InMemoryStore(),
    )

    assert captured["handoff"]["handoff_source"] == "deterministic_fallback"
    assert captured["handoff"]["issue_ids"] == ["issue-1"]
    assert "rejected draft" not in str(captured["handoff"]).lower()
    assert update["persistence_result"]["saved"] is True
    assert update["persistence_result"]["memory"]["status"] == "skipped"
    assert "messages" not in update


def test_memory_failure_does_not_reverse_core_success_or_hide_reply(
    monkeypatch,
) -> None:
    state = resolved_state()

    monkeypatch.setattr(
        persist_module,
        "save_ticket_outcome",
        lambda ticket_id, payload, engine: successful_core_result(payload),
    )
    monkeypatch.setattr(
        persist_module,
        "save_customer_memory",
        lambda *args, **kwargs: {
            "saved": False,
            "memory_id": "turn-1:issue-1",
            "error_code": "memory_write_error",
            "retryable": False,
            "reason": "The memory store is unavailable.",
        },
    )

    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=object(),
        store=InMemoryStore(),
    )

    assert update["persistence_result"]["saved"] is True
    assert update["persistence_result"]["memory"]["status"] == "failed"
    assert update["persistence_result"]["memory"]["failed"] == 1
    assert update["messages"][0].id == "reply-1"
    assert update["trace"][-1]["event_type"] == "memory_write_summary"


def test_thread_ticket_mismatch_stops_before_database_write(monkeypatch) -> None:
    state = resolved_state()

    def unexpected_save(*args, **kwargs):
        raise AssertionError("The database helper must not be called.")

    monkeypatch.setattr(persist_module, "save_ticket_outcome", unexpected_save)

    update = persist_case(
        state,
        {"configurable": {"thread_id": "another-ticket"}},
        udahub_engine=object(),
        store=InMemoryStore(),
    )

    assert update["persistence_result"]["saved"] is False
    assert update["persistence_result"]["error_code"] == "thread_ticket_mismatch"
    assert update["persistence_result"]["attempts"] == 0
    assert "messages" not in update


def test_persist_case_commits_through_the_real_transaction_helper() -> None:
    database_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    udahub.Base.metadata.create_all(database_engine)
    with get_session(database_engine) as session:
        session.add(udahub.Account(account_id="account-1", account_name="CultPass"))
        session.add(
            udahub.User(
                user_id="support-user-1",
                account_id="account-1",
                external_user_id="customer-1",
                user_name="Ada",
            )
        )
        session.add(
            udahub.Ticket(
                ticket_id="ticket-1",
                account_id="account-1",
                user_id="support-user-1",
                channel="chat",
            )
        )
        session.add(
            udahub.TicketMetadata(
                ticket_id="ticket-1",
                status="open",
                main_issue_type="account_access",
                tags="login, access",
            )
        )

    state = resolved_state()
    state["memory_updates"] = []
    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=database_engine,
        store=InMemoryStore(),
    )

    assert update["persistence_result"]["saved"] is True
    assert update["persistence_result"]["trace_event_count"] == 2
    assert update["messages"][0].id == "reply-1"
    with get_session(database_engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        messages = session.query(udahub.TicketMessage).all()
        trace_events = session.query(udahub.TraceEvent).all()

        assert metadata.status == "resolved"
        assert {message.message_id for message in messages} == {"turn-1", "reply-1"}
        assert {event.event_id for event in trace_events} == {
            "turn-1:supervisor:approved",
            "turn-1:persist:outcome",
        }


def test_real_transaction_rejects_nonmatching_review_without_reply() -> None:
    database_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    udahub.Base.metadata.create_all(database_engine)
    with get_session(database_engine) as session:
        session.add(udahub.Account(account_id="account-1", account_name="CultPass"))
        session.add(
            udahub.User(
                user_id="support-user-1",
                account_id="account-1",
                external_user_id="customer-1",
                user_name="Ada",
            )
        )
        session.add(
            udahub.Ticket(
                ticket_id="ticket-1",
                account_id="account-1",
                user_id="support-user-1",
                channel="chat",
            )
        )
        session.add(
            udahub.TicketMetadata(
                ticket_id="ticket-1",
                status="open",
                main_issue_type="account_access",
                tags="login, access",
            )
        )

    state = resolved_state()
    state["review_result"]["draft_id"] = "another-draft"
    update = persist_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        udahub_engine=database_engine,
        store=InMemoryStore(),
    )

    assert update["persistence_result"]["saved"] is False
    assert update["persistence_result"]["error_code"] == "response_not_approved"
    assert "messages" not in update
    with get_session(database_engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"
        assert session.query(udahub.TicketMessage).count() == 0
        assert session.query(udahub.TraceEvent).count() == 0
