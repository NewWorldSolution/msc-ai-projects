"""Executable tests for atomic UDA-Hub ticket outcome persistence."""

import sqlite3
from contextlib import contextmanager
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from agentic.tools import ticket_tools

from agentic.tools.ticket_tools import save_ticket_outcome
from data.models import udahub
from utils import get_session


@pytest.fixture
def engine():
    """Create one in-memory UDA-Hub database with a valid open ticket."""
    database_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    udahub.Base.metadata.create_all(database_engine)

    with get_session(database_engine) as session:
        session.add(
            udahub.Account(
                account_id="account-1",
                account_name="CultPass",
            )
        )
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

    return database_engine


def resolved_outcome() -> dict:
    """Return one valid resolved-turn payload with stable identifiers."""
    return {
        "turn_id": "turn-1",
        "outcome": "resolved",
        "customer_message": {
            "message_id": "turn-1",
            "content": "I cannot sign in.",
        },
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
        "trace_events": [
            {
                "event_id": "event-1",
                "turn_id": "turn-1",
                "issue_id": "issue-1",
                "node": "supervisor",
                "event_type": "final_outcome",
                "outcome": "resolved",
                "reason": "The exact grounded draft was approved.",
                "payload": {
                    "source_ids": ["article-1"],
                    "attempt": 1,
                },
            }
        ],
    }


def test_resolved_outcome_commits_messages_status_and_trace(engine) -> None:
    result = save_ticket_outcome("ticket-1", resolved_outcome(), engine)

    assert result == {
        "saved": True,
        "ticket_id": "ticket-1",
        "turn_id": "turn-1",
        "outcome": "resolved",
        "ticket_status": "resolved",
        "approved_response": resolved_outcome()["approved_response"],
        "handoff_saved": False,
        "trace_event_count": 1,
        "error_code": None,
        "retryable": False,
        "reason": "The support outcome was committed.",
    }

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        messages = (
            session.query(udahub.TicketMessage)
            .order_by(udahub.TicketMessage.message_id)
            .all()
        )
        trace_event = session.get(udahub.TraceEvent, "event-1")

        assert metadata.status == "resolved"
        assert [message.role for message in messages] == [
            udahub.RoleEnum.ai,
            udahub.RoleEnum.user,
        ]
        assert trace_event.payload["source_ids"] == ["article-1"]


def test_information_requested_keeps_ticket_open(engine) -> None:
    payload = resolved_outcome()
    payload["outcome"] = "information_requested"
    payload["approved_response"]["outcome"] = "information_requested"
    payload["trace_events"][0]["outcome"] = "information_requested"

    result = save_ticket_outcome("ticket-1", payload, engine)

    assert result["saved"] is True
    assert result["ticket_status"] == "open"

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"


def test_same_stable_ids_make_a_retry_idempotent(engine) -> None:
    payload = resolved_outcome()

    first_result = save_ticket_outcome("ticket-1", payload, engine)
    second_result = save_ticket_outcome("ticket-1", payload, engine)

    assert first_result["saved"] is True
    assert second_result["saved"] is True

    with get_session(engine) as session:
        assert session.query(udahub.TicketMessage).count() == 2
        assert session.query(udahub.TraceEvent).count() == 1


def test_unapproved_exact_draft_is_rejected_before_any_write(engine) -> None:
    payload = resolved_outcome()
    payload["review_result"]["draft_id"] = "different-draft"

    result = save_ticket_outcome("ticket-1", payload, engine)

    assert result["saved"] is False
    assert result["error_code"] == "response_not_approved"
    assert result["retryable"] is False

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"
        assert session.query(udahub.TicketMessage).count() == 0
        assert session.query(udahub.TraceEvent).count() == 0


def test_handoff_only_escalation_saves_no_ai_message(engine) -> None:
    payload = {
        "turn_id": "turn-2",
        "outcome": "escalated",
        "customer_message": {
            "message_id": "turn-2",
            "content": "My account is blocked.",
        },
        "approved_response": None,
        "review_result": None,
        "handoff": {
            "message_id": "handoff-2",
            "turn_id": "turn-2",
            "issue_ids": ["issue-2"],
            "handoff_source": "deterministic_fallback",
            "request": "Restore account access.",
            "escalation_reason": "The account is blocked.",
            "urgency": "urgent",
            "urgency_reason": "Customer cannot access a paid service.",
            "complexity": "simple",
        },
        "trace_events": [
            {
                "event_id": "event-2",
                "turn_id": "turn-2",
                "issue_id": "issue-2",
                "node": "supervisor",
                "event_type": "final_outcome",
                "outcome": "escalated",
                "reason": "Blocked accounts require human handling.",
                "payload": {},
            }
        ],
    }

    result = save_ticket_outcome("ticket-1", payload, engine)

    assert result["saved"] is True
    assert result["handoff_saved"] is True

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        messages = (
            session.query(udahub.TicketMessage)
            .order_by(udahub.TicketMessage.message_id)
            .all()
        )

        assert metadata.status == "escalated"
        assert [message.role for message in messages] == [
            udahub.RoleEnum.system,
            udahub.RoleEnum.user,
        ]
        handoff_message = messages[0]
        handoff_payload = json.loads(handoff_message.content)
        assert handoff_payload["message_type"] == "human_handoff"
        assert handoff_payload["handoff_source"] == "deterministic_fallback"


def test_message_id_conflict_returns_no_partial_write(engine) -> None:
    with get_session(engine) as session:
        session.add(
            udahub.TicketMessage(
                message_id="reply-1",
                ticket_id="ticket-1",
                role=udahub.RoleEnum.ai,
                content="Different existing content.",
            )
        )

    result = save_ticket_outcome("ticket-1", resolved_outcome(), engine)

    assert result["saved"] is False
    assert result["error_code"] == "message_id_conflict"
    assert result["retryable"] is False

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"
        assert session.query(udahub.TicketMessage).count() == 1
        assert session.get(udahub.TicketMessage, "turn-1") is None
        assert session.query(udahub.TraceEvent).count() == 0


def test_outdated_schema_is_non_retryable_and_writes_nothing(engine) -> None:
    udahub.TraceEvent.__table__.drop(engine)

    result = save_ticket_outcome("ticket-1", resolved_outcome(), engine)

    assert result["saved"] is False
    assert result["error_code"] == "database_error"
    assert result["retryable"] is False

    with get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"
        assert session.query(udahub.TicketMessage).count() == 0


def test_transient_commit_failure_rolls_back_the_whole_batch(
    engine, monkeypatch
) -> None:
    original_get_session = ticket_tools.get_session

    @contextmanager
    def failing_commit(database_engine):
        Session = sessionmaker(bind=database_engine)
        session = Session()
        try:
            yield session
            raise OperationalError(
                "COMMIT",
                {},
                sqlite3.OperationalError("database is locked"),
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # `monkeypatch.setattr` temporarily replaces one module attribute for this test.
    monkeypatch.setattr(ticket_tools, "get_session", failing_commit)

    result = ticket_tools.save_ticket_outcome("ticket-1", resolved_outcome(), engine)

    assert result["saved"] is False
    assert result["error_code"] == "transient_database_error"
    assert result["retryable"] is True

    with original_get_session(engine) as session:
        metadata = session.get(udahub.TicketMetadata, "ticket-1")
        assert metadata.status == "open"
        assert session.query(udahub.TicketMessage).count() == 0
        assert session.query(udahub.TraceEvent).count() == 0
