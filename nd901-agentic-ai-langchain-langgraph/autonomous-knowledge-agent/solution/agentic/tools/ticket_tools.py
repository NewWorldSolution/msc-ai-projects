"""Plain support-record helpers for Intake and atomic outcome persistence."""

import json
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from data.models import udahub
from agentic.tools.database_errors import is_retryable_database_error
from utils import get_session


def load_ticket_context(ticket_id: str, engine: Engine) -> dict[str, Any]:
    """Read a ticket's support account, user mapping and metadata.

    Args:
        ticket_id: Existing ticket ID supplied by the trusted application.
        engine: Prepared UDA-Hub support database engine.

    Returns:
        A dictionary whose outcome is success, missing_input, missing_ticket,
        missing_account, missing_support_user, account_mismatch, missing_metadata
        or database_error. Database failures include error_code and deterministic
        retryable classification.
        Success includes ticket/channel/account identifiers, support-user facts,
        external_user_id and ticket metadata; it contains no CultPass customer
        details or conversation history.

    Notes:
        Checks that the support user and ticket belong to the same account.
        Reads only, closes its session and returns sanitized SQLAlchemy failures.
        Unusable ticket IDs produce missing_input before database access. Only
        recognized transient database failures are retryable. This helper
        performs no retries.
    """

    if not isinstance(ticket_id, str) or not ticket_id.strip():
        return {
            "outcome": "missing_input",
            "ticket_id": None,
            "error_code": "invalid_ticket_id",
            "retryable": False,
            "reason": "A non-empty ticket ID is required.",
        }

    # `get_session` opens a session and guarantees rollback/cleanup on errors.
    try:
        with get_session(engine) as session:
            ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()
            if ticket is None:
                return {
                    "outcome": "missing_ticket",
                    "ticket_id": ticket_id,
                    "reason": f"The ticket '{ticket_id}' not found.",
                }

            account = ticket.account
            support_user = ticket.user
            metadata = ticket.ticket_metadata

            if account is None:
                return {
                    "outcome": "missing_account",
                    "ticket_id": ticket_id,
                    "reason": "The ticket has no valid account relationship.",
                }
            if support_user is None:
                return {
                    "outcome": "missing_support_user",
                    "ticket_id": ticket_id,
                    "reason": "The ticket has no valid support-user relationship.",
                }

            if support_user.account_id != ticket.account_id:
                return {
                    "outcome": "account_mismatch",
                    "ticket_id": ticket_id,
                    "reason": "The support-user's and ticket belong to different accounts.",
                }
            if metadata is None:
                return {
                    "outcome": "missing_metadata",
                    "ticket_id": ticket_id,
                    "reason": "The ticket has no associated metadata.",
                }
            return {
                "outcome": "success",
                "ticket_id": ticket.ticket_id,
                "channel": ticket.channel,
                "account_id": account.account_id,
                "account_name": account.account_name,
                "support_user_id": support_user.user_id,
                "support_user_name": support_user.user_name,
                "external_user_id": support_user.external_user_id,
                "ticket_status": metadata.status,
                "main_issue_type": metadata.main_issue_type,
                "tags": metadata.tags,
            }
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            "outcome": "database_error",
            "ticket_id": ticket_id,
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
            "reason": "The UDA-Hub database query could not be completed.",
        }


def save_ticket_outcome(
    ticket_id: str, outcome: dict[str, Any], engine: Engine
) -> dict[str, Any]:
    """Validate and atomically persist one final support-turn outcome.

    Args:
        ticket_id: Trusted existing UDA-Hub ticket ID.
        outcome: Dictionary containing turn_id, final outcome, current customer
            message, current-turn trace events, optional approved_response and
            review_result, and a handoff for an escalated outcome. Message and
            event IDs must be stable. The customer message ID must equal turn_id;
            an approved response requires a matching approved draft review.
        engine: Prepared UDA-Hub support database engine.

    Returns:
        A dictionary with saved, error_code, retryable and reason, plus validated
        identity where available. Confirmed success includes ticket status,
        approved_response, handoff_saved and trace_event_count. Resolved maps to
        resolved, information_requested to open, and escalated to escalated.
        Identical stored messages/events are reused; conflicting IDs are rejected.

    Notes:
        Writes only support messages, metadata and trace events in one transaction.
        Validation and conflict checks precede mutations; exceptions roll back,
        and success is returned only after commit and session cleanup. No
        conversation checkpoint, memory, business mutation or delivery occurs.
        Persist owns the single permitted transient retry using the same payload.
        Handoff and trace payloads must be JSON serializable. Outcome and
        handoff_source are type-checked before permitted-value validation.
    """
    allowed_case_outcomes = {
        "resolved",
        "information_requested",
        "escalated",
    }
    ticket_status_by_outcome = {
        "resolved": "resolved",
        "information_requested": "open",
        "escalated": "escalated",
    }
    allowed_handoff_sources = {"generated", "deterministic_fallback"}

    if not isinstance(outcome, dict):
        return {
            "saved": False,
            "error_code": "invalid_outcome_payload",
            "retryable": False,
            "reason": "The persistence outcome must be a dictionary.",
        }

    # `dict.get` returns the value for a key, or None when the key is absent.
    turn_id = outcome.get("turn_id")
    case_outcome = outcome.get("outcome")

    if not isinstance(ticket_id, str) or not ticket_id.strip():
        return {
            "saved": False,
            "error_code": "invalid_ticket_id",
            "retryable": False,
            "reason": "A non-empty ticket ID is required.",
        }

    if not isinstance(turn_id, str) or not turn_id.strip():
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "error_code": "invalid_turn_id",
            "retryable": False,
            "reason": "A non-empty current turn ID is required.",
        }

    if not isinstance(case_outcome, str) or case_outcome not in allowed_case_outcomes:
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "invalid_case_outcome",
            "retryable": False,
            "reason": "The case outcome is not supported.",
        }

    customer_message = outcome.get("customer_message")
    if not isinstance(customer_message, dict):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "invalid_customer_message",
            "retryable": False,
            "reason": "A structured current customer message is required.",
        }

    customer_message_id = customer_message.get("message_id")
    customer_content = customer_message.get("content")
    if (
        not isinstance(customer_message_id, str)
        or not customer_message_id.strip()
        or not isinstance(customer_content, str)
        or not customer_content.strip()
    ):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "invalid_customer_message",
            "retryable": False,
            "reason": "The customer message requires a stable ID and content.",
        }

    if customer_message_id != turn_id:
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "turn_message_id_mismatch",
            "retryable": False,
            "reason": "The current customer message ID must match the turn ID.",
        }

    approved_response = outcome.get("approved_response")
    review_result = outcome.get("review_result")
    response_message_id = None
    response_content = None

    # Normal outcomes must never reach persistence without customer wording.
    if case_outcome in {"resolved", "information_requested"} and not isinstance(
        approved_response, dict
    ):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "missing_approved_response",
            "retryable": False,
            "reason": "This case outcome requires an approved customer response.",
        }

    if approved_response is not None:
        if not isinstance(approved_response, dict):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_approved_response",
                "retryable": False,
                "reason": "The approved response must be a dictionary.",
            }

        response_message_id = approved_response.get("message_id")
        response_draft_id = approved_response.get("draft_id")
        response_content = approved_response.get("content")
        response_outcome = approved_response.get("outcome")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                response_message_id,
                response_draft_id,
                response_content,
            )
        ):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_approved_response",
                "retryable": False,
                "reason": (
                    "The approved response requires stable message and draft IDs "
                    "and non-empty content."
                ),
            }

        if response_outcome != case_outcome:
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "response_outcome_mismatch",
                "retryable": False,
                "reason": "The approved response must match the final case outcome.",
            }

        if (
            not isinstance(review_result, dict)
            or review_result.get("draft_id") != response_draft_id
            or review_result.get("verdict") != "approved"
        ):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "response_not_approved",
                "retryable": False,
                "reason": "The exact response draft does not have approval.",
            }

    handoff = outcome.get("handoff")
    handoff_message_id = None
    handoff_content = None

    if case_outcome == "escalated" and not isinstance(handoff, dict):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "missing_handoff",
            "retryable": False,
            "reason": "An escalated outcome requires a structured human handoff.",
        }

    if case_outcome != "escalated" and handoff is not None:
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "unexpected_handoff",
            "retryable": False,
            "reason": "Only an escalated outcome may contain a human handoff.",
        }

    if handoff is not None:
        handoff_message_id = handoff.get("message_id")
        handoff_turn_id = handoff.get("turn_id")
        handoff_source = handoff.get("handoff_source")
        issue_ids = handoff.get("issue_ids")

        if (
            not isinstance(handoff_message_id, str)
            or not handoff_message_id.strip()
            or handoff_turn_id != turn_id
            or not isinstance(handoff_source, str)
            or handoff_source not in allowed_handoff_sources
            or not isinstance(issue_ids, list)
            or not issue_ids
            or not all(
                isinstance(issue_id, str) and issue_id.strip() for issue_id in issue_ids
            )
        ):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_handoff",
                "retryable": False,
                "reason": (
                    "The handoff requires a stable message ID, the current turn ID, "
                    "issue IDs and a valid handoff source."
                ),
            }

        # The explicit marker prevents an internal handoff from being confused
        # with customer-facing content when both use the ticket_messages table.
        persisted_handoff = {**handoff, "message_type": "human_handoff"}
        try:
            # `json.dumps` converts the structured handoff into storable text.
            handoff_content = json.dumps(persisted_handoff, sort_keys=True)
        except (TypeError, ValueError):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_handoff_payload",
                "retryable": False,
                "reason": "The structured handoff is not JSON serializable.",
            }

    message_ids = [customer_message_id]
    if response_message_id is not None:
        message_ids.append(response_message_id)
    if handoff_message_id is not None:
        message_ids.append(handoff_message_id)
    if len(message_ids) != len(set(message_ids)):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "duplicate_message_id",
            "retryable": False,
            "reason": "Customer, response and handoff message IDs must be distinct.",
        }

    trace_events = outcome.get("trace_events")
    if not isinstance(trace_events, list) or not trace_events:
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "missing_trace_events",
            "retryable": False,
            "reason": "At least one structured trace event is required.",
        }

    prepared_trace_events = []
    event_ids = []
    for trace_event in trace_events:
        if not isinstance(trace_event, dict):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_trace_event",
                "retryable": False,
                "reason": "Every trace event must be a dictionary.",
            }

        event_id = trace_event.get("event_id")
        event_turn_id = trace_event.get("turn_id")
        issue_id = trace_event.get("issue_id")
        node = trace_event.get("node")
        event_type = trace_event.get("event_type")
        event_outcome = trace_event.get("outcome")
        event_reason = trace_event.get("reason")
        event_payload = trace_event.get("payload", {})

        if (
            not isinstance(event_id, str)
            or not event_id.strip()
            or event_turn_id != turn_id
            or not isinstance(node, str)
            or not node.strip()
            or not isinstance(event_type, str)
            or not event_type.strip()
            or (issue_id is not None and not isinstance(issue_id, str))
            or (event_outcome is not None and not isinstance(event_outcome, str))
            or (event_reason is not None and not isinstance(event_reason, str))
            or not isinstance(event_payload, dict)
        ):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_trace_event",
                "retryable": False,
                "reason": "A trace event has invalid searchable fields or payload.",
            }

        try:
            # `json.dumps` verifies that SQLAlchemy's JSON column can encode it.
            json.dumps(event_payload)
        except (TypeError, ValueError):
            return {
                "saved": False,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "error_code": "invalid_trace_payload",
                "retryable": False,
                "reason": "A trace-event payload is not JSON serializable.",
            }

        event_ids.append(event_id)
        prepared_trace_events.append(
            {
                "event_id": event_id,
                "ticket_id": ticket_id,
                "turn_id": turn_id,
                "issue_id": issue_id,
                "node": node,
                "event_type": event_type,
                "outcome": event_outcome,
                "reason": event_reason,
                "payload": event_payload,
            }
        )

    if len(event_ids) != len(set(event_ids)):
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "duplicate_trace_event_id",
            "retryable": False,
            "reason": "Trace-event IDs must be unique within the turn payload.",
        }

    ticket_status = ticket_status_by_outcome[case_outcome]
    messages_to_save = [
        {
            "message_id": customer_message_id,
            "role": udahub.RoleEnum.user,
            "content": customer_content,
        }
    ]
    if response_message_id is not None:
        messages_to_save.append(
            {
                "message_id": response_message_id,
                "role": udahub.RoleEnum.ai,
                "content": response_content,
            }
        )
    if handoff_message_id is not None:
        messages_to_save.append(
            {
                "message_id": handoff_message_id,
                "role": udahub.RoleEnum.system,
                "content": handoff_content,
            }
        )

    try:
        with get_session(engine) as session:
            # Session.get loads one row by its primary key, or returns None.
            ticket = session.get(udahub.Ticket, ticket_id)
            if ticket is None:
                return {
                    "saved": False,
                    "ticket_id": ticket_id,
                    "turn_id": turn_id,
                    "error_code": "missing_ticket",
                    "retryable": False,
                    "reason": "The ticket does not exist.",
                }

            metadata = session.get(udahub.TicketMetadata, ticket_id)
            if metadata is None:
                return {
                    "saved": False,
                    "ticket_id": ticket_id,
                    "turn_id": turn_id,
                    "error_code": "missing_metadata",
                    "retryable": False,
                    "reason": "The ticket has no metadata record to update.",
                }

            # Automated follow-up cannot undo an existing human handoff.
            if metadata.status == "escalated" and case_outcome != "escalated":
                return {
                    "saved": False,
                    "ticket_id": ticket_id,
                    "turn_id": turn_id,
                    "error_code": "ticket_already_escalated",
                    "retryable": False,
                    "reason": "The ticket is already assigned to human handling.",
                }

            new_messages = []
            for message_data in messages_to_save:
                existing_message = session.get(
                    udahub.TicketMessage, message_data["message_id"]
                )
                if existing_message is None:
                    new_messages.append(
                        udahub.TicketMessage(ticket_id=ticket_id, **message_data)
                    )
                elif (
                    existing_message.ticket_id != ticket_id
                    or existing_message.role != message_data["role"]
                    or existing_message.content != message_data["content"]
                ):
                    return {
                        "saved": False,
                        "ticket_id": ticket_id,
                        "turn_id": turn_id,
                        "error_code": "message_id_conflict",
                        "retryable": False,
                        "reason": (
                            "A stable message ID already refers to other content."
                        ),
                    }

            new_trace_events = []
            for event_data in prepared_trace_events:
                existing_event = session.get(udahub.TraceEvent, event_data["event_id"])
                if existing_event is None:
                    new_trace_events.append(udahub.TraceEvent(**event_data))
                elif any(
                    getattr(existing_event, field) != value
                    for field, value in event_data.items()
                ):
                    return {
                        "saved": False,
                        "ticket_id": ticket_id,
                        "turn_id": turn_id,
                        "error_code": "trace_event_id_conflict",
                        "retryable": False,
                        "reason": (
                            "A stable trace-event ID already refers to other data."
                        ),
                    }

            # All conflicts are checked before the first mutation, so an early
            # controlled failure cannot commit only part of the requested outcome.
            # Session.add_all schedules every mapped object for one flush.
            session.add_all(new_messages)
            session.add_all(new_trace_events)
            metadata.status = ticket_status
    except IntegrityError:
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": "database_conflict",
            "retryable": False,
            "reason": "The support outcome conflicts with an existing record.",
        }
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
            "reason": "The support outcome could not be saved.",
        }

    # Reaching this line proves get_session exited without a commit error.
    return {
        "saved": True,
        "ticket_id": ticket_id,
        "turn_id": turn_id,
        "outcome": case_outcome,
        "ticket_status": ticket_status,
        "approved_response": approved_response,
        "handoff_saved": handoff is not None,
        "trace_event_count": len(prepared_trace_events),
        "error_code": None,
        "retryable": False,
        "reason": "The support outcome was committed.",
    }
