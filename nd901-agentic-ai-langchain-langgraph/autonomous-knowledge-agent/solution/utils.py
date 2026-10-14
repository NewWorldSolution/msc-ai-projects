# reset_udahub.py
import os
import logging
from uuid import uuid4
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import (
    HumanMessage,
)
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def routing_path(trace_events: list[dict], *, channel: str | None = None,
                 outcome: str | None = None, reply_present: bool = False) -> str:
    """Describe one turn's route through the agents as readable text.

    Built only from the Supervisor's routing_decision trace events, in the order they
    were written, so the line is evidence from the log rather than a description of the
    design. The customer message arrives through the channel, Intake prepares it and every
    specialist returns to Supervisor until Persist saves the turn.
    """
    steps = [channel or "message", "intake", "supervisor"]
    for event in trace_events:
        if not isinstance(event, dict) or event.get("event_type") != "routing_decision":
            continue
        destination = (event.get("payload") or {}).get("destination")
        if not destination:
            continue
        steps.append(destination)
        if destination != "persist":
            steps.append("supervisor")
    if steps[-1] == "persist":
        if outcome == "escalated":
            steps.append("human support" + (" + customer (receipt)" if reply_present else " (no customer reply)"))
        else:
            steps.append({"resolved": "customer (answer)",
                          "information_requested": "customer (question)"}.get(outcome, "end"))
    return " -> ".join(steps)


def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    """Run chat turns and present only a saved, current-turn approved response.

    Intake receives a stable human-message ID and the existing ticket ID.
    Failed execution/saving stops the session; handoff-only turns produce an
    operator diagnostic. Persist owns conversation updates. Presentation events
    are logged without message text or raw provider exceptions.
    """
    # GetLogger creates a named logger using the application's logging setup.
    logger = logging.getLogger(__name__)
    while True:
        user_input = input("User: ")
        print("User:", user_input)
        if user_input.lower() in ("quit", "exit", "q"):
            print("Assistant: Goodbye!")
            break

        # Uuid4 creates a unique ID retained for this turn's graph invocation.
        turn_id = str(uuid4())
        trigger = {"messages": [HumanMessage(content=user_input, id=turn_id)]}
        config = {"configurable": {"thread_id": ticket_id}}
        try:
            result = agent.invoke(input=trigger, config=config)
        except Exception:
            logger.error("workflow_failed ticket=%s turn=%s; no reply presented", ticket_id, turn_id)
            break

        persistence = result.get("persistence_result") if isinstance(result, dict) else None
        if (
            not isinstance(persistence, dict)
            or persistence.get("saved") is not True
            or persistence.get("ticket_id") != ticket_id
            or persistence.get("turn_id") != turn_id
            or result.get("turn_id") != turn_id
        ):
            logger.error("turn_not_saved ticket=%s turn=%s; no reply presented", ticket_id, turn_id)
            break

        approved = result.get("approved_response")
        if approved is None:
            logger.warning("handoff_only ticket=%s turn=%s; no approved reply", ticket_id, turn_id)
            continue
        if (
            not isinstance(approved, dict)
            or approved.get("message_id") != f"{turn_id}:reply"
            or not isinstance(approved.get("content"), str)
            or not approved["content"].strip()
        ):
            logger.error("invalid_approved_response ticket=%s turn=%s; no reply presented", ticket_id, turn_id)
            break

        # A conversation's last message is not proof of approval or saving.
        try:
            print("Assistant:", approved["content"])
        except OSError:
            logger.error("presentation_failed ticket=%s turn=%s", ticket_id, turn_id)
            raise
        logger.info("presentation_succeeded ticket=%s turn=%s", ticket_id, turn_id)


def batch_interface(
    orchestrator: CompiledStateGraph,
    udahub_engine: Engine,
    *,
    account_id: str,
) -> list[dict]:
    """Process stored customer messages and display each saved result in order.

    Read unprocessed user messages from open tickets. Keep the stored message ID
    as turn_id and the ticket ID as thread_id. Persist owns writes; its final
    audit event marks a completed turn. Report skipped/failed cases explicitly.
    Existing incomplete checkpoints require inspection, not automatic restart.
    Closed tickets stay closed. No inbox access, email delivery or new retries.
    """
    from data.models import udahub

    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("A non-empty account_id is required.")
    logger = logging.getLogger(__name__)

    with get_session(udahub_engine) as session:
        # A clarification can be completed while its ticket remains open.
        completed = session.query(udahub.TraceEvent.ticket_id, udahub.TraceEvent.turn_id).join(
            udahub.Ticket, udahub.Ticket.ticket_id == udahub.TraceEvent.ticket_id,
        ).filter(
            udahub.Ticket.account_id == account_id,
            udahub.TraceEvent.node == "persist",
            udahub.TraceEvent.event_type == "final_outcome",
        ).all()
        completed_turns = set(completed)
        rows = (
            session.query(udahub.TicketMessage, udahub.Ticket.channel)
            .join(udahub.Ticket,
                  udahub.Ticket.ticket_id == udahub.TicketMessage.ticket_id)
            .join(udahub.TicketMetadata,
                  udahub.TicketMetadata.ticket_id == udahub.Ticket.ticket_id)
            .filter(
                udahub.Ticket.account_id == account_id,
                udahub.TicketMetadata.status == "open",
                udahub.TicketMessage.role == udahub.RoleEnum.user,
            )
            .order_by(udahub.TicketMessage.created_at, udahub.Ticket.created_at,
                      udahub.Ticket.ticket_id, udahub.TicketMessage.message_id)
            .all()
        )
        queue = [(message.ticket_id, message.message_id, message.content, channel)
                 for message, channel in rows
                 if (message.ticket_id, message.message_id) not in completed_turns]

    results = []

    def display(report):
        """Record each result; print customer wording only after release checks."""
        results.append(report)
        try:
            print(f"\nTicket: {report['ticket_id']}")
            print("Outcome:", report.get("outcome", report["status"]))
            if report.get("trace") is not None:
                print("Routing:", routing_path(report["trace"], channel=report.get("channel"),
                                               outcome=report.get("outcome"),
                                               reply_present=report["reply"] is not None))
            if report["reply"] is not None:
                print(report["reply"])
                logger.info("presentation_succeeded ticket=%s turn=%s",
                            report["ticket_id"], report["turn_id"])
            elif report["status"] == "handoff_only":
                print("Handoff saved for human review. No approved customer reply.")
            elif report["status"] == "skipped":
                print("Message skipped:", report["reason"])
            else:
                print("Processing failed. Error code:", report["error_code"])
        except OSError as exc:
            logger.error("presentation_failed ticket=%s turn=%s exception_type=%s",
                         report["ticket_id"], report["turn_id"], type(exc).__name__)
            raise

    failed_tickets = set()
    for ticket_id, turn_id, content, channel in queue:
        report = {"ticket_id": ticket_id, "turn_id": turn_id,
                  "channel": channel, "status": "failed", "reply": None}
        if ticket_id in failed_tickets:
            report.update(status="skipped", reason="An earlier message on this ticket failed.")
            logger.info("message_skipped ticket=%s turn=%s reason=earlier_failure", ticket_id, turn_id)
            display(report)
            continue
        with get_session(udahub_engine) as session:
            metadata = session.get(udahub.TicketMetadata, ticket_id)
            already_saved = session.query(udahub.TraceEvent).filter_by(
                ticket_id=ticket_id, turn_id=turn_id,
                node="persist", event_type="final_outcome",
            ).first()
            ticket_status = metadata.status if metadata else None
        if ticket_status is None:
            report["error_code"] = "missing_metadata"
            failed_tickets.add(ticket_id)
            logger.error("missing_metadata ticket=%s turn=%s", ticket_id, turn_id)
            display(report)
            continue
        if ticket_status != "open" or already_saved is not None:
            reason = "The ticket is already closed." if ticket_status != "open" else "This message was already processed."
            report.update(status="skipped", reason=reason)
            logger.info("message_skipped ticket=%s turn=%s ticket_status=%s", ticket_id, turn_id, ticket_status)
            display(report)
            continue
        if not isinstance(content, str) or not content.strip():
            report["error_code"] = "empty_customer_message"
            failed_tickets.add(ticket_id)
            logger.error("empty_customer_message ticket=%s turn=%s", ticket_id, turn_id)
            display(report)
            continue
        config = {"configurable": {"thread_id": ticket_id}}
        try:
            # `get_state` reads the thread's saved values and any unfinished nodes.
            snapshot = orchestrator.get_state(config)
        except Exception as exc:
            report["error_code"] = "checkpoint_read_failed"
            failed_tickets.add(ticket_id)
            logger.error("checkpoint_read_failed ticket=%s turn=%s exception_type=%s",
                         ticket_id, turn_id, type(exc).__name__)
            display(report)
            continue
        if snapshot.next or snapshot.values.get("turn_id") == turn_id:
            report["error_code"] = "incomplete_previous_turn"
            failed_tickets.add(ticket_id)
            logger.error("incomplete_previous_turn ticket=%s turn=%s", ticket_id, turn_id)
            display(report)
            continue
        try:
            # `invoke` runs the connected graph with the stored customer message.
            result = orchestrator.invoke(
                input={"messages": [HumanMessage(content=content, id=turn_id)]},
                config=config,
            )
        except Exception as exc:
            report["error_code"] = "workflow_failed"
            failed_tickets.add(ticket_id)
            logger.error("workflow_failed ticket=%s turn=%s exception_type=%s",
                         ticket_id, turn_id, type(exc).__name__)
            display(report)
            continue
        persistence = result.get("persistence_result") if isinstance(result, dict) else None
        if (not isinstance(persistence, dict) or persistence.get("saved") is not True
                or persistence.get("ticket_id") != ticket_id
                or persistence.get("turn_id") != turn_id
                or result.get("turn_id") != turn_id):
            report["error_code"] = "turn_not_saved"
        else:
            approved = result.get("approved_response")
            if approved is None:
                if result.get("outcome") == "escalated" and persistence.get("handoff_saved") is True:
                    report["status"] = "handoff_only"
                else:
                    report["error_code"] = "missing_approved_reply"
            elif (not isinstance(approved, dict)
                  or approved.get("message_id") != f"{turn_id}:reply"
                  or not isinstance(approved.get("content"), str)
                  or not approved["content"].strip()):
                report["error_code"] = "invalid_approved_reply"
            else:
                report.update(status="processed", reply=approved["content"])
        if report["status"] == "failed":
            failed_tickets.add(ticket_id)
            logger.error("%s ticket=%s turn=%s", report["error_code"], ticket_id, turn_id)
        else:
            report.update(
                outcome=result.get("outcome"), model_calls=result.get("model_calls"),
                persistence_result=persistence,
                recalled_at_intake=result.get("customer_memory", []),
                memory_candidates=result.get("memory_updates", []),
                trace=[event for event in result.get("trace", [])
                       if isinstance(event, dict) and event.get("turn_id") == turn_id],
            )
            logger.info("turn_completed ticket=%s turn=%s outcome=%s reply_present=%s",
                        ticket_id, turn_id, report["outcome"], report["reply"] is not None)
        display(report)
    return results
