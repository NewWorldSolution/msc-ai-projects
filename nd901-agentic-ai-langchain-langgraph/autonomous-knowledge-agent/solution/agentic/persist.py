"""Persist approved support outcomes and return the conversation delta."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from sqlalchemy import Engine

from agentic.state import CaseState
from agentic.tools.handoff_tools import assemble_fallback_handoff
from agentic.tools.memory_tools import save_customer_memory
from agentic.tools.ticket_tools import save_ticket_outcome


_CASE_OUTCOMES = {"resolved", "information_requested", "escalated"}


def _trace_delta(
    state: CaseState, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return only events whose stable IDs are not already in graph state."""

    existing_ids = {
        event.get("event_id")
        for event in state.get("trace", [])
        if isinstance(event, dict)
    }
    # why: The trace reducer concatenates rather than merging by ID, so a repeated
    # node invocation must not append the same checkpoint event a second time.
    return [event for event in events if event.get("event_id") not in existing_ids]


def _failure_update(
    state: CaseState,
    *,
    error_code: str,
    reason: str,
    retryable: bool = False,
    attempts: int = 0,
) -> CaseState:
    """Build a controlled terminal persistence failure with no message delta."""

    ticket_id = state.get("ticket_id")
    turn_id = state.get("turn_id")
    safe_turn_id = turn_id if isinstance(turn_id, str) and turn_id else "unknown-turn"
    failure_event = {
        "event_id": f"{safe_turn_id}:persist:failure",
        "turn_id": safe_turn_id,
        "issue_id": None,
        "node": "persist",
        "event_type": "persistence_failed",
        "outcome": state.get("outcome"),
        "reason": reason,
        "payload": {
            "error_code": error_code,
            "retryable": retryable,
            "attempts": attempts,
        },
    }
    return {
        "persistence_result": {
            "saved": False,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "error_code": error_code,
            "retryable": retryable,
            "attempts": attempts,
            "reason": reason,
        },
        "trace": _trace_delta(state, [failure_event]),
    }


def _current_customer_content(state: CaseState, turn_id: str) -> str | None:
    """Find the current text-only human message by its stable turn ID."""

    # why: Conversation history can contain older human turns and approved replies;
    # persistence must save the human message that owns this turn, not messages[-1].
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) != "human" or message.id != turn_id:
            continue
        return message.content if isinstance(message.content, str) else None
    return None


def _memory_summary(
    state: CaseState, *, store: BaseStore
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write eligible resolved-turn memory after the support commit."""

    updates = state.get("memory_updates", [])
    if state.get("outcome") != "resolved":
        return {
            "status": "skipped",
            "attempted": 0,
            "saved": 0,
            "failed": 0,
            "results": [],
            "reason": "Only resolved turns can create long-term memory.",
        }, []

    if not isinstance(updates, list):
        result = {
            "saved": False,
            "memory_id": None,
            "error_code": "invalid_memory_updates",
            "retryable": False,
            "reason": "Memory updates must be supplied as a list.",
        }
        return {
            "status": "failed",
            "attempted": 1,
            "saved": 0,
            "failed": 1,
            "results": [result],
        }, [result]

    if not updates:
        return {
            "status": "not_requested",
            "attempted": 0,
            "saved": 0,
            "failed": 0,
            "results": [],
        }, []

    account_id = state.get("account_id")
    customer_id = state.get("customer_id")
    ticket_id = state.get("ticket_id")
    turn_id = state.get("turn_id")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (account_id, customer_id, ticket_id, turn_id)
    ):
        result = {
            "saved": False,
            "memory_id": None,
            "error_code": "missing_memory_scope",
            "retryable": False,
            "reason": "Resolved-turn memory requires trusted account and customer scope.",
        }
        return {
            "status": "failed",
            "attempted": len(updates),
            "saved": 0,
            "failed": len(updates),
            "results": [result],
        }, [result]

    results = []
    for memory in updates:
        if not isinstance(memory, dict):
            results.append(
                {
                    "saved": False,
                    "memory_id": None,
                    "error_code": "invalid_memory_payload",
                    "retryable": False,
                    "reason": "Each memory update must be a dictionary.",
                }
            )
            continue

        # why: Scope and outcome come from trusted state. A candidate cannot relabel
        # itself as belonging to another ticket/customer or to an unresolved turn.
        scoped_memory = {
            **memory,
            "ticket_id": ticket_id,
            "turn_id": turn_id,
            "outcome": "resolved",
        }
        results.append(
            save_customer_memory(
                account_id,
                customer_id,
                scoped_memory,
                store=store,
            )
        )

    saved_count = sum(result.get("saved") is True for result in results)
    failed_results = [result for result in results if result.get("saved") is not True]
    if not failed_results:
        status = "succeeded"
    elif saved_count:
        status = "partial_failure"
    else:
        status = "failed"
    return {
        "status": status,
        "attempted": len(results),
        "saved": saved_count,
        "failed": len(failed_results),
        "results": results,
    }, failed_results


def persist_case(
    state: CaseState,
    config: RunnableConfig,
    *,
    udahub_engine: Engine,
    store: BaseStore,
) -> CaseState:
    """Execute Supervisor's approved final decision and report the actual save.

    The memory summary is created after the support commit and returned in the
    graph trace for checkpointing; it is not part of that atomic SQL transaction.
    A saved fallback handoff is also returned so checkpoint and support data agree.
    """

    ticket_id = state.get("ticket_id")
    turn_id = state.get("turn_id")
    case_outcome = state.get("outcome")
    configurable = config.get("configurable", {})
    thread_id = (
        configurable.get("thread_id") if isinstance(configurable, dict) else None
    )

    if not isinstance(ticket_id, str) or not ticket_id.strip():
        return _failure_update(
            state,
            error_code="invalid_ticket_id",
            reason="Persist requires a trusted non-empty ticket ID.",
        )
    if thread_id != ticket_id:
        return _failure_update(
            state,
            error_code="thread_ticket_mismatch",
            reason="The checkpoint thread ID must match the trusted ticket ID.",
        )
    if not isinstance(turn_id, str) or not turn_id.strip():
        return _failure_update(
            state,
            error_code="invalid_turn_id",
            reason="Persist requires the current turn's stable message ID.",
        )
    if not isinstance(case_outcome, str) or case_outcome not in _CASE_OUTCOMES:
        return _failure_update(
            state,
            error_code="invalid_case_outcome",
            reason="Supervisor has not supplied a supported final outcome.",
        )

    customer_content = _current_customer_content(state, turn_id)
    if not isinstance(customer_content, str) or not customer_content.strip():
        return _failure_update(
            state,
            error_code="missing_current_customer_message",
            reason="The current text customer message was not found by turn ID.",
        )

    approved_response = state.get("approved_response")
    review_result = state.get("review_result")
    handoff = state.get("handoff")
    if case_outcome == "escalated" and not isinstance(handoff, dict):
        # why: This is a defensive terminal path. It uses only existing verified
        # state and does not spend another model call after escalation is authorized.
        handoff = assemble_fallback_handoff(state)

    persistence_event = {
        "event_id": f"{turn_id}:persist:outcome",
        "turn_id": turn_id,
        "issue_id": None,
        "node": "persist",
        "event_type": "final_outcome",
        "outcome": case_outcome,
        "reason": "Supervisor-authorized outcome committed with its support records.",
        "payload": {
            "approved_response_present": isinstance(approved_response, dict),
            "handoff_present": isinstance(handoff, dict),
        },
    }

    current_turn_events = [
        event
        for event in state.get("trace", [])
        if isinstance(event, dict) and event.get("turn_id") == turn_id
    ]
    prior_persistence_event = next(
        (
            event
            for event in current_turn_events
            if event.get("event_id") == persistence_event["event_id"]
        ),
        None,
    )
    if (
        prior_persistence_event is not None
        and prior_persistence_event != persistence_event
    ):
        return _failure_update(
            state,
            error_code="persistence_event_id_conflict",
            reason="The stable persistence event ID already has different content.",
        )
    if prior_persistence_event is None:
        current_turn_events.append(persistence_event)

    # why: The batch is built once. A permitted retry sends this exact object again,
    # so message, handoff and event identifiers cannot drift between attempts.
    outcome_payload = {
        "turn_id": turn_id,
        "outcome": case_outcome,
        "customer_message": {
            "message_id": turn_id,
            "content": customer_content,
        },
        "approved_response": approved_response,
        "review_result": review_result,
        "handoff": handoff,
        "trace_events": current_turn_events,
    }

    core_result = save_ticket_outcome(ticket_id, outcome_payload, udahub_engine)
    attempts = 1
    if core_result.get("saved") is not True and core_result.get("retryable") is True:
        core_result = save_ticket_outcome(ticket_id, outcome_payload, udahub_engine)
        attempts = 2

    if core_result.get("saved") is not True:
        return _failure_update(
            state,
            error_code=core_result.get("error_code", "persistence_failed"),
            reason=core_result.get("reason", "The support outcome was not saved."),
            retryable=bool(core_result.get("retryable")),
            attempts=attempts,
        )

    # why: Long-term memory is a separate, auxiliary store. It starts only after
    # the atomic support transaction commits and cannot roll that transaction back.
    memory, memory_failures = _memory_summary(state, store=store)
    persistence_result = {**core_result, "attempts": attempts, "memory": memory}
    trace_events = [persistence_event]

    if memory["attempted"]:
        memory_event = {
            "event_id": f"{turn_id}:persist:memory",
            "turn_id": turn_id,
            "issue_id": None,
            "node": "persist",
            "event_type": "memory_write_summary",
            "outcome": case_outcome,
            "reason": (
                "Eligible customer memory was saved."
                if not memory_failures
                else "One or more customer-memory writes failed after the core commit."
            ),
            "payload": {
                "attempted": memory["attempted"],
                "saved": memory["saved"],
                "failed": memory["failed"],
                "error_codes": [result.get("error_code") for result in memory_failures],
            },
        }
        trace_events.append(memory_event)

    update: CaseState = {
        "handoff": handoff,
        "persistence_result": persistence_result,
        "trace": _trace_delta(state, trace_events),
    }
    persisted_response = core_result.get("approved_response")
    if isinstance(persisted_response, dict):
        # api: AIMessage's id is the stable key used by add_messages to merge retries.
        update["messages"] = [
            AIMessage(
                content=persisted_response["content"],
                id=persisted_response["message_id"],
            )
        ]
    return update
