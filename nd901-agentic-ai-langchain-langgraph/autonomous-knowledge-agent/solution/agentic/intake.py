"""Deterministic ticket loading, customer-only lookup and memory recall."""

from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agentic.state import CaseState
from agentic.tools.customer_tools import lookup_customer
from agentic.tools.memory_tools import search_customer_memory
from agentic.tools.ticket_tools import load_ticket_context


def _lookup_with_retry(
    operation: Callable[[], dict[str, Any]],
    *,
    attempts: dict[str, int],
    lookup_key: str,
    failure_outcome: str,
    unexpected_error_code: str,
    reason: str,
) -> dict[str, Any]:
    """Run a lookup at most twice, counting attempts under a stable turn key.

    The caller supplies the trusted operation and a copy of the current turn's
    counters. Only a classified technical failure can spend the second attempt.
    Unknown exceptions are sanitized, and exhausted keys never access storage.
    """
    used = attempts.get(lookup_key, 0)
    if not isinstance(used, int) or isinstance(used, bool) or used < 0:
        # why: Malformed checkpoint counters cannot grant a fresh retry allowance.
        used = 2
    result = {
        "outcome": failure_outcome,
        "records": [],
        "reason": "The lookup attempt allowance is exhausted.",
        "error_code": "lookup_attempts_exhausted",
        "retryable": False,
    }
    while used < 2:
        used += 1
        attempts[lookup_key] = used
        try:
            result = operation()
        except Exception:
            return {
                "outcome": failure_outcome,
                "records": [],
                "reason": reason,
                "error_code": unexpected_error_code,
                "retryable": False,
            }
        if not isinstance(result, dict):
            return {
                "outcome": failure_outcome,
                "records": [],
                "reason": "The lookup returned an invalid result.",
                "error_code": "invalid_lookup_result",
                "retryable": False,
            }
        if result.get("outcome") != failure_outcome or result.get("retryable") is not True:
            break
    return result


def prepare_case(
    state: CaseState, config: RunnableConfig, *, store: BaseStore
) -> CaseState:
    """Prepare one trusted customer turn and return its state delta.

    Args:
        state: Checkpointed case state whose latest message must be a HumanMessage
            with a stable ID.
        config: Runtime configuration containing the checkpoint thread ID and the
            trusted UDA-Hub and CultPass database engines.
        store: Injected long-term store used for account/customer-scoped recall.

    Returns:
        A CaseState delta with refreshed ticket/customer facts, historical memory,
        new-turn controls, a controlled preparation result and at most one
        new trace event. Preparation is ready, customer_not_found or failed; memory
        recall failure is reported separately from verified current facts.

    Notes:
        Ticket history is not rehydrated, and subscriptions or reservations are
        not prefetched. Omitting messages, archived tool results and the
        ticket-scoped identity-correction record preserves their existing state.
        A prepared same-turn reentry returns no delta; it cannot renew budgets or
        overwrite terminal decisions. Each lookup permits one classified transient
        retry, recorded in technical_attempts and preparation provenance.
        Raw database or store exceptions are never exposed.
    """
    # api: LangGraph injects the compiled BaseStore into the named store parameter.
    messages = state.get("messages", [])
    current_message = messages[-1] if isinstance(messages, list) and messages else None
    # api: getattr(object, name, default) returns the default when the attribute
    # is absent instead of raising AttributeError.
    turn_id = getattr(current_message, "id", None)

    if (
        getattr(current_message, "type", None) != "human"
        or not isinstance(turn_id, str)
        or not turn_id.strip()
    ):
        return {
            # why: Omitted keys survive graph merging; clear prior release signals.
            "outcome": None,
            "approved_response": None,
            "handoff": None,
            "persistence_result": None,
            "messaging_disabled": True,
            "preparation_result": {
                "outcome": "failed",
                "reason": "Intake requires a current HumanMessage with a stable ID.",
                "error_code": "invalid_current_message",
                "retryable": False,
                "ticket_lookup_outcome": "not_attempted",
                "customer_lookup_outcome": "not_attempted",
                "memory_recall_outcome": "not_attempted",
            }
        }

    configurable = config.get("configurable", {})
    trusted_context = configurable if isinstance(configurable, dict) else {}
    thread_id = trusted_context.get("thread_id")
    udahub_engine = trusted_context.get("udahub_engine")
    cultpass_engine = trusted_context.get("cultpass_engine")

    is_new_turn = state.get("turn_id") != turn_id
    prior_preparation = state.get("preparation_result")
    if (
        not is_new_turn
        and isinstance(prior_preparation, dict)
        and prior_preparation.get("thread_id") == thread_id
    ):
        # why: A completed preparation (including failure) has spent its allowance.
        return {}

    # why: Only a new human-message ID resets current-turn controls. History and
    # ticket-scoped identity correction are preserved by omitting them from the delta.
    update: CaseState = {}
    if is_new_turn:
        update = {
            "turn_id": turn_id,
            "account_context": {},
            "customer_memory": [],
            "memory_updates": [],
            "classification_status": "pending",
            "classification": None,
            "issues": [],
            "assignment": None,
            "retrieved_articles": [],
            "technical_attempts": {},
            "workflow_steps": 0,
            "model_calls": 0,
            "response_stage": None,
            "response_draft": None,
            "draft_blocker": None,
            "review_result": None,
            "review_history": [],
            "original_review_attempts": 0,
            "acknowledgement_review_attempts": 0,
            "approved_response": None,
            "outcome": None,
            "escalation_reason": None,
            "acknowledgement_permitted": False,
            "messaging_disabled": False,
            "handoff": None,
            "persistence_result": None,
        }
    attempts = dict(update.get("technical_attempts", state.get("technical_attempts", {})))
    lookup_attempts = {"ticket": 0, "customer": 0}

    ticket_lookup_outcome = "not_attempted"
    customer_lookup_outcome = "not_attempted"
    memory_recall_outcome = "not_attempted"
    memory_error_code = None

    preparation_outcome = "failed"
    preparation_reason = "Intake could not prepare the current turn."
    preparation_error_code: str | None = "intake_preparation_failed"
    preparation_retryable = False

    if not isinstance(thread_id, str) or not thread_id.strip():
        preparation_reason = "A non-empty checkpoint thread ID is required."
        preparation_error_code = "invalid_thread_id"
    elif udahub_engine is None or cultpass_engine is None:
        preparation_reason = "Trusted Intake database context is incomplete."
        preparation_error_code = "invalid_intake_context"
    else:
        ticket_key = f"{turn_id}:intake:ticket:{thread_id}"
        # api: lambda creates a deferred call so both attempts reuse trusted arguments.
        ticket_result = _lookup_with_retry(
            lambda: load_ticket_context(thread_id, udahub_engine),
            attempts=attempts,
            lookup_key=ticket_key,
            failure_outcome="database_error",
            unexpected_error_code="unexpected_ticket_lookup_error",
            reason="The ticket context could not be loaded.",
        )
        lookup_attempts["ticket"] = attempts.get(ticket_key, 0)

        if not isinstance(ticket_result, dict):
            ticket_result = {
                "outcome": "database_error",
                "reason": "The ticket lookup returned an invalid result.",
                "error_code": "invalid_ticket_lookup_result",
                "retryable": False,
            }

        ticket_lookup_outcome = ticket_result.get("outcome", "database_error")
        if ticket_lookup_outcome != "success":
            preparation_reason = ticket_result.get(
                "reason", "The ticket is invalid or inaccessible."
            )
            preparation_error_code = ticket_result.get(
                "error_code", str(ticket_lookup_outcome)
            )
            preparation_retryable = ticket_result.get("retryable") is True
        else:
            ticket_id = ticket_result.get("ticket_id")
            account_id = ticket_result.get("account_id")
            channel = ticket_result.get("channel")
            mapped_customer_id = ticket_result.get("external_user_id")

            valid_ticket_context = (
                ticket_id == thread_id
                and isinstance(account_id, str)
                and bool(account_id.strip())
                and isinstance(channel, str)
                and bool(channel.strip())
            )
            if not valid_ticket_context:
                preparation_reason = "The ticket lookup returned incomplete identity."
                preparation_error_code = "invalid_ticket_context"
            else:
                customer_id = (
                    mapped_customer_id
                    if isinstance(mapped_customer_id, str)
                    and bool(mapped_customer_id.strip())
                    else None
                )
                update.update(
                    {
                        "ticket_id": ticket_id,
                        "account_id": account_id,
                        "customer_id": customer_id,
                        "channel": channel,
                        "account_context": {
                            "account_id": account_id,
                            "account_name": ticket_result.get("account_name"),
                            "support_user_id": ticket_result.get("support_user_id"),
                            "support_user_name": ticket_result.get("support_user_name"),
                            "ticket_status": ticket_result.get("ticket_status"),
                            "main_issue_type": ticket_result.get("main_issue_type"),
                            "tags": ticket_result.get("tags"),
                        },
                    }
                )

                customer_key = f"{turn_id}:intake:customer:{customer_id}"
                customer_result = _lookup_with_retry(
                    lambda: lookup_customer(cultpass_engine, customer_id=customer_id),
                    attempts=attempts,
                    lookup_key=customer_key,
                    failure_outcome="technical_failure",
                    unexpected_error_code="unexpected_customer_lookup_error",
                    reason="The CultPass customer lookup could not be completed.",
                )
                lookup_attempts["customer"] = attempts.get(customer_key, 0)

                if not isinstance(customer_result, dict) or not isinstance(
                    customer_result.get("records"), list
                ):
                    customer_result = {
                        "outcome": "technical_failure",
                        "records": [],
                        "reason": "The customer lookup returned an invalid result.",
                        "error_code": "invalid_customer_lookup_result",
                        "retryable": False,
                    }

                customer_lookup_outcome = customer_result.get(
                    "outcome", "technical_failure"
                )
                customer_records = customer_result["records"]

                if customer_lookup_outcome == "found":
                    valid_customer_record = (
                        len(customer_records) == 1
                        and isinstance(customer_records[0], dict)
                        and customer_records[0].get("user_id") == customer_id
                    )
                    if not valid_customer_record:
                        preparation_reason = "The customer result does not match the trusted ticket identity."
                        preparation_error_code = "customer_scope_mismatch"
                    else:
                        allowed_customer_fields = {
                            "user_id",
                            "full_name",
                            "email",
                            "is_blocked",
                            "created_at",
                            "updated_at",
                        }
                        customer_record: dict[str, Any] = {}
                        serializable = True
                        for field, value in customer_records[0].items():
                            if field not in allowed_customer_fields:
                                continue
                            if isinstance(value, date):
                                value = value.isoformat()
                            elif value is not None and not isinstance(
                                value, (str, int, float, bool)
                            ):
                                serializable = False
                                break
                            customer_record[field] = value

                        if not serializable:
                            preparation_reason = (
                                "The customer lookup returned unsupported field values."
                            )
                            preparation_error_code = "invalid_customer_record"
                        else:
                            update["account_context"].update(customer_record)
                            preparation_outcome = "ready"
                            preparation_reason = (
                                "Ticket and customer facts are prepared."
                            )
                            preparation_error_code = None
                elif customer_lookup_outcome == "not_found":
                    preparation_outcome = "customer_not_found"
                    preparation_reason = customer_result.get(
                        "reason", "The mapped CultPass customer was not found."
                    )
                    preparation_error_code = None
                elif customer_lookup_outcome in {
                    "missing_input",
                    "technical_failure",
                }:
                    preparation_reason = customer_result.get(
                        "reason", "The customer lookup could not prepare trusted facts."
                    )
                    preparation_error_code = customer_result.get(
                        "error_code", "customer_lookup_failed"
                    )
                    preparation_retryable = customer_result.get("retryable") is True
                else:
                    preparation_reason = (
                        "The customer lookup returned an unsupported outcome."
                    )
                    preparation_error_code = "invalid_customer_lookup_outcome"

                if customer_id is None:
                    memory_recall_outcome = "skipped"
                else:
                    try:
                        recalled_memory = search_customer_memory(
                            account_id, customer_id, store=store
                        )
                        if not isinstance(recalled_memory, list) or not all(
                            isinstance(memory, dict) for memory in recalled_memory
                        ):
                            raise TypeError(
                                "Customer memory must be a list of records."
                            )
                        update["customer_memory"] = recalled_memory
                        memory_recall_outcome = "completed"
                    except Exception:
                        # why: Historical memory is auxiliary context. Its failure
                        # does not erase separately verified current customer facts.
                        update["customer_memory"] = []
                        memory_recall_outcome = "technical_failure"
                        memory_error_code = "memory_recall_error"

    update["technical_attempts"] = attempts
    preparation_result = {
        "thread_id": thread_id,
        "lookup_attempts": lookup_attempts,
        "outcome": preparation_outcome,
        "reason": preparation_reason,
        "error_code": preparation_error_code,
        "retryable": preparation_retryable,
        "ticket_lookup_outcome": ticket_lookup_outcome,
        "customer_lookup_outcome": customer_lookup_outcome,
        "memory_recall_outcome": memory_recall_outcome,
        "memory_error_code": memory_error_code,
    }
    update["preparation_result"] = preparation_result

    preparation_event = {
        "event_id": f"{turn_id}:intake:preparation",
        "turn_id": turn_id,
        "issue_id": None,
        "node": "intake",
        "event_type": "case_preparation",
        "outcome": preparation_outcome,
        "reason": preparation_reason,
        "payload": {
            "lookup_attempts": lookup_attempts,
            "ticket_lookup_outcome": ticket_lookup_outcome,
            "customer_lookup_outcome": customer_lookup_outcome,
            "memory_recall_outcome": memory_recall_outcome,
            "error_code": preparation_error_code,
            "memory_error_code": memory_error_code,
            "retryable": preparation_retryable,
        },
    }
    prior_event_ids = {
        event.get("event_id")
        for event in state.get("trace", [])
        if isinstance(event, dict)
    }
    if preparation_event["event_id"] not in prior_event_ids:
        update["trace"] = [preparation_event]

    return update
