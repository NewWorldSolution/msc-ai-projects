"""Scoped historical customer memory through the graph's supplied store."""

from typing import Any

from langgraph.store.base import BaseStore


_MEMORY_NAMESPACE = "customer_memory"
_MEMORY_TYPES = {"preference", "resolved_issue"}


def search_customer_memory(
    account_id: str, customer_id: str, *, store: BaseStore
) -> list[dict[str, Any]]:
    """Recall up to 100 historical records for one account and customer.

    Args:
        account_id: Trusted non-empty support account ID.
        customer_id: Trusted non-empty external customer ID.
        store: Caller-supplied LangGraph store.

    Returns:
        Stored values ordered by stable store key, with memory_id taken from
        that key. An empty list means the search completed without matches.

    Raises:
        ValueError: An account or customer ID is missing or unusable.
        Exception: Store failures propagate for the Intake caller to handle
            separately from an empty successful recall.

    Notes:
        Reads the customer_memory/account/customer namespace prefix and performs
        no writes. Results are historical context, not current business evidence.
        Stored values are expected to be dictionary records.
    """

    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("Customer-memory recall requires a non-empty account ID.")
    if not isinstance(customer_id, str) or not customer_id.strip():
        raise ValueError("Customer-memory recall requires a non-empty customer ID.")

    namespace = (_MEMORY_NAMESPACE, account_id, customer_id)
    # api: store.search returns records whose namespace starts with this tuple.
    items = store.search(namespace, limit=100)
    # why: The stable store key is returned with each value so callers retain
    # provenance and can distinguish otherwise similar historical memories.
    return [
        {**item.value, "memory_id": item.key}
        for item in sorted(items, key=lambda item: item.key)
    ]


def save_customer_memory(
    account_id: str,
    customer_id: str,
    memory: dict[str, Any],
    *,
    store: BaseStore,
) -> dict[str, Any]:
    """Validate and save one historical memory from a resolved support turn.

    Args:
        account_id: Trusted non-empty account ID supplied by Persist.
        customer_id: Trusted non-empty customer ID supplied by Persist.
        memory: Dictionary containing memory_id, memory_type (preference or
            resolved_issue), content, source_refs, ticket_id, turn_id and
            outcome=resolved. References must be non-empty strings.
        store: Caller-supplied LangGraph store.

    Returns:
        A dictionary with saved, memory_id, error_code, retryable and reason;
        success also includes memory_type. An identical existing record is an
        idempotent success; conflicting content under the same ID is rejected.
        Store read/write exceptions become sanitized memory_write_error results.

    Notes:
        Writes only the scoped long-term memory store, after the caller confirms
        support persistence. No retries or business writes occur. The caller
        owns factual eligibility and source verification; this helper checks
        reference shape, not the referenced evidence. Extra payload fields are
        preserved. Memory type is validated as a string before checking its
        permitted values.
    """

    if (
        not isinstance(account_id, str)
        or not account_id.strip()
        or not isinstance(customer_id, str)
        or not customer_id.strip()
    ):
        return {
            "saved": False,
            "memory_id": None,
            "error_code": "invalid_memory_scope",
            "retryable": False,
            "reason": "A non-empty account and customer ID are required.",
        }
    if not isinstance(memory, dict):
        return {
            "saved": False,
            "memory_id": None,
            "error_code": "invalid_memory_payload",
            "retryable": False,
            "reason": "Customer memory must be a dictionary.",
        }

    memory_id = memory.get("memory_id")
    memory_type = memory.get("memory_type")
    content = memory.get("content")
    source_refs = memory.get("source_refs")
    ticket_id = memory.get("ticket_id")
    turn_id = memory.get("turn_id")
    outcome = memory.get("outcome")

    if (
        not isinstance(memory_id, str)
        or not memory_id.strip()
        or not isinstance(memory_type, str)
        or memory_type not in _MEMORY_TYPES
        or not isinstance(content, str)
        or not content.strip()
        or not isinstance(ticket_id, str)
        or not ticket_id.strip()
        or not isinstance(turn_id, str)
        or not turn_id.strip()
    ):
        return {
            "saved": False,
            "memory_id": memory_id,
            "error_code": "invalid_memory_payload",
            "retryable": False,
            "reason": (
                "Memory requires stable IDs, an allowed type, content, ticket and turn."
            ),
        }
    if outcome != "resolved":
        return {
            "saved": False,
            "memory_id": memory_id,
            "error_code": "ineligible_memory_outcome",
            "retryable": False,
            "reason": "Only a resolved support turn may create long-term memory.",
        }
    if (
        not isinstance(source_refs, list)
        or not source_refs
        or not all(
            isinstance(source_ref, str) and source_ref.strip()
            for source_ref in source_refs
        )
    ):
        return {
            "saved": False,
            "memory_id": memory_id,
            "error_code": "missing_memory_sources",
            "retryable": False,
            "reason": "Memory requires at least one non-empty source reference.",
        }

    namespace = (_MEMORY_NAMESPACE, account_id, customer_id)
    stored_memory = {
        **memory,
        "memory_id": memory_id,
        "memory_type": memory_type,
        "content": content,
        "source_refs": source_refs,
        "ticket_id": ticket_id,
        "turn_id": turn_id,
        "outcome": outcome,
    }
    try:
        # api: store.get returns the item at one exact namespace/key or None.
        existing = store.get(namespace, memory_id)
        if existing is not None and existing.value != stored_memory:
            return {
                "saved": False,
                "memory_id": memory_id,
                "error_code": "memory_id_conflict",
                "retryable": False,
                "reason": "The stable memory ID already refers to different content.",
            }
        if existing is None:
            # api: store.put writes one dictionary under a namespace and stable key.
            store.put(namespace, memory_id, stored_memory)
    except Exception:
        # why: Store implementations can expose different backend exceptions. This
        # auxiliary boundary reports a controlled failure without masking core save.
        return {
            "saved": False,
            "memory_id": memory_id,
            "error_code": "memory_write_error",
            "retryable": False,
            "reason": "The eligible customer memory could not be saved.",
        }

    return {
        "saved": True,
        "memory_id": memory_id,
        "memory_type": memory_type,
        "error_code": None,
        "retryable": False,
        "reason": "The eligible customer memory was saved.",
    }
