"""Tests for scoped, evidenced customer memory."""

import pytest
from langgraph.store.memory import InMemoryStore

from agentic.tools.memory_tools import (
    save_customer_memory,
    search_customer_memory,
)


def resolved_memory() -> dict:
    """Return one eligible resolved-issue memory record."""
    return {
        "memory_id": "turn-1:issue-1",
        "memory_type": "resolved_issue",
        "content": "Password reset guidance resolved the sign-in issue.",
        "source_refs": ["turn-1", "article-1"],
        "ticket_id": "ticket-1",
        "turn_id": "turn-1",
        "outcome": "resolved",
    }


def test_save_and_recall_are_scoped_by_account_and_customer() -> None:
    store = InMemoryStore()

    result = save_customer_memory(
        "account-1",
        "customer-1",
        resolved_memory(),
        store=store,
    )

    assert result["saved"] is True
    assert search_customer_memory("account-1", "customer-1", store=store)[0][
        "content"
    ].startswith("Password reset")
    assert search_customer_memory("account-1", "customer-2", store=store) == []
    assert search_customer_memory("account-2", "customer-1", store=store) == []


def test_same_memory_id_and_content_are_idempotent() -> None:
    store = InMemoryStore()
    memory = resolved_memory()

    first = save_customer_memory("account-1", "customer-1", memory, store=store)
    second = save_customer_memory("account-1", "customer-1", memory, store=store)

    assert first["saved"] is True
    assert second["saved"] is True
    assert len(search_customer_memory("account-1", "customer-1", store=store)) == 1


def test_same_memory_id_cannot_replace_different_content() -> None:
    store = InMemoryStore()
    original = resolved_memory()
    conflicting = resolved_memory()
    conflicting["content"] = "A different claim using the same stable ID."

    first = save_customer_memory("account-1", "customer-1", original, store=store)
    second = save_customer_memory("account-1", "customer-1", conflicting, store=store)

    assert first["saved"] is True
    assert second["saved"] is False
    assert second["error_code"] == "memory_id_conflict"
    recalled = search_customer_memory("account-1", "customer-1", store=store)
    assert recalled[0]["content"] == original["content"]


def test_unresolved_or_unsourced_memory_is_rejected() -> None:
    store = InMemoryStore()
    escalated = resolved_memory()
    escalated["outcome"] = "escalated"
    unsourced = resolved_memory()
    unsourced["source_refs"] = []

    escalated_result = save_customer_memory(
        "account-1", "customer-1", escalated, store=store
    )
    unsourced_result = save_customer_memory(
        "account-1", "customer-1", unsourced, store=store
    )

    assert escalated_result["error_code"] == "ineligible_memory_outcome"
    assert unsourced_result["error_code"] == "missing_memory_sources"
    assert search_customer_memory("account-1", "customer-1", store=store) == []


def test_backend_write_failure_is_controlled() -> None:
    class FailingStore:
        def __init__(self):
            self.put_called = False

        def get(self, namespace, key):
            # An absent record lets save_customer_memory reach the write.
            return None

        def put(self, namespace, key, value):
            self.put_called = True
            raise RuntimeError("backend details must not escape")

    store = FailingStore()
    result = save_customer_memory(
        "account-1",
        "customer-1",
        resolved_memory(),
        store=store,
    )

    assert store.put_called is True
    assert result == {
        "saved": False,
        "memory_id": "turn-1:issue-1",
        "error_code": "memory_write_error",
        "retryable": False,
        "reason": "The eligible customer memory could not be saved.",
    }


def test_recall_failure_remains_distinct_from_empty_memory() -> None:
    class FailingStore:
        def search(self, namespace, *, limit):
            raise RuntimeError("store unavailable")

    with pytest.raises(RuntimeError, match="store unavailable"):
        search_customer_memory(
            "account-1",
            "customer-1",
            store=FailingStore(),
        )
