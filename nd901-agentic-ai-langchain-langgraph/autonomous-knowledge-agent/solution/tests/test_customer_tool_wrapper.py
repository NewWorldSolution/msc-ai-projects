"""Focused tests for the trusted customer-tool wrapper."""

from agentic.tools import customer_tools


def trusted_config(engine: object | None = None) -> dict:
    """Return one complete trusted call context for the decorated tool."""
    return {
        "configurable": {
            "cultpass_engine": engine if engine is not None else object(),
            "trusted_customer_id": "customer-1",
            "turn_id": "turn-1",
            "issue_id": "issue-1",
            "lookup_key": "lookup_customer_tool:customer_id=customer-1",
            "attempt": 1,
        }
    }


def test_wrapper_calls_shared_query_and_attaches_trusted_provenance(monkeypatch) -> None:
    engine = object()
    calls = []

    def fake_lookup(database_engine, *, customer_id=None, email=None):
        calls.append((database_engine, customer_id, email))
        return {
            "outcome": "found",
            "records": [
                {
                    "user_id": "customer-1",
                    "full_name": "Ada Lovelace",
                    "email": "ada@example.com",
                    "is_blocked": False,
                }
            ],
            "reason": "Customer found.",
            "error_code": None,
            "retryable": False,
        }

    monkeypatch.setattr(customer_tools, "lookup_customer", fake_lookup)

    result = customer_tools.lookup_customer_tool.invoke(
        {"customer_id": "customer-1"}, config=trusted_config(engine)
    )

    assert calls == [(engine, "customer-1", None)]
    assert result["outcome"] == "found"
    assert result["tool_name"] == "lookup_customer_tool"
    assert result["turn_id"] == "turn-1"
    assert result["issue_id"] == "issue-1"
    assert result["lookup_key"] == "lookup_customer_tool:customer_id=customer-1"
    assert result["attempt"] == 1


def test_wrapper_rejects_customer_id_switch_before_query(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("A mismatched customer ID must not reach the query.")

    monkeypatch.setattr(customer_tools, "lookup_customer", fail_if_called)

    result = customer_tools.lookup_customer_tool.invoke(
        {"customer_id": "customer-2"}, config=trusted_config()
    )

    assert result["outcome"] == "technical_failure"
    assert result["records"] == []
    assert result["error_code"] == "customer_scope_mismatch"
    assert result["retryable"] is False


def test_wrapper_discards_email_candidate_with_wrong_ticket_identity(monkeypatch) -> None:
    def mismatched_lookup(database_engine, *, customer_id=None, email=None):
        return {
            "outcome": "found",
            "records": [
                {
                    "user_id": "customer-2",
                    "full_name": "Wrong Customer",
                    "email": email,
                    "is_blocked": False,
                }
            ],
            "reason": "Customer found.",
            "error_code": None,
            "retryable": False,
        }

    monkeypatch.setattr(customer_tools, "lookup_customer", mismatched_lookup)

    result = customer_tools.lookup_customer_tool.invoke(
        {"email": "wrong@example.com"}, config=trusted_config()
    )

    assert result["outcome"] == "technical_failure"
    assert result["records"] == []
    assert result["error_code"] == "customer_scope_mismatch"


def test_wrapper_returns_controlled_failure_without_exception_details(monkeypatch) -> None:
    def failing_lookup(*args, **kwargs):
        raise RuntimeError("secret database address")

    monkeypatch.setattr(customer_tools, "lookup_customer", failing_lookup)

    result = customer_tools.lookup_customer_tool.invoke(
        {"customer_id": "customer-1"}, config=trusted_config()
    )

    assert result["outcome"] == "technical_failure"
    assert result["error_code"] == "unexpected_customer_lookup_error"
    assert result["retryable"] is False
    assert "secret database address" not in result["reason"]


def test_wrapper_fails_closed_when_trusted_context_is_incomplete(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("An incomplete context must not reach the query.")

    monkeypatch.setattr(customer_tools, "lookup_customer", fail_if_called)

    result = customer_tools.lookup_customer_tool.invoke(
        {"customer_id": "customer-1"},
        config={"configurable": {"thread_id": "ticket-1"}},
    )

    assert result["outcome"] == "technical_failure"
    assert result["records"] == []
    assert result["error_code"] == "invalid_tool_context"
    assert result["retryable"] is False
