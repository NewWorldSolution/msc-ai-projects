"""Tests for deterministic escalation fallback handoffs."""

from langchain_core.messages import HumanMessage

from agentic.tools.handoff_tools import assemble_fallback_handoff


def test_fallback_copies_verified_state_and_excludes_rejected_draft() -> None:
    state = {
        "ticket_id": "ticket-1",
        "turn_id": "turn-1",
        "messages": [HumanMessage(content="My account is blocked.", id="turn-1")],
        "escalation_reason": "Blocked accounts require human handling.",
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
                "missing_details": ["Manual unblock authority"],
                "article_ids": ["article-1"],
            }
        ],
        "tool_results": [
            {
                "turn_id": "turn-1",
                "issue_id": "issue-1",
                "tool_name": "customer_lookup",
                "lookup_key": "customer-1",
                "outcome": "found",
                "records": [{"account_status": "blocked"}],
                "attempt": 1,
            }
        ],
        "retrieved_articles": [{"article_id": "article-1"}],
        "response_draft": {"content": "Rejected wording."},
        "review_history": [{"reason": "Rejected wording was unsupported."}],
    }

    handoff = assemble_fallback_handoff(state)

    assert handoff["message_id"] == "turn-1:handoff"
    assert handoff["issue_ids"] == ["issue-1"]
    assert handoff["request"] == "My account is blocked."
    assert handoff["classification"]["urgency"] == "urgent"
    assert handoff["verified_facts"][0]["records"] == [{"account_status": "blocked"}]
    assert handoff["source_references"] == ["article-1"]
    assert handoff["gaps"] == ["Manual unblock authority"]
    assert "Rejected wording" not in str(handoff)


def test_unavailable_classification_is_explicit_and_issue_id_is_stable() -> None:
    state = {
        "ticket_id": "ticket-2",
        "turn_id": "turn-2",
        "messages": [HumanMessage(content="Please help.", id="turn-2")],
        "classification_status": "failed",
        "classification": {"reason": "The classification call failed."},
        "issues": [],
        "tool_results": [],
        "retrieved_articles": [],
        "escalation_reason": "Normal processing could not continue.",
    }

    handoff = assemble_fallback_handoff(state)

    assert handoff["issue_ids"] == ["turn-2:unclassified"]
    assert handoff["classification"] == {
        "status": "failed",
        "urgency": None,
        "urgency_reason": None,
        "complexity": None,
        "unavailable_reason": "The classification call failed.",
    }
