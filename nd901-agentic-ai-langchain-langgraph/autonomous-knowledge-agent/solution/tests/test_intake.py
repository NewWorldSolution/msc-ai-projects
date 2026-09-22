"""Focused tests for deterministic Intake preparation."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from agentic import intake
from data.models import cultpass, udahub
from utils import get_session


def make_engine():
    """Return one shared-connection in-memory SQLite engine."""
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def seeded_resources(*, include_customer: bool = True):
    """Build trusted UDA-Hub/CultPass resources for one ticket."""
    udahub_engine = make_engine()
    cultpass_engine = make_engine()
    udahub.Base.metadata.create_all(udahub_engine)
    cultpass.Base.metadata.create_all(cultpass_engine)

    with get_session(udahub_engine) as session:
        session.add(
            udahub.Account(account_id="account-1", account_name="CultPass Support")
        )
        session.add(
            udahub.User(
                user_id="support-user-1",
                account_id="account-1",
                external_user_id="customer-1",
                user_name="Ada Support",
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
                main_issue_type="subscription",
                tags="subscription",
            )
        )

    if include_customer:
        with get_session(cultpass_engine) as session:
            session.add(
                cultpass.User(
                    user_id="customer-1",
                    full_name="Ada Lovelace",
                    email="ada@example.com",
                    is_blocked=False,
                )
            )

    return udahub_engine, cultpass_engine


def trusted_config(udahub_engine, cultpass_engine, *, thread_id="ticket-1"):
    """Return Intake's trusted runtime configuration."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "udahub_engine": udahub_engine,
            "cultpass_engine": cultpass_engine,
        }
    }


def current_state():
    """Return a new turn carrying old checkpointed state that Intake must reset."""
    return {
        "messages": [
            HumanMessage(content="Earlier request", id="turn-0"),
            AIMessage(content="Earlier approved reply", id="turn-0:response"),
            HumanMessage(content="What is my status?", id="turn-1"),
        ],
        "classification_status": "completed",
        "classification": {"urgency": "normal"},
        "issues": [{"issue_id": "old-issue"}],
        "assignment": {"agent": "knowledge"},
        "retrieved_articles": [{"article_id": "old-article"}],
        "technical_attempts": {"old": 2},
        "workflow_steps": 20,
        "model_calls": 8,
        "response_stage": "original",
        "response_draft": {"content": "old draft"},
        "draft_blocker": {"reason": "old blocker"},
        "review_result": {"verdict": "reject"},
        "review_history": [{"verdict": "reject"}],
        "original_review_attempts": 2,
        "acknowledgement_review_attempts": 1,
        "approved_response": {"content": "old reply"},
        "outcome": "resolved",
        "escalation_reason": "old reason",
        "acknowledgement_permitted": True,
        "messaging_disabled": True,
        "handoff": {"message_id": "old-handoff"},
        "persistence_result": {"saved": True},
        "identity_correction": {"asked": True},
        "tool_results": [{"turn_id": "turn-0", "outcome": "found"}],
        "trace": [{"event_id": "turn-0:intake", "turn_id": "turn-0"}],
    }


def test_prepare_case_loads_scoped_facts_memory_and_resets_turn_state() -> None:
    udahub_engine, cultpass_engine = seeded_resources()
    store = InMemoryStore()
    store.put(
        ("customer_memory", "account-1", "customer-1"),
        "memory-1",
        {"memory_type": "preference", "content": "Prefers email."},
    )
    state = current_state()

    result = intake.prepare_case(
        state,
        trusted_config(udahub_engine, cultpass_engine),
        store=store,
    )

    assert result["preparation_result"]["outcome"] == "ready"
    assert result["ticket_id"] == "ticket-1"
    assert result["account_id"] == "account-1"
    assert result["customer_id"] == "customer-1"
    assert result["turn_id"] == "turn-1"
    assert result["channel"] == "chat"
    assert result["account_context"]["user_id"] == "customer-1"
    assert result["account_context"]["is_blocked"] is False
    assert isinstance(result["account_context"]["created_at"], str)
    assert result["customer_memory"][0]["memory_id"] == "memory-1"

    assert result["classification_status"] == "pending"
    assert result["classification"] is None
    assert result["issues"] == []
    assert result["assignment"] is None
    assert result["retrieved_articles"] == []
    assert result["technical_attempts"] == {
        "turn-1:intake:ticket:ticket-1": 1,
        "turn-1:intake:customer:customer-1": 1,
    }
    assert result["workflow_steps"] == 0
    assert result["model_calls"] == 0
    assert result["review_history"] == []
    assert result["original_review_attempts"] == 0
    assert result["acknowledgement_review_attempts"] == 0
    assert result["outcome"] is None
    assert result["messaging_disabled"] is False
    assert result["persistence_result"] is None

    assert "messages" not in result
    assert "tool_results" not in result
    assert "identity_correction" not in result
    assert result["trace"] == [
        {
            "event_id": "turn-1:intake:preparation",
            "turn_id": "turn-1",
            "issue_id": None,
            "node": "intake",
            "event_type": "case_preparation",
            "outcome": "ready",
            "reason": "Ticket and customer facts are prepared.",
            "payload": {
                "lookup_attempts": {"ticket": 1, "customer": 1},
                "ticket_lookup_outcome": "success",
                "customer_lookup_outcome": "found",
                "memory_recall_outcome": "completed",
                "error_code": None,
                "memory_error_code": None,
                "retryable": False,
            },
        }
    ]


def test_prepare_case_requires_latest_message_to_be_stable_human(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Invalid message input must stop before database access.")

    monkeypatch.setattr(intake, "load_ticket_context", fail_if_called)
    state = {"messages": [AIMessage(content="Not a new customer turn", id="reply-1")]}

    result = intake.prepare_case(
        state,
        {"configurable": {"thread_id": "ticket-1"}},
        store=InMemoryStore(),
    )

    assert result["preparation_result"]["outcome"] == "failed"
    assert result["preparation_result"]["error_code"] == "invalid_current_message"
    assert "turn_id" not in result
    assert "trace" not in result


def test_prepare_case_reports_inaccessible_ticket_without_fabricating_identity() -> (
    None
):
    udahub_engine, cultpass_engine = seeded_resources()
    state = {"messages": [HumanMessage(content="Help", id="turn-1")]}

    result = intake.prepare_case(
        state,
        trusted_config(
            udahub_engine,
            cultpass_engine,
            thread_id="missing-ticket",
        ),
        store=InMemoryStore(),
    )

    assert result["preparation_result"]["outcome"] == "failed"
    assert result["preparation_result"]["ticket_lookup_outcome"] == "missing_ticket"
    assert result["preparation_result"]["error_code"] == "missing_ticket"
    assert "ticket_id" not in result
    assert "account_id" not in result
    assert "customer_id" not in result


def test_prepare_case_keeps_customer_not_found_distinct_from_memory() -> None:
    udahub_engine, cultpass_engine = seeded_resources(include_customer=False)
    store = InMemoryStore()
    store.put(
        ("customer_memory", "account-1", "customer-1"),
        "memory-1",
        {"memory_type": "resolved_issue", "content": "Historical context."},
    )

    result = intake.prepare_case(
        {"messages": [HumanMessage(content="Help", id="turn-1")]},
        trusted_config(udahub_engine, cultpass_engine),
        store=store,
    )

    assert result["preparation_result"]["outcome"] == "customer_not_found"
    assert result["preparation_result"]["customer_lookup_outcome"] == "not_found"
    assert result["preparation_result"]["memory_recall_outcome"] == "completed"
    assert result["customer_memory"][0]["content"] == "Historical context."
    assert "user_id" not in result["account_context"]


def test_prepare_case_reports_memory_failure_without_discarding_current_facts(
    monkeypatch,
) -> None:
    udahub_engine, cultpass_engine = seeded_resources()

    def failing_recall(*args, **kwargs):
        raise RuntimeError("private store endpoint")

    monkeypatch.setattr(intake, "search_customer_memory", failing_recall)

    result = intake.prepare_case(
        {"messages": [HumanMessage(content="Help", id="turn-1")]},
        trusted_config(udahub_engine, cultpass_engine),
        store=InMemoryStore(),
    )

    assert result["preparation_result"]["outcome"] == "ready"
    assert result["preparation_result"]["memory_recall_outcome"] == "technical_failure"
    assert result["preparation_result"]["memory_error_code"] == "memory_recall_error"
    assert "private store endpoint" not in result["preparation_result"]["reason"]
    assert result["account_context"]["user_id"] == "customer-1"
    assert result["customer_memory"] == []


def test_prepare_case_rejects_customer_record_outside_ticket_scope(monkeypatch) -> None:
    udahub_engine, cultpass_engine = seeded_resources()

    def wrong_customer(*args, **kwargs):
        return {
            "outcome": "found",
            "records": [{"user_id": "customer-2", "is_blocked": False}],
            "reason": "Customer found.",
            "error_code": None,
            "retryable": False,
        }

    monkeypatch.setattr(intake, "lookup_customer", wrong_customer)

    result = intake.prepare_case(
        {"messages": [HumanMessage(content="Help", id="turn-1")]},
        trusted_config(udahub_engine, cultpass_engine),
        store=InMemoryStore(),
    )

    assert result["preparation_result"]["outcome"] == "failed"
    assert result["preparation_result"]["error_code"] == "customer_scope_mismatch"
    assert "user_id" not in result["account_context"]


def test_prepare_case_emits_no_duplicate_trace_delta_for_same_turn() -> None:
    udahub_engine, cultpass_engine = seeded_resources()
    state = {"messages": [HumanMessage(content="Help", id="turn-1")]}

    first = intake.prepare_case(
        state,
        trusted_config(udahub_engine, cultpass_engine),
        store=InMemoryStore(),
    )
    state["trace"] = first["trace"]

    second = intake.prepare_case(
        state,
        trusted_config(udahub_engine, cultpass_engine),
        store=InMemoryStore(),
    )

    assert "trace" not in second
