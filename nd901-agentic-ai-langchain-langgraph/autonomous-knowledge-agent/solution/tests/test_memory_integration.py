"""Memory integration with scripted reviews and real SQLite writes."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from agentic import intake
from agentic.agents import supervisor
from agentic.persist import persist_case
from agentic.tools.memory_tools import search_customer_memory
from data.models import udahub
from utils import get_session

from test_intake import seeded_resources, trusted_config
from test_model_budget import case, assessment
from test_supervisor_routes import config
from test_supervisor_review import review_case, model_with, assessment_result, merge


def test_review_receives_historical_memory(review_case):
    review_case["customer_memory"] = [{"content": "Please keep replies concise."}]
    model = model_with(assessment_result())
    supervisor.review_response(review_case, model=model, reserve_model_call=Mock())
    # `call_args.args` exposes the mock's latest positional arguments.
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0]
    context = json.loads(prompt.messages[-1].content.split("\n", 1)[1])
    assert context["historical_customer_memory"] == review_case["customer_memory"]


@pytest.mark.parametrize("changed", [False, True])
def test_approval_replay_checks_the_reviewed_memory(review_case, config, changed):
    review_case["customer_memory"] = [{"content": "Please keep replies concise."}]
    first = supervisor.supervise_case(
        review_case, config, model=model_with(assessment_result())
    )
    updated = merge(review_case, first)
    if changed:
        updated["customer_memory"][0]["content"] = "Please give detailed replies."
    model = Mock()
    replay = supervisor.supervise_case(updated, config, model=model)
    assert replay.goto == ("escalation" if changed else "persist")
    if changed:
        assert replay.update["approved_response"] is None
    assert not model.mock_calls


@pytest.mark.parametrize("resolved", [False, True])
def test_memory_candidates_save_and_recall_on_another_ticket(review_case, config, resolved):
    core, business = seeded_resources()
    store = InMemoryStore()
    try:
        with get_session(core) as session:
            session.add_all([
                udahub.Ticket(
                    ticket_id="ticket-2", account_id="account-1",
                    user_id="support-user-1", channel="chat",
                ),
                udahub.TicketMetadata(
                    ticket_id="ticket-2", status="open", tags="returning-customer",
                ),
            ])
        quote = "Please keep replies concise."
        review_case["messages"] = [
            HumanMessage(content="Help me sign in. " + quote, id="turn-1")
        ]
        review_case["classification"]["stated_preferences"] = [quote]
        review_case["memory_updates"] = []
        if resolved:
            review_case["outcome"] = "resolved"
            review_case["response_draft"].update(
                outcome="resolved", addressed_issue_ids=["turn-1:issue:1"],
                body="Use the verified password reset steps.",
            )
            review_case["issues"][0].update(
                coverage="supported", article_ids=["article-1"]
            )
            review_case["retrieved_articles"] = [{
                "article_id": "article-1",
                "excerpt": "Use the verified password reset steps.",
                "gate_pass": True,
            }]
        before = deepcopy(review_case)
        result = supervisor.supervise_case(
            review_case, config,
            model=model_with(assessment_result(outcome=review_case["outcome"])),
        )
        assert review_case == before
        candidates = result.update.get("memory_updates", [])
        assert len(candidates) == (2 if resolved else 0)

        saved_state = merge(review_case, result)
        saved = persist_case(saved_state, config, udahub_engine=core, store=store)
        assert saved["persistence_result"]["saved"] is True
        if resolved:
            assert saved["persistence_result"]["memory"]["saved"] == 2
            repeated = persist_case(saved_state, config, udahub_engine=core, store=store)
            assert repeated["persistence_result"]["saved"] is True

        returning = intake.prepare_case(
            {"messages": [
                HumanMessage(content="I need reset guidance again.", id="turn-2")
            ]},
            trusted_config(core, business, thread_id="ticket-2"),
            store=store,
        )
        assert returning["preparation_result"]["outcome"] == "ready"
        assert len(returning["customer_memory"]) == (2 if resolved else 0)
        assert returning["account_context"]["is_blocked"] is False
        assert not search_customer_memory("account-1", "another-customer", store=store)
    finally:
        core.dispose()
        business.dispose()


def test_existing_escalation_avoids_model_calls(case, config):
    case["account_context"]["ticket_status"] = "escalated"
    model = Mock()
    result = supervisor.supervise_case(case, config, model=model)
    assert result.goto == "persist"
    assert result.update["classification_status"] == "skipped"
    assert result.update["approved_response"] is None
    assert result.update["memory_updates"] == []
    assert not model.mock_calls
