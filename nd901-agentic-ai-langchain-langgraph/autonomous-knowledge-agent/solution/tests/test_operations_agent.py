"""Operations integration: fake models and real read-only SQLite tools."""

from copy import deepcopy
import json
from threading import Event
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import event

from agentic.agents import operations, supervisor
from agentic.tools import subscription_tools
from test_review_regressions import business_engine


class ScriptedOperationsModel:
    """Return explicit responses while recording model-visible context."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []
        self.bound_names = []

    def bind_tools(self, tools):
        self.bound_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages, config=None):
        self.messages.append(deepcopy(messages))
        if not self.responses:
            raise AssertionError("Unexpected extra model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return response


def request(name="lookup_subscription", args=None, call_id="call1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args or {}, "id": call_id, "type": "tool_call",
    }])


def done():
    return AIMessage(content="Lookup findings are complete.")


@pytest.fixture
def case():
    return {
        "ticket_id": "ticket1", "account_id": "a1", "customer_id": "c1", "turn_id": "turn1",
        "messages": [HumanMessage(content="Read my requested business facts", id="turn1")],
        "assignment": {"agent": "operations", "task": "Establish the assigned facts",
                       "issue_ids": ["issue1"], "lookup_modes": {"issue1": {
                           "lookup_reservations": "single", "lookup_experiences": "single"}}},
        "issues": [{"issue_id": "issue1", "required_business_facts": ["Stored facts"],
                    "category": "subscription", "needs_operations": True, "missing_details": []}],
        "preparation_result": {"outcome": "ready", "ticket_lookup_outcome": "success"},
        "classification_status": "completed", "account_context": {"is_blocked": False},
        "messaging_disabled": False, "model_calls": 0, "technical_attempts": {},
        "trace": [], "tool_results": [],
    }


@pytest.fixture
def config(business_engine):
    return {"configurable": {"thread_id": "ticket1", "cultpass_engine": business_engine,
                            "limits": {"max_model_calls": 12, "max_graph_steps": 10,
                                       "call_timeout_seconds": 1.0}}}


def run(case, config, *responses):
    before = deepcopy(case)
    original_config = dict(config["configurable"])
    model = ScriptedOperationsModel(*responses)
    result = operations.handle_operations(case, config, model=model)
    assert case == before
    assert config["configurable"] == original_config
    assert "messages" not in result and "assignment" not in result and "outcome" not in result
    json.dumps(result, allow_nan=False)
    return result, model


def error(result):
    return result["trace"][-1]["payload"]["error"]


@pytest.mark.parametrize("name,args,outcome", [
    ("lookup_customer_tool", {}, "found"),
    ("lookup_subscription", {}, "found"),
    ("lookup_reservations", {"reservation_id": "r1"}, "found"),
    ("lookup_reservations", {"reservation_id": "other"}, "not_found"),
    ("lookup_experiences", {"experience_id": "e1"}, "found"),
    ("lookup_experiences", {"experience_id": "absent"}, "not_found"),
    ("lookup_experiences", {}, "missing_input"),
    ("lookup_experiences", {"title": "Concert"}, "ambiguous"),
])
def test_real_tools_and_outcomes(case, config, name, args, outcome):
    responses = [request(name, args)]
    if outcome in {"found", "not_found"}:
        responses.append(done())
    result, model = run(case, config, *responses)
    assert error(result) is None
    finding = result["tool_results"][0]
    assert finding["outcome"] == outcome
    assert finding["tool_name"] == name
    assert finding["issue_id"] == "issue1" and finding["turn_id"] == "turn1"
    assert finding["attempt"] == 1
    assert result["model_calls"] == len(responses)
    assert set(model.bound_names) == {"lookup_customer_tool", "lookup_subscription",
                                      "lookup_reservations", "lookup_experiences"}


def test_list_experiences_requires_no_customer_and_only_reads(case, config, business_engine):
    case["customer_id"] = None
    case["assignment"]["lookup_modes"]["issue1"]["lookup_experiences"] = "list"
    statements = []
    def observe(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(business_engine, "before_cursor_execute", observe)
    try:
        result, _ = run(case, config, request("lookup_experiences", {"title": "Concert"}), done())
    finally:
        event.remove(business_engine, "before_cursor_execute", observe)
    assert error(result) is None
    assert len(result["tool_results"][0]["records"]) == 2
    assert statements and all(s.lstrip().upper().startswith("SELECT") for s in statements)


@pytest.mark.parametrize("args,expected", [
    ({"customer_id": "c2"}, "operations_customer_scope_mismatch"),
    ({"trusted_customer_id": "c2"}, "operations_invalid_tool_call_arguments"),
    ({"customer_id": 12}, "operations_invalid_tool_arguments"),
])
def test_invalid_requests_never_access_database(case, config, business_engine, args, expected):
    statements = []
    def observe(*args): statements.append(True)
    event.listen(business_engine, "before_cursor_execute", observe)
    try:
        result, _ = run(case, config, request(args=args))
    finally:
        event.remove(business_engine, "before_cursor_execute", observe)
    assert error(result)["error_code"] == expected
    assert not result["tool_results"] and not statements


def test_no_mode_and_unpermitted_tool_fail(case, config):
    case["assignment"]["lookup_modes"]["issue1"] = {}
    result, _ = run(case, config, request("lookup_experiences", {"experience_id": "e1"}))
    assert error(result)["error_code"] == "operations_lookup_mode_unavailable"
    result, _ = run(case, config, request("cancel_reservation"))
    assert error(result)["error_code"] == "operations_unpermitted_tool"


def test_no_tool_calls_cannot_invent_success(case, config):
    result, _ = run(case, config, done())
    assert error(result)["error_code"] == "operations_no_verified_findings"
    assert result["tool_results"] == []


def test_dependent_lookup_receives_real_tool_message(case, config):
    def next_lookup(messages):
        finding = json.loads(messages[-1].content)
        assert isinstance(messages[-1], ToolMessage)
        assert finding["records"][0]["experience_id"] == "e1"
        return request("lookup_experiences", {"experience_id": finding["records"][0]["experience_id"]}, "call2")
    result, model = run(case, config, request("lookup_reservations", {"reservation_id": "r1"}), next_lookup, done())
    assert error(result) is None
    assert len(result["tool_results"]) == 2 and len(model.messages) == 3


@pytest.mark.parametrize("name,args", [("lookup_subscription", {}), ("lookup_experiences", {"experience_id":"absent"})])
def test_completed_results_reused_without_query_or_duplicate_delta(case, config, business_engine, name, args):
    first, _ = run(case, config, request(name, args), done())
    case["tool_results"] = first["tool_results"]
    case["technical_attempts"] = first["technical_attempts"]
    statements = []
    def observe(*args): statements.append(True)
    event.listen(business_engine, "before_cursor_execute", observe)
    try:
        result, _ = run(case, config, request(name, args), done())
    finally:
        event.remove(business_engine, "before_cursor_execute", observe)
    assert error(result) is None and result["tool_results"] == [] and statements == []
    assert result["trace"][0]["payload"]["reused"] is True


def test_changed_turn_or_mode_does_not_reuse(case, config):
    first, _ = run(case, config, request("lookup_experiences", {"title":"Concert"}))
    case["tool_results"] = first["tool_results"]
    case["assignment"]["lookup_modes"]["issue1"]["lookup_experiences"] = "list"
    result, _ = run(case, config, request("lookup_experiences", {"title":"Concert"}), done())
    assert result["tool_results"][0]["outcome"] == "found"
    assert result["tool_results"][0]["lookup_key"] != first["tool_results"][0]["lookup_key"]
    case["tool_results"][0]["turn_id"] = "old"
    case["assignment"]["lookup_modes"]["issue1"]["lookup_experiences"] = "single"
    result, _ = run(case, config, request("lookup_experiences", {"title":"Concert"}))
    assert result["tool_results"]


def test_two_issues_share_lookup_but_have_separate_provenance(case, config):
    case["issues"].append({"issue_id":"issue2", "required_business_facts":["Subscription"]})
    case["assignment"]["issue_ids"].append("issue2")
    case["assignment"]["lookup_modes"]["issue2"] = {}
    result, _ = run(case, config, request(), done(), request(), done())
    assert error(result) is None
    assert [r["issue_id"] for r in result["tool_results"]] == ["issue1","issue2"]
    assert result["trace"][1]["payload"]["reused"] is True
    assert len(result["technical_attempts"]) == 1
    case["tool_results"] = result["tool_results"]
    case["technical_attempts"] = result["technical_attempts"]
    repeated, _ = run(case, config, request(), done(), request(), done())
    assert repeated["tool_results"] == []


@pytest.mark.parametrize("retryable,count", [(True,2),(False,1)])
def test_database_failures_retry_only_when_tool_classifies_transient(case, config, monkeypatch, retryable, count):
    from sqlalchemy.exc import SQLAlchemyError, TimeoutError
    failing = Mock(side_effect=TimeoutError("SECRET") if retryable else SQLAlchemyError("SECRET"))
    monkeypatch.setattr(subscription_tools, "get_session", failing)
    result, _ = run(case, config, request())
    assert failing.call_count == count
    assert len(result["tool_results"]) == count
    assert all(r["outcome"] == "technical_failure" for r in result["tool_results"])
    assert "SECRET" not in json.dumps(result)
    assert error(result)["error_code"] == "operations_lookup_failed"
    case["tool_results"] = result["tool_results"]
    case["technical_attempts"] = result["technical_attempts"]
    run(case, config, request())
    assert failing.call_count == count  # Re-entry grants no additional database attempt.


def test_transient_database_retry_can_succeed(case, config, monkeypatch):
    from sqlalchemy.exc import TimeoutError
    original = subscription_tools.get_session
    calls = []
    def flaky(engine):
        calls.append(1)
        if len(calls) == 1: raise TimeoutError("SECRET")
        return original(engine)
    monkeypatch.setattr(subscription_tools, "get_session", flaky)
    result, _ = run(case, config, request(), done())
    assert error(result) is None
    assert [r["outcome"] for r in result["tool_results"]] == ["technical_failure","found"]
    assert [r["attempt"] for r in result["tool_results"]] == [1,2]


def test_model_failures_and_budget_are_counted(case, config):
    result, _ = run(case, config, TimeoutError("SECRET"), request(), done())
    assert error(result) is None and result["model_calls"] == 3
    result, _ = run(case, config, RuntimeError("SECRET"))
    assert error(result) and result["model_calls"] == 1
    assert "SECRET" not in json.dumps(result)
    case["model_calls"] = config["configurable"]["limits"]["max_model_calls"]
    result, model = run(case, config)
    assert not model.messages
    assert error(result)["error_code"] == "operations_model_budget_exhausted"


def test_timeout_does_not_retry_an_outstanding_call(case, config):
    release = Event()
    completed = Event()
    def slow(messages):
        release.wait(2)
        completed.set()
        return done()
    config["configurable"]["limits"]["call_timeout_seconds"] = 0.01
    try:
        result, model = run(case, config, slow)
        assert error(result)["error_code"] == "operations_model_timeout"
        assert result["model_calls"] == 1 and len(model.messages) == 1
    finally:
        release.set()
        assert completed.wait(1)


def test_repeated_requests_and_step_limit_are_bounded(case, config):
    result, model = run(case, config, request(), request())
    assert error(result)["error_code"] == "operations_repeated_lookup"
    assert len(result["tool_results"]) == 1 and len(model.messages) == 2
    config["configurable"]["limits"]["max_graph_steps"] = 1
    result, _ = run(case, config, request())
    assert error(result)["error_code"] == "operations_step_limit"


def test_supervisor_sees_operations_failure_even_with_existing_findings(case, config):
    good, _ = run(case, config, request(), done())
    case["tool_results"] = good["tool_results"]
    failed, _ = run(case, config, RuntimeError("SECRET"))
    case["trace"] = failed["trace"]
    model = Mock()
    command = supervisor.supervise_case(case, config, model=model, classifier_model=model)
    assert command.goto == "escalation"
    assert command.update["outcome"] == "escalated"
    assert not model.mock_calls


def test_supervisor_budget_failure_uses_fallback(case, config):
    case["model_calls"] = 12
    failed, _ = run(case, config)
    case["trace"] = failed["trace"]
    command = supervisor.supervise_case(case, config, model=Mock(), classifier_model=Mock())
    assert command.goto == "persist"
    assert command.update["handoff"]["handoff_source"] == "deterministic_fallback"


def test_terminal_case_rejected_before_model(case, config):
    case["account_context"]["is_blocked"] = True
    model = ScriptedOperationsModel()
    with pytest.raises(ValueError, match="does not permit"):
        operations.handle_operations(case, config, model=model)
    assert not model.messages


@pytest.mark.parametrize("field,value", [("account_id", "a2"), ("customer_id", "c2")])
def test_different_scope_is_not_reused_or_shown_to_model(case, config, field, value):
    first, _ = run(case, config, request(), done())
    case["tool_results"] = first["tool_results"]
    case[field] = value
    result, model = run(case, config, request(), done())
    context = model.messages[0][-1].content.split(
        "Existing findings for this issue in the current turn:\n"
    )[1]
    assert json.loads(context) == []
    assert result["tool_results"]
    assert result["tool_results"][0]["lookup_key"] != first["tool_results"][0]["lookup_key"]
    if field == "customer_id":
        assert result["tool_results"][0]["outcome"] == "not_found"


def test_email_keys_preserve_exact_equality(case, config):
    first, _ = run(case, config, request("lookup_customer_tool", {"email":"one@example.com"}), done())
    case["tool_results"] = first["tool_results"]
    second, _ = run(case, config, request("lookup_customer_tool", {"email":"ONE@example.com"}), done())
    assert second["tool_results"][0]["outcome"] == "not_found"
    assert first["tool_results"][0]["lookup_key"] != second["tool_results"][0]["lookup_key"]


def test_malformed_tool_result_is_sanitized_and_not_retried(case, config, monkeypatch):
    from langchain_core.tools import BaseTool
    original = BaseTool.invoke
    def broken(self, args, config=None, **kwargs):
        if self.name == "lookup_subscription":
            return {"outcome":"found", "records":[{"secret":"SECRET"}]}
        return original(self, args, config=config, **kwargs)
    monkeypatch.setattr(BaseTool, "invoke", broken)
    result, _ = run(case, config, request())
    assert result["tool_results"][0]["error_code"] == "operations_tool_execution_failed"
    assert result["tool_results"][0]["retryable"] is False
    assert len(result["tool_results"]) == 1
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("modes", [None, {}, {"issue1":{"lookup_experiences":"many"}},
                                 {"issue1":{"lookup_subscription":"single"}}])
def test_invalid_assignment_modes_fail_before_model(case, config, modes):
    case["assignment"]["lookup_modes"] = modes
    model = ScriptedOperationsModel()
    with pytest.raises(ValueError):
        operations.handle_operations(case, config, model=model)
    assert model.bound_names == [] and model.messages == []


def test_later_invalid_request_prevents_earlier_query_in_same_batch(case, config, business_engine):
    response = AIMessage(content="", tool_calls=[
        {"id":"one", "name":"lookup_subscription", "args":{}},
        {"id":"two", "name":"lookup_reservations", "args":{"customer_id":"c2", "reservation_id":"other"}},
    ])
    statements = []
    def observe(*args): statements.append(True)
    event.listen(business_engine, "before_cursor_execute", observe)
    try:
        result, _ = run(case, config, response)
    finally:
        event.remove(business_engine, "before_cursor_execute", observe)
    assert error(result)["error_code"] == "operations_customer_scope_mismatch"
    assert result["tool_results"] == [] and statements == []
