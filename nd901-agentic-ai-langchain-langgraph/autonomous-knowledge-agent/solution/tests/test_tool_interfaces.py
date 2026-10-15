"""Model-facing schemas and SQL checks for the Operations tool set."""

import pytest
from sqlalchemy import event

from agentic.tools.customer_tools import lookup_customer_tool
from agentic.tools.experience_tools import lookup_experiences
from agentic.tools.reservation_tools import lookup_reservations
from agentic.tools.subscription_tools import lookup_subscription
from test_tool_contracts import business_engine, config


TOOLS = [lookup_customer_tool, lookup_subscription, lookup_reservations, lookup_experiences]


@pytest.mark.parametrize("tool", TOOLS)
def test_model_schema_hides_trusted_runtime_context(tool):
    """The model can supply lookup arguments, but cannot replace trusted scope."""
    properties = tool.tool_call_schema.model_json_schema()["properties"]
    assert not ({"config", "cultpass_engine", "trusted_customer_id", "turn_id",
                 "issue_id", "lookup_key", "attempt", "lookup_mode"} & properties.keys())


def test_decorated_tools_have_stable_distinct_names():
    """Operations can dispatch each declared tool without a name collision."""
    assert {tool.name for tool in TOOLS} == {
        "lookup_customer_tool", "lookup_subscription", "lookup_reservations", "lookup_experiences",
    }


@pytest.mark.parametrize("tool,args", [
    (lookup_customer_tool, {"customer_id": "c1"}),
    (lookup_subscription, {"customer_id": "c1"}),
    (lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (lookup_experiences, {"experience_id": "e1"}),
])
def test_operations_tools_issue_only_reads(business_engine, tool, args):
    """Observe executed statements so a successful lookup cannot hide business writes."""
    statements = []

    def record_statement(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(business_engine, "before_cursor_execute", record_statement)
    try:
        result = tool.invoke(args, config=config(business_engine))
    finally:
        event.remove(business_engine, "before_cursor_execute", record_statement)
    assert result["outcome"] == "found"
    assert statements
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
