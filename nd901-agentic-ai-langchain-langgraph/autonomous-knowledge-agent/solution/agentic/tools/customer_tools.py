"""Shared customer-only query and decorated Operations entry point."""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from data.models import cultpass
from utils import get_session, model_to_dict
from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error


def lookup_customer(
    engine: Engine, *, customer_id: str | None = None, email: str | None = None
) -> LookupResult:
    """Look up one CultPass customer using exactly one usable unique key.

    Args:
        engine: Trusted CultPass engine selected for the ticket's account.
        customer_id: Exact User.user_id, mutually exclusive with email.
        email: Exact stored User.email; case and whitespace are not normalized.

    Returns:
        A LookupResult with outcome, records, reason, error_code and retryable.
        No usable key produces missing_input; two usable keys produce a
        non-retryable technical_failure. A valid lookup returns found with one
        customer or not_found with no records. Unique keys are never ambiguous.
        Records contain only User columns, including source user_id. Datetime
        columns become ISO strings; nullable timestamps remain None.

    Notes:
        This helper reads only business data, closes its session, classifies
        SQLAlchemy failures and performs no retries. The engine provides account
        scope; the caller must enforce ticket-customer identity and attach turn,
        issue, lookup-key and attempt provenance. No relationship is loaded and
        no identity mapping is changed. Non-SQLAlchemy exceptions propagate.
    """
    # api: isinstance(value, str) checks the value's runtime type.
    usable_customer_id = isinstance(customer_id, str) and bool(customer_id.strip())
    usable_email = isinstance(email, str) and bool(email.strip())

    if not usable_customer_id and not usable_email:
        return {
            "outcome": "missing_input",
            "records": [],
            "reason": "Provide exactly one usable customer ID or email.",
            "error_code": "missing_lookup_key",
            "retryable": False,
        }

    if usable_customer_id and usable_email:
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "Provide either customer ID or email, not both.",
            "error_code": "conflicting_lookup_keys",
            "retryable": False,
        }

    try:
        # api: get_session opens a session and guarantees rollback/cleanup on errors.
        with get_session(engine) as session:
            query = session.query(cultpass.User)

            # api: filter_by(...).first() uses bound exact equality and returns one
            # mapped row or None.
            if usable_customer_id:
                customer = query.filter_by(user_id=customer_id).first()
            else:
                customer = query.filter_by(email=email).first()

            if customer is None:
                return {
                    "outcome": "not_found",
                    "records": [],
                    "reason": "No CultPass customer matched the supplied lookup key.",
                    "error_code": None,
                    "retryable": False,
                }

            # api: model_to_dict returns only the mapped table columns; SQLAlchemy
            # relationships such as subscription and reservations are excluded.
            customer_record = model_to_dict(customer)
            for field in ("created_at", "updated_at"):
                value = customer_record[field]
                # api: isoformat() converts a stored datetime into JSON-safe text.
                customer_record[field] = value.isoformat() if value is not None else None
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass customer lookup could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }

    return {
        "outcome": "found",
        "records": [customer_record],
        "reason": "CultPass customer found.",
        "error_code": None,
        "retryable": False,
    }


# api: @tool exposes declared arguments; RunnableConfig is supplied by the caller.
@tool
def lookup_customer_tool(
    config: RunnableConfig, *, customer_id: str | None = None, email: str | None = None
) -> LookupResult:
    """Read the ticket customer's facts by one exact customer ID or email.

    Args:
        config: Trusted RunnableConfig whose configurable mapping supplies
            cultpass_engine, trusted_customer_id, turn_id, issue_id, lookup_key
            and a positive integer attempt. LangChain injects this argument.
        customer_id: Exact customer ID; it must match trusted_customer_id.
        email: Exact stored email, used instead of customer_id. Any matching
            record must still belong to trusted_customer_id.

    Returns:
        The shared customer LookupResult plus tool_name and validated trusted
        provenance. Missing context, identity mismatches and unexpected helper
        failures return sanitized, non-retryable technical_failure results.
        Missing context may leave corresponding provenance fields unavailable.
        Found records contain JSON-safe customer values.

    Notes:
        The model supplies lookup arguments, never engines or trusted scope.
        This wrapper performs no retries, business writes or identity relinking.
        LangChain validates model argument types before entering the function.
    """
    configurable = config.get("configurable", {})
    trusted_context = configurable if isinstance(configurable, dict) else {}

    engine = trusted_context.get("cultpass_engine")
    trusted_customer_id = trusted_context.get("trusted_customer_id")
    turn_id = trusted_context.get("turn_id")
    issue_id = trusted_context.get("issue_id")
    lookup_key = trusted_context.get("lookup_key")
    attempt = trusted_context.get("attempt")

    provenance: LookupResult = {"tool_name": "lookup_customer_tool"}
    if isinstance(turn_id, str) and turn_id.strip():
        provenance["turn_id"] = turn_id
    if isinstance(issue_id, str) and issue_id.strip():
        provenance["issue_id"] = issue_id
    if isinstance(lookup_key, str) and lookup_key.strip():
        provenance["lookup_key"] = lookup_key
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0:
        provenance["attempt"] = attempt

    has_trusted_context = (
        engine is not None
        and isinstance(trusted_customer_id, str)
        and bool(trusted_customer_id.strip())
        and "turn_id" in provenance
        and "issue_id" in provenance
        and "lookup_key" in provenance
        and "attempt" in provenance
    )
    if not has_trusted_context:
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted customer lookup context is incomplete.",
            "error_code": "invalid_tool_context",
            "retryable": False,
            **provenance,
        }

    usable_customer_id = isinstance(customer_id, str) and bool(customer_id.strip())
    usable_email = isinstance(email, str) and bool(email.strip())
    if usable_customer_id and not usable_email and customer_id != trusted_customer_id:
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The requested customer does not match the trusted ticket identity.",
            "error_code": "customer_scope_mismatch",
            "retryable": False,
            **provenance,
        }

    try:
        result = lookup_customer(engine, customer_id=customer_id, email=email)
    except Exception:
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The customer lookup could not be completed.",
            "error_code": "unexpected_customer_lookup_error",
            "retryable": False,
            **provenance,
        }

    if not isinstance(result, dict) or not isinstance(result.get("records"), list):
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The customer lookup returned an invalid result.",
            "error_code": "invalid_customer_lookup_result",
            "retryable": False,
            **provenance,
        }

    records = result["records"]
    valid_cardinality = result.get("outcome") != "found" or len(records) == 1
    valid_records = all(isinstance(record, dict) for record in records)
    if not valid_cardinality or not valid_records:
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The customer lookup returned an invalid result.",
            "error_code": "invalid_customer_lookup_result",
            "retryable": False,
            **provenance,
        }

    if any(record.get("user_id") != trusted_customer_id for record in records):
        return {
            "outcome": "technical_failure",
            "records": [],
            "reason": "The matched customer does not agree with the trusted ticket identity.",
            "error_code": "customer_scope_mismatch",
            "retryable": False,
            **provenance,
        }

    # api: {**first, **second} copies mappings; later keys override earlier ones.
    return {**result, **provenance}
