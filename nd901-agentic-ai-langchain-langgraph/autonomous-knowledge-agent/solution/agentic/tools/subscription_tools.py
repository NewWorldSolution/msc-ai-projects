"""Read-only subscription lookup; separate from the narrow Intake query."""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
from data.models import cultpass
from utils import get_session


_TOOL_NAME = "lookup_subscription"


@tool
def lookup_subscription(
    config: RunnableConfig, *, customer_id: str | None = None
) -> LookupResult:
    """Read the trusted customer's unique subscription and stored quota.

    Args:
        customer_id: Exact customer ID that must equal trusted_customer_id.
            Omission produces missing_input.
        config: Trusted RunnableConfig supplying cultpass_engine,
            trusted_customer_id, turn_id, issue_id, lookup_key and a positive
            integer attempt.

    Returns:
        A LookupResult with trusted provenance, reason and retryable. A match
        returns found with subscription_id, user_id, status, tier and
        monthly_quota. No row produces not_found. Unusable input produces
        missing_input; invalid context, scope mismatch, uniqueness violations
        and database failures produce technical_failure. Never returns ambiguous.

    Notes:
        Reads Subscription only, closes its session and performs no retry or
        business write. Stored quota is not calculated remaining usage.
        Only recognized transient database failures are retryable. LangChain
        validates other model argument types before entering the function.
    """
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        configurable = {}

    turn_id = configurable.get("turn_id")
    issue_id = configurable.get("issue_id")
    lookup_key = configurable.get("lookup_key")
    attempt = configurable.get("attempt")

    provenance: LookupResult = {"tool_name": _TOOL_NAME}
    if isinstance(turn_id, str) and turn_id.strip():
        provenance["turn_id"] = turn_id
    if isinstance(issue_id, str) and issue_id.strip():
        provenance["issue_id"] = issue_id
    if isinstance(lookup_key, str) and lookup_key.strip():
        provenance["lookup_key"] = lookup_key
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0:
        provenance["attempt"] = attempt

    # api: isinstance(value, str) checks the value's runtime type.
    if not isinstance(customer_id, str) or not customer_id.strip():
        return {
            **provenance,
            "outcome": "missing_input",
            "records": [],
            "reason": "A non-empty customer ID is required.",
            "error_code": "missing_customer_id",
            "retryable": False,
        }

    cultpass_engine = configurable.get("cultpass_engine")
    trusted_customer_id = configurable.get("trusted_customer_id")

    if cultpass_engine is None:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The trusted CultPass engine is unavailable.",
            "error_code": "missing_cultpass_engine",
            "retryable": False,
        }

    if not isinstance(trusted_customer_id, str) or not trusted_customer_id.strip():
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted customer scope is unavailable.",
            "error_code": "missing_trusted_customer_scope",
            "retryable": False,
        }

    if customer_id != trusted_customer_id:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The requested customer is outside the trusted call scope.",
            "error_code": "customer_scope_mismatch",
            "retryable": False,
        }

    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in (turn_id, issue_id, lookup_key)
    ):
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Current turn, issue, and lookup provenance are required.",
            "error_code": "invalid_lookup_provenance",
            "retryable": False,
        }

    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The calling wrapper must supply a positive lookup attempt.",
            "error_code": "invalid_lookup_attempt",
            "retryable": False,
        }

    try:
        # api: get_session opens a session and guarantees rollback/cleanup on errors.
        with get_session(cultpass_engine) as session:
            # api: one_or_none returns the sole match or None and rejects multiple rows.
            subscription = (
                session.query(cultpass.Subscription)
                .filter_by(user_id=customer_id)
                .one_or_none()
            )
            record = (
                None
                if subscription is None
                else {
                    "subscription_id": subscription.subscription_id,
                    "user_id": subscription.user_id,
                    "status": subscription.status,
                    "tier": subscription.tier,
                    "monthly_quota": subscription.monthly_quota,
                }
            )
    except MultipleResultsFound:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Subscription uniqueness could not be verified.",
            "error_code": "subscription_cardinality_violation",
            "retryable": False,
        }
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass subscription query could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }

    if record is None:
        return {
            **provenance,
            "outcome": "not_found",
            "records": [],
            "reason": "No subscription exists for the trusted customer.",
            "retryable": False,
        }

    return {
        **provenance,
        "outcome": "found",
        "records": [record],
        "reason": "Subscription found for the trusted customer.",
        "retryable": False,
    }
