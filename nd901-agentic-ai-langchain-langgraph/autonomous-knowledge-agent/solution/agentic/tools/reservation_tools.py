"""Read-only reservation lookup by ID, event details or customer list."""

from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
from data.models import cultpass
from utils import get_session


_TOOL_NAME = "lookup_reservations"


def _to_iso(value: datetime | None) -> str | None:
    """Convert a nullable stored datetime into an ISO string.

    Args:
        value: A datetime column value or None.

    Returns:
        value.isoformat() for a datetime, otherwise None for a missing value.

    Notes:
        No time zone is inferred or converted, and no state is changed.
    """

    # `datetime.isoformat()` converts a datetime to stable text.
    return value.isoformat() if value is not None else None


def _reservation_record(
    reservation: cultpass.Reservation,
    experience: cultpass.Experience,
) -> dict[str, Any]:
    """Copy a reservation and its joined experience into a plain record.

    Args:
        reservation: Loaded CultPass Reservation belonging to the trusted customer.
        experience: Loaded Experience referenced by reservation.experience_id.

    Returns:
        Stored reservation_id, user_id, experience_id, status and audit times,
        plus nested stored experience details. Datetimes become ISO strings;
        nullable values remain None.

    Notes:
        The caller owns customer scope and join correctness. This helper copies
        loaded columns without additional queries, writes or inferred values.
    """

    return {
        "reservation_id": reservation.reservation_id,
        "user_id": reservation.user_id,
        "experience_id": reservation.experience_id,
        "status": reservation.status,
        "created_at": _to_iso(reservation.created_at),
        "updated_at": _to_iso(reservation.updated_at),
        "experience": {
            "experience_id": experience.experience_id,
            "title": experience.title,
            "description": experience.description,
            "location": experience.location,
            "when": _to_iso(experience.when),
            "slots_available": experience.slots_available,
            "is_premium": experience.is_premium,
            "created_at": _to_iso(experience.created_at),
            "updated_at": _to_iso(experience.updated_at),
        },
    }


@tool
def lookup_reservations(
    config: RunnableConfig,
    *,
    customer_id: str | None = None,
    reservation_id: str | None = None,
    event_name: str | None = None,
    event_date: str | None = None,
) -> LookupResult:
    """Read the trusted customer's reservations by ID, event details or list.

    Args:
        customer_id: Exact ID that must equal trusted_customer_id. Omission
            produces missing_input.
        config: Trusted RunnableConfig supplying cultpass_engine,
            trusted_customer_id, lookup_mode (single or list), turn_id,
            issue_id, lookup_key and a positive integer attempt.
        reservation_id: Exact reservation ID, exclusive with event filters.
        event_name: Exact associated Experience.title for descriptive search.
        event_date: ISO calendar date filtering the associated Experience.when.

    Returns:
        A LookupResult with stored reservation/experience records, reason,
        retryable and trusted provenance. Exact-ID matches are never ambiguous.
        Multiple descriptive matches are ambiguous in single mode and found in
        list mode. Customer-only input is allowed for list mode. No matches
        produce not_found; unusable details produce missing_input; scope,
        context and database failures produce technical_failure.

    Notes:
        Every query retains customer scope, including exact-ID lookup. No booking,
        cancellation or availability write occurs. Sessions are closed and this
        tool performs no retries. Only recognized transient database failures are
        retryable. Dates must use YYYY-MM-DD and precede 9999-12-31; invalid
        dates are rejected before database access. LangChain validates other
        model argument types before entering the function.
    """

    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        configurable = {}

    cultpass_engine = configurable.get("cultpass_engine")
    trusted_customer_id = configurable.get("trusted_customer_id")
    lookup_mode = configurable.get("lookup_mode")

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

    if (
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt > 0
    ):
        provenance["attempt"] = attempt

    # Validate the customer supplied to the tool.
    if not isinstance(customer_id, str) or not customer_id.strip():
        return {
            **provenance,
            "outcome": "missing_input",
            "records": [],
            "reason": "A non-empty customer ID is required.",
            "error_code": "missing_customer_id",
            "retryable": False,
        }

    if cultpass_engine is None:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The trusted CultPass engine is unavailable.",
            "error_code": "missing_cultpass_engine",
            "retryable": False,
        }

    if (
        not isinstance(trusted_customer_id, str)
        or not trusted_customer_id.strip()
    ):
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted customer scope is unavailable.",
            "error_code": "missing_trusted_customer_scope",
            "retryable": False,
        }

    # A model-supplied customer ID cannot switch the trusted ticket identity.
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

    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The calling wrapper must supply a positive attempt.",
            "error_code": "invalid_lookup_attempt",
            "retryable": False,
        }

    if lookup_mode not in {"single", "list"}:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted lookup mode must be single or list.",
            "error_code": "invalid_lookup_mode",
            "retryable": False,
        }

    usable_reservation_id = (
        isinstance(reservation_id, str)
        and bool(reservation_id.strip())
    )
    usable_event_name = (
        isinstance(event_name, str)
        and bool(event_name.strip())
    )

    parsed_event_date: date | None = None

    if event_date is not None:
        if not isinstance(event_date, str) or not event_date.strip():
            return {
                **provenance,
                "outcome": "missing_input",
                "records": [],
                "reason": "Event date must use YYYY-MM-DD and be earlier than 9999-12-31.",
                "error_code": "invalid_event_date",
                "retryable": False,
            }

        try:
            # `date.fromisoformat()` validates and parses YYYY-MM-DD.
            parsed_event_date = date.fromisoformat(event_date)
            if parsed_event_date.isoformat() != event_date or parsed_event_date == date.max:
                raise ValueError("Event date must be a supported YYYY-MM-DD date.")
        except ValueError:
            return {
                **provenance,
                "outcome": "missing_input",
                "records": [],
                "reason": "Event date must use YYYY-MM-DD and be earlier than 9999-12-31.",
                "error_code": "invalid_event_date",
                "retryable": False,
            }

    descriptive_search = (
        usable_event_name or parsed_event_date is not None
    )

    # Exact-ID and descriptive searches are separate lookup modes.
    if usable_reservation_id and descriptive_search:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": (
                "Use either reservation ID or event details, not both."
            ),
            "error_code": "conflicting_reservation_lookup_modes",
            "retryable": False,
        }

    # Customer ID alone is valid only when the assigned task requests a list.
    if (
        not usable_reservation_id
        and not descriptive_search
        and lookup_mode != "list"
    ):
        return {
            **provenance,
            "outcome": "missing_input",
            "records": [],
            "reason": (
                "Provide a reservation ID or event details when one "
                "reservation is required."
            ),
            "error_code": "missing_reservation_lookup",
            "retryable": False,
        }

    try:
        # `get_session` guarantees rollback and cleanup when querying fails.
        with get_session(cultpass_engine) as session:
            # `join()` combines Reservation with its referenced Experience row.
            query = (
                session.query(
                    cultpass.Reservation,
                    cultpass.Experience,
                )
                .join(
                    cultpass.Experience,
                    cultpass.Reservation.experience_id
                    == cultpass.Experience.experience_id,
                )
                .filter(
                    cultpass.Reservation.user_id == customer_id
                )
            )

            if usable_reservation_id:
                # Customer scope remains enforced even for a unique ID.
                query = query.filter(
                    cultpass.Reservation.reservation_id
                    == reservation_id
                )
            else:
                if usable_event_name:
                    query = query.filter(
                        cultpass.Experience.title == event_name
                    )

                if parsed_event_date is not None:
                    # Experience.when is a datetime, so a calendar date
                    # is represented by a half-open daily range.
                    day_start = datetime.combine(
                        parsed_event_date,
                        datetime.min.time(),
                    )
                    day_end = day_start + timedelta(days=1)

                    query = query.filter(
                        cultpass.Experience.when >= day_start,
                        cultpass.Experience.when < day_end,
                    )

            # `order_by(...).all()` returns every match in stable order.
            matches = (
                query.order_by(
                    cultpass.Experience.when,
                    cultpass.Reservation.reservation_id,
                )
                .all()
            )

            # Convert rows while the database session remains open.
            records = [
                _reservation_record(reservation, experience)
                for reservation, experience in matches
            ]

    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass reservation query could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }
    if not records:
        return {
            **provenance,
            "outcome": "not_found",
            "records": [],
            "reason": (
                "No reservation belonging to the trusted customer "
                "matched the supplied search."
            ),
            "retryable": False,
        }

    # Exact reservation IDs are unique and never ambiguous.
    if usable_reservation_id:
        return {
            **provenance,
            "outcome": "found",
            "records": records,
            "reason": "Reservation found for the trusted customer.",
            "retryable": False,
        }

    # One-result tasks must not silently choose between multiple reservations.
    if lookup_mode == "single" and len(records) > 1:
        return {
            **provenance,
            "outcome": "ambiguous",
            "records": records,
            "reason": (
                "Multiple reservations matched; additional event details "
                "are required."
            ),
            "retryable": False,
        }

    return {
        **provenance,
        "outcome": "found",
        "records": records,
        "reason": "Matching reservations found for the trusted customer.",
        "retryable": False,
    }
