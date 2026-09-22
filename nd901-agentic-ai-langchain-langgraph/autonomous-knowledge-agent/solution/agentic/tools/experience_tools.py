"""Read-only experience lookup, including non-customer-specific facts."""

from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
from data.models import cultpass
from utils import get_session

_TOOL_NAME = "lookup_experiences"

def _experience_record(
    experience: cultpass.Experience,
) -> dict[str, Any]:
    """Copy one loaded Experience row into a plain record.

    Args:
        experience: A mapped CultPass Experience with its column values loaded.

    Returns:
        Stored experience_id, title, description, location, event time,
        availability, premium flag and timestamps. Datetimes become ISO strings;
        nullable audit timestamps remain None.

    Notes:
        Read columns while the session is open. This helper performs no lookup,
        relationship loading, inference or mutation. The required when column
        must contain a datetime.
    """

    # api: datetime.isoformat() converts a datetime into a stable text value.
    return {
        "experience_id": experience.experience_id,
        "title": experience.title,
        "description": experience.description,
        "location": experience.location,
        "when": experience.when.isoformat(),
        "slots_available": experience.slots_available,
        "is_premium": experience.is_premium,
        "created_at": (
            experience.created_at.isoformat()
            if experience.created_at is not None
            else None
        ),
        "updated_at": (
            experience.updated_at.isoformat()
            if experience.updated_at is not None
            else None
        ),
    }

@tool
def lookup_experiences(
    config: RunnableConfig,
    *,
    experience_id: str | None = None,
    title: str | None = None,
    event_date: str | None = None,
    location: str | None = None,
) -> LookupResult:
    """Read CultPass experiences by exact ID or descriptive equality filters.

    Args:
        config: Trusted RunnableConfig supplying cultpass_engine, lookup_mode
            (single or list), turn_id, issue_id, lookup_key and a positive
            integer attempt. The engine provides account scope; customer
            identity is not required.
        experience_id: Exact primary key, exclusive with descriptive filters.
        title: Exact stored title for descriptive search.
        event_date: ISO calendar date used to filter the stored when column.
        location: Exact stored location for descriptive search.

    Returns:
        A LookupResult with records, reason, retryable and trusted provenance.
        Missing or invalid search details produce missing_input; conflicting
        modes or invalid context produce technical_failure. Exact IDs return
        found with one record or not_found. Multiple descriptive matches produce
        ambiguous in single mode and found with all matches in list mode.
        Returned records contain stored source IDs and ISO datetime strings.

    Notes:
        Reads only Experience rows, closes its session and performs no retry or
        availability change. Only recognized transient database failures are
        retryable. Dates must use YYYY-MM-DD and precede 9999-12-31; invalid
        dates are rejected before database access.
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

    if (
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt > 0
    ):
        provenance["attempt"] = attempt

    usable_experience_id = (
        isinstance(experience_id, str)
        and bool(experience_id.strip())
    )
    usable_title = isinstance(title, str) and bool(title.strip())
    usable_location = (
        isinstance(location, str)
        and bool(location.strip())
    )

    parsed_event_date: date | None = None

    if event_date is not None:
        if not isinstance(event_date, str) or not event_date.strip():
            return {
                **provenance,
                "outcome": "missing_input",
                "records": [],
                "reason": "Event date must use YYYY-MM-DD.",
                "error_code": "invalid_event_date",
                "retryable": False,
            }

        try:
            # api: date.fromisoformat validates and parses a YYYY-MM-DD value.
            parsed_event_date = date.fromisoformat(event_date)
            if parsed_event_date.isoformat() != event_date or parsed_event_date == date.max:
                raise ValueError("Event date must be a supported YYYY-MM-DD date.")
        except ValueError:
            return {
                **provenance,
                "outcome": "missing_input",
                "records": [],
                "reason": "Event date must use YYYY-MM-DD.",
                "error_code": "invalid_event_date",
                "retryable": False,
            }

    descriptive_search = (
        usable_title
        or parsed_event_date is not None
        or usable_location
    )

    if not usable_experience_id and not descriptive_search:
        return {
            **provenance,
            "outcome": "missing_input",
            "records": [],
            "reason": (
                "Provide an experience ID or at least one of title, "
                "event date, and location."
            ),
            "error_code": "missing_experience_lookup",
            "retryable": False,
        }

    # why: ID lookup and descriptive search are separate modes. Combining them
    # could silently hide conflicting caller inputs.
    if usable_experience_id and descriptive_search:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": (
                "Use either experience ID or descriptive search fields, "
                "not both."
            ),
            "error_code": "conflicting_experience_lookup_modes",
            "retryable": False,
        }

    lookup_mode = configurable.get("lookup_mode")
    if lookup_mode not in ("single", "list"):
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted lookup mode must be single or list.",
            "error_code": "invalid_lookup_mode",
            "retryable": False,
        }

    cultpass_engine = configurable.get("cultpass_engine")

    if cultpass_engine is None:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The trusted CultPass engine is unavailable.",
            "error_code": "missing_cultpass_engine",
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
            "reason": (
                "Current turn, issue, and lookup provenance are required."
            ),
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
            "reason": (
                "The calling wrapper must supply a positive lookup attempt."
            ),
            "error_code": "invalid_lookup_attempt",
            "retryable": False,
        }

    try:
        # api: get_session guarantees rollback and cleanup if querying fails.
        with get_session(cultpass_engine) as session:
            if usable_experience_id:
                # api: Session.get loads one row by its primary key or returns None.
                experience = session.get(
                    cultpass.Experience,
                    experience_id,
                )
                experiences = (
                    [] if experience is None else [experience]
                )
            else:
                query = session.query(cultpass.Experience)

                if usable_title:
                    # api: SQLAlchemy comparison expressions use bound parameters.
                    query = query.filter(
                        cultpass.Experience.title == title
                    )

                if usable_location:
                    query = query.filter(
                        cultpass.Experience.location == location
                    )

                if parsed_event_date is not None:
                    # why: `when` is a datetime, so one calendar date is represented
                    # by a half-open range: start inclusive, next day exclusive.
                    day_start = datetime.combine(
                        parsed_event_date,
                        datetime.min.time(),
                    )
                    day_end = day_start + timedelta(days=1)

                    query = query.filter(
                        cultpass.Experience.when >= day_start,
                        cultpass.Experience.when < day_end,
                    )

                # api: order_by(...).all() returns every match in stable order.
                experiences = (
                    query.order_by(
                        cultpass.Experience.when,
                        cultpass.Experience.experience_id,
                    )
                    .all()
                )

            # why: Convert rows while the session is open; no lazy relationship
            # loading is needed after the database session closes.
            records = [
                _experience_record(experience)
                for experience in experiences
            ]

    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass experience query could not be completed.",
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
            "reason": "No CultPass experiences matched the supplied search.",
            "retryable": False,
        }

    if not usable_experience_id and lookup_mode == "single" and len(records) > 1:
        return {
            **provenance,
            "outcome": "ambiguous",
            "records": records,
            "reason": "Multiple experiences matched; more identifying details are required.",
            "retryable": False,
        }

    return {
        **provenance,
        "outcome": "found",
        "records": records,
        "reason": "CultPass experiences found.",
        "retryable": False,
    }
