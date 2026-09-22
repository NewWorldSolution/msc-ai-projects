"""Classify transient failures for the project's SQLite database engines."""

import sqlite3

from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError


def is_retryable_database_error(error: SQLAlchemyError) -> bool:
    """Recognize SQLite lock/busy failures and connection-pool timeouts.

    Args:
        error: SQLAlchemy exception caught at a database boundary.

    Returns:
        True only for recognized transient conditions. Permanent or unknown
        errors, including errors from unrecognized backends, return False.

    Notes:
        Performs no retries, logging or database access. Exact-message
        fallback supports test doubles without a SQLite error code.
    """
    if isinstance(error, SQLAlchemyTimeoutError):
        return True
    if not isinstance(error, OperationalError):
        return False

    original = error.orig
    if not isinstance(original, sqlite3.OperationalError):
        return False

    # api: sqlite_errorcode exposes SQLite's numeric result code on real errors.
    code = getattr(original, "sqlite_errorcode", None)
    if isinstance(code, int):
        # api: & 0xFF extracts the primary code from an extended SQLite code.
        return (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)

    # why: Exact known messages support test doubles without guessing from keywords.
    return str(original).lower() in {
        "database is locked",
        "database is busy",
        "database table is locked",
        "database schema is locked",
    }