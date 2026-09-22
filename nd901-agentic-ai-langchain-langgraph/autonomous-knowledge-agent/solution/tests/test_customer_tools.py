"""Customer-only lookup contract tests."""

import sqlite3
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from agentic.tools import customer_tools
from data.models import cultpass
from utils import get_session


@pytest.fixture
def cultpass_engine():
    """Return one in-memory CultPass database shared across test connections."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    cultpass.Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add(
            cultpass.User(
                user_id="customer-1",
                full_name="Ada Lovelace",
                email="Ada@Example.com",
                is_blocked=False,
            )
        )
    return engine


@pytest.mark.parametrize(
    ("kwargs", "outcome", "error_code"),
    [
        ({}, "missing_input", "missing_lookup_key"),
        ({"customer_id": "   "}, "missing_input", "missing_lookup_key"),
        (
            {"customer_id": "customer-1", "email": "Ada@Example.com"},
            "technical_failure",
            "conflicting_lookup_keys",
        ),
    ],
)
def test_lookup_customer_validates_exactly_one_usable_key(
    cultpass_engine, kwargs, outcome, error_code
) -> None:
    result = customer_tools.lookup_customer(cultpass_engine, **kwargs)

    assert result["outcome"] == outcome
    assert result["records"] == []
    assert result["reason"]
    assert result["error_code"] == error_code
    assert result["retryable"] is False


def test_lookup_customer_returns_only_user_columns(cultpass_engine) -> None:
    result = customer_tools.lookup_customer(
        cultpass_engine, customer_id="customer-1"
    )

    assert result["outcome"] == "found"
    assert result["retryable"] is False
    assert result["records"][0]["user_id"] == "customer-1"
    assert result["records"][0]["full_name"] == "Ada Lovelace"
    assert result["records"][0]["email"] == "Ada@Example.com"
    assert result["records"][0]["is_blocked"] is False
    assert set(result["records"][0]) == {
        "user_id",
        "full_name",
        "email",
        "is_blocked",
        "created_at",
        "updated_at",
    }


def test_lookup_customer_preserves_stored_email_equality(cultpass_engine) -> None:
    exact = customer_tools.lookup_customer(cultpass_engine, email="Ada@Example.com")
    different_case = customer_tools.lookup_customer(
        cultpass_engine, email="ada@example.com"
    )

    assert exact["outcome"] == "found"
    assert different_case["outcome"] == "not_found"


def test_lookup_customer_returns_not_found_for_unknown_id(cultpass_engine) -> None:
    result = customer_tools.lookup_customer(cultpass_engine, customer_id="unknown")

    assert result["outcome"] == "not_found"
    assert result["records"] == []
    assert result["error_code"] is None
    assert result["retryable"] is False


def test_lookup_customer_classifies_transient_operational_errors(
    cultpass_engine, monkeypatch
) -> None:
    @contextmanager
    def failing_session(_engine):
        raise OperationalError(
            "SELECT users",
            {},
            sqlite3.OperationalError("database is locked"),
        )
        yield

    monkeypatch.setattr(customer_tools, "get_session", failing_session)

    result = customer_tools.lookup_customer(
        cultpass_engine, customer_id="customer-1"
    )

    assert result["outcome"] == "technical_failure"
    assert result["records"] == []
    assert result["error_code"] == "transient_database_error"
    assert result["retryable"] is True


def test_lookup_customer_classifies_other_sqlalchemy_errors(
    cultpass_engine, monkeypatch
) -> None:
    @contextmanager
    def failing_session(_engine):
        raise SQLAlchemyError("query failed")
        yield

    monkeypatch.setattr(customer_tools, "get_session", failing_session)

    result = customer_tools.lookup_customer(
        cultpass_engine, customer_id="customer-1"
    )

    assert result["outcome"] == "technical_failure"
    assert result["records"] == []
    assert result["error_code"] == "database_error"
    assert result["retryable"] is False
