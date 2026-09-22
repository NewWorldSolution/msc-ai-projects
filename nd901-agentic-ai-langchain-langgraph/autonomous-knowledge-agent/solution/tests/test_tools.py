"""Tool-contract scenario placeholders; no database queries are implemented."""

import pytest


pytestmark = pytest.mark.skip(reason="L1 scaffold: implement tool scenario bodies first.")


def test_customer_lookup_requires_one_exact_key_and_returns_only_customer_facts() -> None:
    """TODO: ID/email each return at most one customer; neither or both keys
    produce their controlled failure. No subscription fetch, ambiguity or relink.
    """


def test_subscription_lookup_is_unique_per_customer() -> None:
    """TODO: Found/absent/missing-input/error variants; never ambiguous and never
    infer remaining usage or a completed business write from stored quota.
    """


def test_reservation_and_experience_cardinality_depends_on_lookup_mode() -> None:
    """TODO: Unique IDs cannot be ambiguous; descriptive single-record searches
    may be ambiguous, while list tasks return found lists. Enforce customer scope.
    """


def test_transient_lookup_failures_use_independent_bounded_attempts() -> None:
    """TODO: Tool-layer retryability is deterministic; one retry per scoped key,
    no retry of completed empty results and no failure interpreted as absence.
    """
