"""Shared application-level model-call reservations for one sequential case run."""

from collections.abc import Callable

from agentic.state import CaseState


class ModelCallBudgetExceeded(RuntimeError):
    """The current turn has no remaining model-call allowance."""


def build_model_call_guard(
    working_case: CaseState,
    *,
    max_model_calls: int,
) -> Callable[[], None]:
    """Return a guard shared by the Supervisor and its specialists.

    Each admitted attempt consumes one reservation, including failed attempts.
    The working state must already contain Intake's non-negative model_calls.
    This guard assumes sequential calls; it does not count SDK HTTP retries.
    """
    if type(max_model_calls) is not int or max_model_calls <= 0:
        raise ValueError("max_model_calls must be a positive integer.")

    def reserve_model_call() -> None:
        """Reserve one attempt or raise without changing an exhausted count."""
        used_calls = working_case.get("model_calls")
        if type(used_calls) is not int or used_calls < 0:
            raise ValueError("model_calls must be a non-negative integer.")
        if used_calls >= max_model_calls:
            raise ModelCallBudgetExceeded(
                f"Exceeded the maximum of {max_model_calls} model calls for this turn."
            )
        working_case["model_calls"] = used_calls + 1

    return reserve_model_call
