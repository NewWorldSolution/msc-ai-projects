"""Current-turn case contracts and graph update reducers; ARCHITECTURE.md 5.1."""

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps

# Literal lists allowed values; Pydantic validates them in model schemas.
IssueCategory = Literal[
    "account_access",
    "subscription",
    "reservation",
    "experience_information",
    "unknown",
    "unsupported",
]
Urgency = Literal["normal", "urgent"]
Complexity = Literal["simple", "complex"]
CaseOutcome = Literal["resolved", "information_requested", "escalated"]
LookupOutcome = Literal[
    "found", "not_found", "ambiguous", "missing_input", "technical_failure"
]


class LookupResult(TypedDict, total=False):
    """Controlled tool result; wrappers attach trusted turn/issue provenance.

    The consuming agents (Response sources, Escalation handoff) validate the outcome and
    the mode-specific cardinality; records contains only scoped facts, and an empty
    completed lookup is not an error.
    """

    outcome: LookupOutcome
    records: list[dict[str, Any]]
    reason: str
    error_code: str
    retryable: bool
    tool_name: str
    lookup_key: str
    turn_id: str
    issue_id: str
    attempt: int


# `total=False` permits partial dictionaries, including a node's state delta.
class CaseState(TypedDict, total=False):
    """One ticket's checkpointed state; nodes return only their changed fields.

    Caller must supply a HumanMessage with a stable ID and configurable.thread_id.
    Intake validates that input; optional keys here support partial node updates.
    Models, database engines and retrieval resources are not checkpointed in state.
    """

    # `add_messages` merges by message ID, so a repeated stable ID is replaced.
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_id: str
    account_id: str
    customer_id: str | None
    channel: str
    turn_id: str
    account_context: dict[str, Any]
    customer_memory: list[dict[str, Any]]
    # Replacement: explicit candidates for Persist; only resolved turns may write them.
    memory_updates: list[dict[str, Any]]
    preparation_result: dict[str, Any]
    classification_status: Literal["pending", "completed", "failed", "skipped"]
    classification: dict[str, Any] | None
    issues: list[dict[str, Any]]
    assignment: dict[str, Any] | None

    # Annotated[..., add] concatenates list deltas; return new entries only.
    tool_results: Annotated[list[LookupResult], add]
    trace: Annotated[list[dict[str, Any]], add]
    # Replacement: Knowledge returns the COMPLETE current-turn merged article list.
    retrieved_articles: list[dict[str, Any]]
    technical_attempts: dict[str, int]
    workflow_steps: int
    model_calls: int

    response_stage: Literal["original", "acknowledgement"] | None
    response_draft: dict[str, Any] | None
    draft_blocker: dict[str, Any] | None
    review_result: dict[str, Any] | None
    # Replacement: Supervisor owns the complete review history for this turn.
    review_history: list[dict[str, Any]]
    original_review_attempts: int
    acknowledgement_review_attempts: int
    approved_response: dict[str, Any] | None
    outcome: CaseOutcome | None
    escalation_reason: str | None
    acknowledgement_permitted: bool
    messaging_disabled: bool
    handoff: dict[str, Any] | None
    # Ticket-scoped: survives a new turn; other per-turn controls reset at Intake.
    identity_correction: dict[str, Any]
    persistence_result: dict[str, Any] | None
    remaining_steps: RemainingSteps
