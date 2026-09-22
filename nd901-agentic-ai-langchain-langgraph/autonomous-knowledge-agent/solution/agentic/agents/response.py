"""Prepare complete channel-aware wording for Supervisor's assigned outcome."""

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.state import CaseState


def draft_response(state: CaseState, *, model: BaseChatModel) -> CaseState:
    """Return a reviewable draft or precise blocker; architecture 4.5.

    TODO: Use a local ChatPromptTemplate and model.with_structured_output with
    the draft/blocker schema, following the L4 routing lesson's explicit invoke
    sequence. Return structured fields directly, without a second formatting call.
    TODO: Draft chat body or email subject/greeting/body/closing for the assigned
    resolved, information_requested or escalated outcome, covering all issues.
    TODO: Use accepted articles and verified facts; do not claim a business write,
    unsupported resolution, human response time or completed delivery.
    TODO: Return response_draft with draft_id, outcome, channel/body/subject,
    addressed issue IDs, claim-to-evidence mapping and clarification-question basis.
    TODO: If drafting is blocked, return draft_blocker with the exact missing
    evidence instead. A blocker is not a draft rejection or completed review.
    TODO: Include previous review reasons in a permitted redraft. Clear stale
    approval on a new draft without resetting Supervisor-owned stage counters.
    TODO: Return to Supervisor; never deliver or append a draft to messages.
    """
    raise NotImplementedError("L1 scaffold: implement the agreed Response contract.")
