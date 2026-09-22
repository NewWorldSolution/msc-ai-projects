"""Prepare human handoffs and always return them to Supervisor."""

from langchain_core.language_models.chat_models import BaseChatModel

from agentic.state import CaseState


def prepare_escalation(state: CaseState, *, model: BaseChatModel) -> CaseState:
    """Preserve the unresolved case without writing a customer reply; section 4.6.

    TODO: Assemble request, escalation reason, issue status, verified findings and
    source references, attempted work, gaps, failures, IDs, urgency and complexity.
    TODO: Use a local ChatPromptTemplate and model.with_structured_output for
    the handoff schema, following architecture 5.9's explicit invoke sequence.
    TODO: Generate a structured handoff using existing state only; validate it and
    set handoff_source=generated. A suggested investigation is optional.
    TODO: On generation/validation failure, immediately use the shared deterministic
    helper in tools/handoff_tools.py; no extra generation/repair call. Skip generation
    if the model is already unavailable. Unknown facts remain explicitly unknown.
    TODO: Return the handoff/source and failure details to Supervisor in every case.
    No direct route to Response or Persist; Supervisor controls acknowledgement
    permission and final persistence. Rejected drafts are not factual evidence.
    """
    raise NotImplementedError("L1 scaffold: implement the agreed handoff contract.")
