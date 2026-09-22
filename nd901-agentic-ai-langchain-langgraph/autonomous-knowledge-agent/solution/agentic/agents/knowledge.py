"""Retrieve and assess article evidence for Supervisor-assigned issues."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from agentic.state import CaseState


def retrieve_knowledge(
    state: CaseState, config: RunnableConfig, *, model: BaseChatModel
) -> CaseState:
    """Return article evidence and per-issue coverage; architecture 4.3/5.4.

    TODO: Follow the lesson's retrieval/prompt separation: search_knowledge
    supplies scored source documents; a local ChatPromptTemplate and structured
    model output assess the evidence. Do not append internal prompts to messages.
    TODO: Select focused queries for assigned issues and use search_knowledge.
    Validate source IDs/excerpts in code; use the model for applicability,
    sufficiency and consistency within retrieved evidence, with reasons/excerpts.
    TODO: Return supported, targeted_retry_needed, awaiting_business_facts or
    evidence_problem for each assigned issue. Reliability cannot certify freshness.
    Store that value in each issue's coverage field, with coverage_reason,
    article_ids and retrieval-attempt flags. Preserve unassigned issue assessments.
    Honor assignment.retrieval_mode=targeted without resetting retry allowances.
    TODO: Return the COMPLETE current-turn retrieved_articles list, preserving
    other issues' accepted evidence and checking conflicts with added articles.
    TODO: Finish initial search coverage before concluding no relevant evidence
    exists at all. Allow one targeted search per unanswered issue/turn, independent
    of technical retries. Reassessment with new facts does not consume a search.
    TODO: Log score, threshold and gate_pass together per issue/article candidate;
    empty retrieval has score=None, gate_pass=False and reason=no_candidates.
    TODO: Return gaps/conflicts/failures to Supervisor; no business lookup, wording,
    routing or ticket outcome is decided here.
    """
    raise NotImplementedError("L1 scaffold: implement the agreed evidence contract.")
