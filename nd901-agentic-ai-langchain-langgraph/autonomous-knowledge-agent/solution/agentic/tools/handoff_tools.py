"""Shared deterministic handoff assembly; no database access or model call."""

from typing import Any

from agentic.state import CaseState


def _current_request(state: CaseState) -> str:
    """Extract the text-only human message that owns the current turn.

    Args:
        state: Case state containing turn_id and LangChain messages.

    Returns:
        Content of the last human message whose ID equals turn_id, or an empty
        string when no matching text-only message exists.

    Notes:
        Older turns and assistant messages are ignored. No state is changed,
        and non-text message content is not converted into invented request text.
    """

    turn_id = state.get("turn_id")
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "human" and message.id == turn_id:
            return message.content if isinstance(message.content, str) else ""
    return ""


def assemble_fallback_handoff(state: CaseState) -> dict[str, Any]:
    """Assemble an unsaved deterministic handoff from existing case state.

    Args:
        state: Trusted current case state containing ticket and turn identity,
            issues, messages, classification, tool results, article references
            and the escalation reason.

    Returns:
        A dictionary with a stable handoff message ID, request, issue IDs,
        classification availability, existing account/customer identifiers, current
        Intake account_context, found tool records, source references, attempted
        lookups, gaps and technical tool failures. Missing classified
        issues receive a stable unclassified identifier, not invented facts.

    Notes:
        Tool results are filtered to the current turn. Issues and articles are
        expected to have been reset by Intake. Rejected drafts and review prose
        are excluded. No model call, database write or notification occurs.
        Copied values retain their existing types and must already be serializable;
        this helper does not sanitize datetimes or verify new evidence.
    """

    turn_id = state.get("turn_id")
    ticket_id = state.get("ticket_id")
    issues = [issue for issue in state.get("issues", []) if isinstance(issue, dict)]
    issue_ids = [
        issue["issue_id"]
        for issue in issues
        if isinstance(issue.get("issue_id"), str) and issue["issue_id"].strip()
    ]
    if not issue_ids:
        # why: A failed/skipped classifier can leave no issue object. The synthetic
        # ID labels that unclassified work without inventing its category or facts.
        issue_ids = [f"{turn_id}:unclassified"]

    per_issue_status = [
        {
            key: issue[key]
            for key in (
                "issue_id",
                "task",
                "category",
                "status",
                "coverage",
                "missing_details",
                "article_ids",
                "initial_retrieval_attempted",
                "targeted_retry_attempted",
            )
            if key in issue
        }
        for issue in issues
    ]

    current_tool_results = [
        result
        for result in state.get("tool_results", [])
        if isinstance(result, dict) and result.get("turn_id") == turn_id
    ]
    attempted_lookups = [
        {
            key: result[key]
            for key in (
                "tool_name",
                "lookup_key",
                "issue_id",
                "outcome",
                "reason",
                "error_code",
                "attempt",
            )
            if key in result
        }
        for result in current_tool_results
    ]
    verified_facts = [
        {
            "tool_name": result.get("tool_name"),
            "lookup_key": result.get("lookup_key"),
            "issue_id": result.get("issue_id"),
            "records": result.get("records", []),
        }
        for result in current_tool_results
        if result.get("outcome") == "found"
    ]
    technical_errors = [
        {
            key: result[key]
            for key in ("tool_name", "lookup_key", "issue_id", "error_code", "reason")
            if key in result
        }
        for result in current_tool_results
        if result.get("outcome") == "technical_failure"
    ]

    source_references = []
    for article in state.get("retrieved_articles", []):
        if not isinstance(article, dict):
            continue
        source_id = article.get("article_id", article.get("source_id"))
        if isinstance(source_id, str) and source_id not in source_references:
            source_references.append(source_id)
    for issue in issues:
        article_ids = issue.get("article_ids", [])
        if not isinstance(article_ids, list):
            continue
        for article_id in article_ids:
            if isinstance(article_id, str) and article_id not in source_references:
                source_references.append(article_id)

    gaps = []
    for issue in issues:
        missing_details = issue.get("missing_details", [])
        if isinstance(missing_details, list):
            gaps.extend(
                detail
                for detail in missing_details
                if isinstance(detail, str) and detail.strip()
            )
        gap_reason = issue.get("gap_reason")
        if isinstance(gap_reason, str) and gap_reason.strip():
            gaps.append(gap_reason)

    classification_status = state.get("classification_status", "skipped")
    classification = state.get("classification")
    if classification_status == "completed" and isinstance(classification, dict):
        assessment = {
            "status": "completed",
            "urgency": classification.get("urgency"),
            "urgency_reason": classification.get("urgency_reason"),
            "complexity": classification.get("complexity"),
            "unavailable_reason": None,
        }
    else:
        assessment = {
            "status": classification_status,
            "urgency": None,
            "urgency_reason": None,
            "complexity": None,
            "unavailable_reason": (
                classification.get("reason")
                if isinstance(classification, dict)
                else "Classification was not available."
            ),
        }

    # why: This payload deliberately excludes response_draft and review history;
    # rejected customer wording is not evidence for a human support decision.
    return {
        "message_id": f"{turn_id}:handoff",
        "ticket_id": ticket_id,
        "account_id": state.get("account_id"),
        "customer_id": state.get("customer_id"),
        "account_context": dict(state.get("account_context", {})),
        "turn_id": turn_id,
        "issue_ids": issue_ids,
        "handoff_source": "deterministic_fallback",
        "request": _current_request(state),
        "escalation_reason": state.get("escalation_reason"),
        "issues": per_issue_status,
        "classification": assessment,
        "verified_facts": verified_facts,
        "source_references": source_references,
        "attempted_lookups": attempted_lookups,
        "gaps": gaps,
        "technical_errors": technical_errors,
    }
