"""Account-scoped article retrieval with traceable relevance scores."""

from math import isfinite
from typing import Any

from langchain_core.runnables import RunnableConfig
from openai import APIConnectionError, RateLimitError
from sqlalchemy.exc import SQLAlchemyError

from data.models import udahub
from agentic.tools.database_errors import is_retryable_database_error
from utils import get_session


HIGHER_IS_MORE_RELEVANT = "higher_is_more_relevant"


def search_knowledge(query: str, config: RunnableConfig) -> dict[str, Any]:
    """Retrieve account-scoped article candidates and verify their source text.

    Args:
        query: Non-empty search text; surrounding whitespace is removed.
        config: Trusted RunnableConfig supplying knowledge_store, udahub_engine,
            trusted_account_id, turn_id, issue_id and retrieval_settings.
            Settings require a finite threshold in [0, 1], positive integer
            candidate_count, non-empty embedding_model and the score convention
            higher_is_more_relevant.

    Returns:
        A dictionary with status, query, articles, gate_records, reason,
        error_code, retryable and available account/turn/issue provenance.
        Completed searches include all verified candidates, even below-threshold
        candidates, with score, threshold and gate_pass recorded together.
        Empty searches are completed with no articles and a no_candidates gate
        whose score is None. Invalid input produces missing_input; retrieval,
        metadata or source-verification failures produce technical_failure.

    Notes:
        The vector search and authoritative Knowledge query are account-scoped.
        Returned titles, content and excerpts come from verified stored sources.
        This helper reads only, closes its database session, performs no retry
        and makes no semantic sufficiency or case-outcome decision. The caller
        owns retry budgets and persistence of gate events. Recognized transient
        database failures, built-in timeout/connection exceptions and OpenAI
        connection/timeout/rate-limit exceptions are retryable. Unknown exceptions
        are sanitized and non-retryable.
    """
    configurable = config.get("configurable", {})
    trusted_context = configurable if isinstance(configurable, dict) else {}

    knowledge_store = trusted_context.get("knowledge_store")
    udahub_engine = trusted_context.get("udahub_engine")
    retrieval_settings = trusted_context.get("retrieval_settings")
    account_id = trusted_context.get("trusted_account_id")
    turn_id = trusted_context.get("turn_id")
    issue_id = trusted_context.get("issue_id")

    provenance: dict[str, Any] = {}
    if isinstance(account_id, str) and account_id.strip():
        provenance["account_id"] = account_id
    if isinstance(turn_id, str) and turn_id.strip():
        provenance["turn_id"] = turn_id
    if isinstance(issue_id, str) and issue_id.strip():
        provenance["issue_id"] = issue_id

    def technical_failure(
        error_code: str,
        reason: str,
        *,
        retryable: bool = False,
    ) -> dict[str, Any]:
        """Build an empty, sanitized knowledge-search failure result.

        Args:
            error_code: Stable machine-readable classification.
            reason: Controlled explanation, never a raw exception or connection URL.
            retryable: Whether the caller may consider a technical retry.

        Returns:
            A technical_failure dictionary with the enclosing query and available
            trusted provenance, empty articles and empty gate_records.

        Notes:
            This nested helper does not log, raise, access storage or retry.
        """
        return {
            "status": "technical_failure",
            "query": query,
            "articles": [],
            "gate_records": [],
            "reason": reason,
            "error_code": error_code,
            "retryable": retryable,
            **provenance,
        }

    if not isinstance(query, str) or not query.strip():
        return {
            "status": "missing_input",
            "query": query,
            "articles": [],
            "gate_records": [],
            "reason": "missing_query",
            "error_code": "missing_query",
            "retryable": False,
            **provenance,
        }

    has_trusted_context = (
        knowledge_store is not None
        and udahub_engine is not None
        and isinstance(retrieval_settings, dict)
        and "account_id" in provenance
        and "turn_id" in provenance
        and "issue_id" in provenance
    )
    if not has_trusted_context:
        return technical_failure(
            "invalid_tool_context",
            "Trusted knowledge-search context is incomplete.",
        )

    threshold = retrieval_settings.get("threshold")
    candidate_count = retrieval_settings.get("candidate_count")
    embedding_model = retrieval_settings.get("embedding_model")
    score_convention = retrieval_settings.get("score_convention")

    valid_threshold = (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and isfinite(float(threshold))
        and 0.0 <= float(threshold) <= 1.0
    )
    valid_candidate_count = (
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count > 0
    )
    valid_embedding_model = (
        isinstance(embedding_model, str) and bool(embedding_model.strip())
    )

    if (
        not valid_threshold
        or not valid_candidate_count
        or not valid_embedding_model
        or score_convention != HIGHER_IS_MORE_RELEVANT
    ):
        return technical_failure(
            "invalid_retrieval_settings",
            "Prepared retrieval settings are invalid or unsupported.",
        )

    threshold = float(threshold)
    normalized_query = query.strip()

    try:
        # api: similarity_search_with_relevance_scores returns
        # (Document, score) pairs, where Chroma relevance is higher-is-better.
        candidates = knowledge_store.similarity_search_with_relevance_scores(
            normalized_query,
            k=candidate_count,
            filter={"account_id": account_id},
        )

        if not candidates:
            no_candidate_gate = {
                "account_id": account_id,
                "turn_id": turn_id,
                "issue_id": issue_id,
                "article_id": None,
                "score": None,
                "threshold": threshold,
                "gate_pass": False,
                "reason": "no_candidates",
                "embedding_model": embedding_model,
                "score_convention": score_convention,
            }
            return {
                "status": "completed",
                "query": normalized_query,
                "articles": [],
                "gate_records": [no_candidate_gate],
                "reason": "no_candidates",
                "error_code": None,
                "retryable": False,
                **provenance,
            }

        validated_candidates: list[tuple[Any, float, str]] = []
        article_ids: list[str] = []

        for candidate in candidates:
            if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
                return technical_failure(
                    "invalid_retrieval_result",
                    "The knowledge store returned an invalid candidate.",
                )

            document, raw_score = candidate

            # why: A low-relevance Chroma candidate can have a negative finite
            # score. Preserve the raw value for gating instead of calling it an error.
            valid_score = (
                isinstance(raw_score, (int, float))
                and not isinstance(raw_score, bool)
                and isfinite(float(raw_score))
            )
            if not valid_score:
                return technical_failure(
                    "invalid_relevance_score",
                    "The knowledge store returned an invalid relevance score.",
                )

            metadata = getattr(document, "metadata", None)
            article_id = (
                metadata.get("article_id") if isinstance(metadata, dict) else None
            )
            candidate_account_id = (
                metadata.get("account_id") if isinstance(metadata, dict) else None
            )

            if (
                not isinstance(article_id, str)
                or not article_id.strip()
                or candidate_account_id != account_id
            ):
                return technical_failure(
                    "invalid_source_metadata",
                    "A retrieved candidate lacks trusted source metadata.",
                )

            validated_candidates.append(
                (document, float(raw_score), article_id)
            )
            article_ids.append(article_id)

        with get_session(udahub_engine) as session:
            # api: Column.in_(values) creates a parameterized SQL IN condition.
            source_rows = (
                session.query(udahub.Knowledge)
                .filter(
                    udahub.Knowledge.account_id == account_id,
                    udahub.Knowledge.article_id.in_(article_ids),
                )
                .all()
            )

            # Copy source fields while the SQLAlchemy session is still open.
            source_by_id = {
                row.article_id: {
                    "article_id": row.article_id,
                    "title": row.title,
                    "content": row.content,
                    "tags": row.tags,
                }
                for row in source_rows
            }

        articles: list[dict[str, Any]] = []
        gate_records: list[dict[str, Any]] = []

        for document, score, article_id in validated_candidates:
            source = source_by_id.get(article_id)
            if source is None:
                return technical_failure(
                    "source_record_not_found",
                    "A retrieved candidate could not be verified against its source.",
                )

            page_content = getattr(document, "page_content", None)
            if not isinstance(page_content, str) or not page_content.strip():
                return technical_failure(
                    "invalid_source_excerpt",
                    "A retrieved candidate has no verifiable excerpt.",
                )

            candidate_text = page_content.strip()
            source_content = source["content"]

            if candidate_text in source_content:
                excerpt = candidate_text
            elif source_content in candidate_text:
                excerpt = source_content
            else:
                return technical_failure(
                    "source_content_mismatch",
                    "A retrieved excerpt does not match its source article.",
                )

            gate_pass = score >= threshold
            gate_reason = "threshold_met" if gate_pass else "below_threshold"

            gate_record = {
                "account_id": account_id,
                "turn_id": turn_id,
                "issue_id": issue_id,
                "article_id": article_id,
                "score": score,
                "threshold": threshold,
                "gate_pass": gate_pass,
                "reason": gate_reason,
                "embedding_model": embedding_model,
                "score_convention": score_convention,
            }

            articles.append(
                {
                    **source,
                    "excerpt": excerpt,
                    "score": score,
                    "threshold": threshold,
                    "gate_pass": gate_pass,
                    "embedding_model": embedding_model,
                    "score_convention": score_convention,
                }
            )
            gate_records.append(gate_record)

        return {
            "status": "completed",
            "query": normalized_query,
            "articles": articles,
            "gate_records": gate_records,
            "reason": "candidates_evaluated",
            "error_code": None,
            "retryable": False,
            **provenance,
        }

    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return technical_failure(
            "transient_retrieval_error" if retryable else "retrieval_database_error",
            "The knowledge search could not be completed.",
            retryable=retryable,
        )
    # api: OpenAI APITimeoutError inherits APIConnectionError; neither requires
    # reading exception text or exposing provider details to classify the failure.
    except (TimeoutError, ConnectionError, APIConnectionError, RateLimitError):
        return technical_failure(
            "transient_retrieval_error",
            "The knowledge search could not be completed.",
            retryable=True,
        )
    except Exception:
        return technical_failure(
            "knowledge_retrieval_error",
            "The knowledge search could not be completed.",
        )
