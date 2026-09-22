# Agent tools

This folder contains the narrow support tools used by the specialist agents.
Prepared resources and trusted scope enter through runtime configuration; tools do
not create database engines, vector stores, or credentials.

## Trusted runtime configuration

`build_workflow` constructs a node-local `RunnableConfig` and injects the values
below into `config["configurable"]`. The language model supplies the search query;
it must not supply or override account, turn, or issue scope.

### Knowledge search

| Field | Type | Supplied by | Purpose |
|---|---|---|---|
| `knowledge_store` | `VectorStore` | `build_workflow` | Prepared, scored CultPass article index. Indexed documents must include `article_id` and `account_id` metadata. |
| `udahub_engine` | SQLAlchemy `Engine` | `build_workflow` | Reads authoritative UDA-Hub article records for source verification. |
| `retrieval_settings` | `RetrievalSettings` | `build_workflow` | Carries the evaluated threshold, candidate count, embedding model, and score convention. |
| `trusted_account_id` | non-empty `str` | Verified ticket context | Limits vector retrieval and source-record matching to the ticket's account. |
| `turn_id` | non-empty `str` | Active graph state | Associates retrieval evidence with the current customer turn. |
| `issue_id` | non-empty `str` | Supervisor assignment | Associates each score-gate record with the issue being evaluated. |

`retrieval_settings` has this contract:

| Field | Required value |
|---|---|
| `threshold` | Finite `float` from `0.0` to `1.0`, selected from the recorded retrieval evaluation rather than a default invented in code. |
| `candidate_count` | Positive `int` selected during retrieval evaluation. |
| `embedding_model` | Non-empty model identifier matching the prepared index. |
| `score_convention` | `"higher_is_more_relevant"`; a valid score passes when `score >= threshold`. |

The workflow copies the caller's configuration before adding these values. Resource
objects remain in runtime configuration or closures and are never stored in graph
state or checkpoints. Do not put API keys, connection strings, or other credentials
in this documented configuration contract.

## Knowledge-search result contract

`search_knowledge` returns evidence for the Knowledge agent; it does not choose a
case outcome or consume retry allowances.

- `status="completed"` means retrieval finished, including an empty search.
- `status="missing_input"` means the query was unusable.
- `status="technical_failure"` includes a stable `error_code` and deterministic
  `retryable` flag for the calling wrapper.
- `articles` contains source-verified article ID, title, content, excerpt, relevance
  score, threshold, gate result, embedding model, and score convention.
- `gate_records` keeps `score`, `threshold`, and `gate_pass` together for each
  issue/article evaluation.
- An empty completed search returns no articles and one gate record with
  `score=None`, `gate_pass=False`, and `reason="no_candidates"`.
