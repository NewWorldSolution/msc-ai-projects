# Foundation fixes — 2026-09-17

> Internal study/change report. Exclude this file and `TOOL_CORRECTIONS.md` from
> the final submission package. Keep them while studying. Keep implementation,
> useful docstrings and executable regression tests.

## Result

The confirmed foundation defects from the supplied review have been corrected.
**125 tests pass; 12 remain skipped.** The same result passes in an isolated export
of the files selected for version control, using the existing project environment.
You can begin implementing agents against this foundation. This is not a claim
that the unfinished agents, workflow, presentation layer or live retrieval work.

No existing dataset, database model, notebook, utility implementation or requirements
file was edited. SHA-256 comparisons verified 12 protected original files. No setup
notebook was executed, database file rebuilt, article invented, dependency installed
or live model/embedding request made. Tests use in-memory SQLite and fake providers.
The environment example was restored verbatim from Git; no local credentials were
read or copied into this report.

## How to read old and new versions

The code comparisons below use the **working files at the start of this request**
as the old version, rather than the older Git commit. This matters because several
completed tool implementations were already uncommitted when this work started.
A `-` line is old code; a `+` line is new code; each hunk header gives the old and new line locations.
New test files have no old version. The appendix includes their complete contents
as additions. Public tool names and signatures remain unchanged.

## What changed and why

| File / function | Old behavior | New behavior | Demonstrating test |
|---|---|---|---|
| `agentic/intake.py` / `prepare_case` | Invalid input updated only preparation status, retaining prior reply/save signals after state merge. | Clears outcome, approved response, handoff and persistence result; disables messaging. | `test_invalid_intake_cannot_reuse_saved_reply` uses real reducers and then calls Persist defensively. |
| `agentic/intake.py` / `prepare_case` | Every invocation reset current-turn controls. | Resets only for a different human-message ID; completed preparation for the same turn/thread returns an empty delta. | `test_same_turn_keeps_budgets_and_terminal_state`, `test_new_turn_resets_only_current_turn_allowances`. |
| `agentic/intake.py` / new `_lookup_with_retry` | Ticket/customer helpers ran once even after classified transient failures. | At most two calls per stable lookup key; unknown exceptions and completed absence are not retried. Records counts in state, preparation result and trace payload. | Parameterized `test_intake_retry_budget`, absence and unknown-exception tests. |
| `agentic/tools/knowledge_tools.py` / `search_knowledge` | Rejected finite negative and above-one scores as technical failures. | Retains the actual finite numeric score and compares it to the threshold; rejects booleans, NaN and infinity. | `test_relevance_score_validity_and_gate`. |
| `agentic/tools/customer_tools.py` / `lookup_customer` | Classified errors by substrings and connection-invalidated flag. | Uses the existing shared database classifier for SQLite busy/locked errors and pool timeouts. | Parameterized `test_shared_database_retry_policy`. |
| `agentic/tools/ticket_tools.py` / `save_ticket_outcome` | Used the same substring classification. | Uses the shared classifier; integrity conflicts remain non-retryable. | Same boundary tests plus existing rollback tests. |
| `agentic/tools/ticket_tools.py` / `save_ticket_outcome` | Could overwrite escalated status with open or resolved. | Rejects non-escalated automated outcomes on an already escalated ticket before any mutation. | `test_escalated_ticket_cannot_be_reopened`. |
| `agentic/persist.py` / `persist_case` | Saved a generated fallback but omitted it from the returned delta. | Returns the exact saved handoff with the successful result. | `test_fallback_handoff_is_returned_after_real_save`. |
| `agentic/persist.py` / `persist_case` | An unhashable outcome could raise TypeError in set membership. | Checks string type before allowed-value membership. | `test_persist_rejects_unhashable_outcome_without_sql`. |
| `agentic/persist.py` / docstring | Did not state where the post-commit memory event is stored. | Explicitly documents checkpoint-only memory summary reporting. | `test_memory_summary_is_checkpoint_only_after_core_commit`. |
| `agentic/tools/knowledge_tools.py` / exception handling | Missed installed OpenAI SDK connection, timeout and rate-limit classes. | Recognizes those exceptions explicitly; authentication and unknown errors remain sanitized/non-retryable. | `test_provider_retry_policy`, no provider calls. |
| `agentic/tools/knowledge_tools.py`, `reservation_tools.py` | Unused substring-marker constant and doubled import whitespace. | Removes the unused constant and corrects the import whitespace. | Syntax/import checks. |
| `tests/test_intake.py` | Expected no recorded attempts after successful lookups. | Requires one ticket and one customer attempt, with matching trace provenance. | Existing success test remains executable. |
| `tests/test_foundation_fixes.py` | File absent. | Adds 35 regression cases for these defects and preserved boundaries. | 35 passing cases. |
| `tests/test_tool_interfaces.py` | File absent. | Adds 9 model-schema, dispatch-name and actual read-only SQL cases. | 9 passing cases. |
| `.env.example` | Tracked template missing from the working tree. | Restores the original template with an empty key. | Exact comparison with HEAD. |
| `TOOL_CORRECTIONS.md` | Top status still reported the earlier Intake defects as pending. | Adds a superseded-status banner linking to this report, preserving the review history. | Documentation inspection. |

### Intake result additions

`preparation_result.thread_id` ties completed preparation to its trusted ticket
thread. `preparation_result.lookup_attempts` and the preparation trace payload now
record `ticket` and `customer` call counts. `technical_attempts` stores those counts
under keys such as `turn-1:intake:ticket:ticket-1` and
`turn-1:intake:customer:customer-1`.

A terminal result may still report that its underlying error was transient via
`retryable=True`; this classification does not grant further attempts. Intake has
already consumed its maximum of two and returns no work on prepared same-turn
reentry. Supervisor must handle failed preparation rather than retrying Intake as
an agent operation. A new human turn receives new counters.

## Version-control correction

The supplied review correctly identified runtime files missing from Git. Existing
`agentic/tools/database_errors.py`, `data/models/cultpass.py`, `utils.py`,
`requirements.txt` and `tests/test_review_regressions.py` are included in the fix
commit without content changes. Existing uncommitted tool implementations are also
included so a clean checkout has the code actually tested. This is tracking existing
work, not claiming these files were written during this request.

Only the scoped source/tests and review/learning records are included. Existing
untracked setup notebooks, data assets and other unrelated work are preserved in
place. They still need individual packaging review before submission; the isolated
export test proves code/test importability, not a complete runnable demo package.
The original `.env.example` restoration matches HEAD and therefore has no net Git
diff. A local `.env` is not staged.

## Disposition of the remaining review suggestions

- **Cosine index configuration:** no Chroma construction exists in the current
  setup notebooks; application setup is still a scaffold. There is no existing
  index-creation line to amend. Keep original notebooks/data intact. During agent
  integration, create the scored store with an explicit distance metric, record
  that metric alongside the score convention, and calibrate the threshold using
  real queries. Negative finite candidate scores are now handled correctly in
  the existing retrieval helper. No index was built or rebuilt here.
- **Memory summary audit:** selected the review's documentation option. The support
  transaction commits before auxiliary memory runs. Its result/event is returned
  for checkpointing, not retroactively included in that transaction's SQL rows.
  This limitation is now explicit and tested. A second audit-write transaction
  has not been added; SQL-only querying of post-commit memory outcomes remains a
  future enhancement if required.
- **Identity correction:** retain the architecture's trusted-mapping requirement.
  A different email match still returns `customer_scope_mismatch` without exposing
  that other customer's facts. This controlled code is already available for the
  agent/handoff path. A clarification is not authorization to relink identity.
  A permanently broken mapping requires human escalation. Do not weaken scope
  checks or silently change the agreed clarification policy.
- **Placeholder tests:** `tests/test_tools.py` and `tests/test_workflow.py` are
  preserved. Automatic approval review rejected wholesale replacement of the
  former as potentially destructive. Executable schema/read-only checks were
  added in `test_tool_interfaces.py` instead. Four old tool placeholders and
  eight workflow placeholders remain skipped and are not counted as coverage.
- **Agents/application:** model/tool binding, workflow routing, Operations attempt
  ownership, the application presentation guard, live score calibration and the
  additional support articles remain the next-stage work. Original articles and
  starter utility code were not replaced. The current app is not ready to run.

## Validation performed

From the solution directory, before fixes:

```bash
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Result: **81 passed, 12 skipped**, three existing warnings.

With the first regression file added, before implementation fixes:

```bash
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_foundation_fixes.py --tb=short
```

Result: **22 failed, 10 passed**. These failures demonstrated the defects before
correction. Three further regression cases and nine interface cases were added
later.

After fixes, the full-suite command above reports **125 passed, 12 skipped**,
with the same three dependency/deprecation warnings. Tests also pass with that
result from an isolated export of the selected Git index, using the project
interpreter and `-B -m pytest -q -p no:cacheprovider`.

Additional checks:

- In-memory `compile(source, filename, 'exec')`: all 34 Python files in `agentic/`
  and `tests/` passed without generating bytecode.
- `importlib.import_module`: all 13 selected tools/state/Intake/Persist/workflow
  modules imported in the existing environment.
- `git diff --check -- agentic tests .env.example`: passed.
- Full staged whitespace check flags three pre-existing formatting locations in
  the unchanged `utils.py` (two trailing spaces) and `data/models/cultpass.py`
  (an extra EOF blank line). These were preserved to honor the original-file
  constraint; checks on edited implementation/test/report files pass.
- SHA-256 before/after checks: all 12 protected files unchanged.
- Ruff, mypy, pyflakes and flake8 are absent from the environment; those checks
  could not run. No dependencies were installed.
- An initial import-check command accidentally used system Python and failed
  because that interpreter lacks `langchain_core`; the same check then passed
  using the required project interpreter. Test runs used the project environment.
- No fresh dependency installation, live model call, live embedding request,
  complete workflow or production database operation was tested.

## Learning records and submission cleanup

Only new entries were appended to repository-level `knowledge/errors.md` and
`knowledge/idioms.md`. Prior entries and unrelated working changes were preserved.
These records remain outside the submission package. The appended text appears
at the end of this report for an exact audit.

Keep this report while studying, then exclude it and other temporary correction
reports from the submission. Preserve actual code, useful docstrings and executable
tests. Do not delete original datasets or notebooks during report cleanup.

## Exact old/new changes

### `agentic/intake.py`

```diff
--- OLD/agentic/intake.py
+++ NEW/agentic/intake.py
@@ -2,0 +3 @@
+from collections.abc import Callable
@@ -12,0 +14,52 @@
+
+
+def _lookup_with_retry(
+    operation: Callable[[], dict[str, Any]],
+    *,
+    attempts: dict[str, int],
+    lookup_key: str,
+    failure_outcome: str,
+    unexpected_error_code: str,
+    reason: str,
+) -> dict[str, Any]:
+    """Run a lookup at most twice, counting attempts under a stable turn key.
+
+    The caller supplies the trusted operation and a copy of the current turn's
+    counters. Only a classified technical failure can spend the second attempt.
+    Unknown exceptions are sanitized, and exhausted keys never access storage.
+    """
+    used = attempts.get(lookup_key, 0)
+    if not isinstance(used, int) or isinstance(used, bool) or used < 0:
+        # why: Malformed checkpoint counters cannot grant a fresh retry allowance.
+        used = 2
+    result = {
+        "outcome": failure_outcome,
+        "records": [],
+        "reason": "The lookup attempt allowance is exhausted.",
+        "error_code": "lookup_attempts_exhausted",
+        "retryable": False,
+    }
+    while used < 2:
+        used += 1
+        attempts[lookup_key] = used
+        try:
+            result = operation()
+        except Exception:
+            return {
+                "outcome": failure_outcome,
+                "records": [],
+                "reason": reason,
+                "error_code": unexpected_error_code,
+                "retryable": False,
+            }
+        if not isinstance(result, dict):
+            return {
+                "outcome": failure_outcome,
+                "records": [],
+                "reason": "The lookup returned an invalid result.",
+                "error_code": "invalid_lookup_result",
+                "retryable": False,
+            }
+        if result.get("outcome") != failure_outcome or result.get("retryable") is not True:
+            break
+    return result
@@ -29 +82 @@
-        reset turn-scoped controls, a controlled preparation result and at most one
+        new-turn controls, a controlled preparation result and at most one
@@ -36,0 +90,3 @@
+        A prepared same-turn reentry returns no delta; it cannot renew budgets or
+        overwrite terminal decisions. Each lookup permits one classified transient
+        retry, recorded in technical_attempts and preparation provenance.
@@ -51,0 +108,6 @@
+            # why: Omitted keys survive graph merging; clear prior release signals.
+            "outcome": None,
+            "approved_response": None,
+            "handoff": None,
+            "persistence_result": None,
+            "messaging_disabled": True,
@@ -63,31 +124,0 @@
-    # why: These are all current-turn fields. Omitting messages, tool_results and
-    # identity_correction preserves their checkpointed or ticket-scoped history.
-    update: CaseState = {
-        "turn_id": turn_id,
-        "account_context": {},
-        "customer_memory": [],
-        "memory_updates": [],
-        "classification_status": "pending",
-        "classification": None,
-        "issues": [],
-        "assignment": None,
-        "retrieved_articles": [],
-        "technical_attempts": {},
-        "workflow_steps": 0,
-        "model_calls": 0,
-        "response_stage": None,
-        "response_draft": None,
-        "draft_blocker": None,
-        "review_result": None,
-        "review_history": [],
-        "original_review_attempts": 0,
-        "acknowledgement_review_attempts": 0,
-        "approved_response": None,
-        "outcome": None,
-        "escalation_reason": None,
-        "acknowledgement_permitted": False,
-        "messaging_disabled": False,
-        "handoff": None,
-        "persistence_result": None,
-    }
-
@@ -98,0 +130,45 @@
+
+    is_new_turn = state.get("turn_id") != turn_id
+    prior_preparation = state.get("preparation_result")
+    if (
+        not is_new_turn
+        and isinstance(prior_preparation, dict)
+        and prior_preparation.get("thread_id") == thread_id
+    ):
+        # why: A completed preparation (including failure) has spent its allowance.
+        return {}
+
+    # why: Only a new human-message ID resets current-turn controls. History and
+    # ticket-scoped identity correction are preserved by omitting them from the delta.
+    update: CaseState = {}
+    if is_new_turn:
+        update = {
+            "turn_id": turn_id,
+            "account_context": {},
+            "customer_memory": [],
+            "memory_updates": [],
+            "classification_status": "pending",
+            "classification": None,
+            "issues": [],
+            "assignment": None,
+            "retrieved_articles": [],
+            "technical_attempts": {},
+            "workflow_steps": 0,
+            "model_calls": 0,
+            "response_stage": None,
+            "response_draft": None,
+            "draft_blocker": None,
+            "review_result": None,
+            "review_history": [],
+            "original_review_attempts": 0,
+            "acknowledgement_review_attempts": 0,
+            "approved_response": None,
+            "outcome": None,
+            "escalation_reason": None,
+            "acknowledgement_permitted": False,
+            "messaging_disabled": False,
+            "handoff": None,
+            "persistence_result": None,
+        }
+    attempts = dict(update.get("technical_attempts", state.get("technical_attempts", {})))
+    lookup_attempts = {"ticket": 0, "customer": 0}
@@ -117,9 +193,11 @@
-        try:
-            ticket_result = load_ticket_context(thread_id, udahub_engine)
-        except Exception:
-            ticket_result = {
-                "outcome": "database_error",
-                "reason": "The ticket context could not be loaded.",
-                "error_code": "unexpected_ticket_lookup_error",
-                "retryable": False,
-            }
+        ticket_key = f"{turn_id}:intake:ticket:{thread_id}"
+        # api: lambda creates a deferred call so both attempts reuse trusted arguments.
+        ticket_result = _lookup_with_retry(
+            lambda: load_ticket_context(thread_id, udahub_engine),
+            attempts=attempts,
+            lookup_key=ticket_key,
+            failure_outcome="database_error",
+            unexpected_error_code="unexpected_ticket_lookup_error",
+            reason="The ticket context could not be loaded.",
+        )
+        lookup_attempts["ticket"] = attempts.get(ticket_key, 0)
@@ -185,12 +263,10 @@
-                try:
-                    customer_result = lookup_customer(
-                        cultpass_engine, customer_id=customer_id
-                    )
-                except Exception:
-                    customer_result = {
-                        "outcome": "technical_failure",
-                        "records": [],
-                        "reason": "The CultPass customer lookup could not be completed.",
-                        "error_code": "unexpected_customer_lookup_error",
-                        "retryable": False,
-                    }
+                customer_key = f"{turn_id}:intake:customer:{customer_id}"
+                customer_result = _lookup_with_retry(
+                    lambda: lookup_customer(cultpass_engine, customer_id=customer_id),
+                    attempts=attempts,
+                    lookup_key=customer_key,
+                    failure_outcome="technical_failure",
+                    unexpected_error_code="unexpected_customer_lookup_error",
+                    reason="The CultPass customer lookup could not be completed.",
+                )
+                lookup_attempts["customer"] = attempts.get(customer_key, 0)
@@ -302,0 +379 @@
+    update["technical_attempts"] = attempts
@@ -303,0 +381,2 @@
+        "thread_id": thread_id,
+        "lookup_attempts": lookup_attempts,
@@ -323,0 +403 @@
+            "lookup_attempts": lookup_attempts,
```

### `agentic/persist.py`

```diff
--- OLD/agentic/persist.py
+++ NEW/agentic/persist.py
@@ -206 +206,6 @@
-    """Execute Supervisor's approved final decision and report the actual save."""
+    """Execute Supervisor's approved final decision and report the actual save.
+
+    The memory summary is created after the support commit and returned in the
+    graph trace for checkpointing; it is not part of that atomic SQL transaction.
+    A saved fallback handoff is also returned so checkpoint and support data agree.
+    """
@@ -234 +239 @@
-    if case_outcome not in _CASE_OUTCOMES:
+    if not isinstance(case_outcome, str) or case_outcome not in _CASE_OUTCOMES:
@@ -354,0 +360 @@
+        "handoff": handoff,
```

### `agentic/tools/customer_tools.py`

```diff
--- OLD/agentic/tools/customer_tools.py
+++ NEW/agentic/tools/customer_tools.py
@@ -6 +6 @@
-from sqlalchemy.exc import OperationalError, SQLAlchemyError
+from sqlalchemy.exc import SQLAlchemyError
@@ -9,0 +10 @@
+from agentic.tools.database_errors import is_retryable_database_error
@@ -87,15 +88,2 @@
-    except OperationalError as error:
-        error_text = str(error.orig).lower()
-        transient_markers = (
-            "database is locked",
-            "database is busy",
-            "deadlock",
-            "timeout",
-            "temporarily unavailable",
-            "connection reset",
-        )
-        # api: connection_invalidated reports that SQLAlchemy discarded the failed
-        # connection, which makes a later attempt potentially useful.
-        retryable = error.connection_invalidated or any(
-            marker in error_text for marker in transient_markers
-        )
+    except SQLAlchemyError as error:
+        retryable = is_retryable_database_error(error)
@@ -110,8 +97,0 @@
-        }
-    except SQLAlchemyError:
-        return {
-            "outcome": "technical_failure",
-            "records": [],
-            "reason": "The CultPass customer lookup could not be completed.",
-            "error_code": "database_error",
-            "retryable": False,
```

### `agentic/tools/knowledge_tools.py`

```diff
--- OLD/agentic/tools/knowledge_tools.py
+++ NEW/agentic/tools/knowledge_tools.py
@@ -6,0 +7 @@
+from openai import APIConnectionError, RateLimitError
@@ -15,12 +15,0 @@
-
-_TRANSIENT_MARKERS = (
-    "database is locked",
-    "database is busy",
-    "deadlock",
-    "timeout",
-    "timed out",
-    "temporarily unavailable",
-    "connection reset",
-    "connection aborted",
-    "rate limit",
-)
@@ -55,2 +44,3 @@
-        database failures and built-in timeout/connection exceptions are
-        retryable. Unknown exceptions are sanitized and non-retryable.
+        database failures, built-in timeout/connection exceptions and OpenAI
+        connection/timeout/rate-limit exceptions are retryable. Unknown exceptions
+        are sanitized and non-retryable.
@@ -211,0 +202,2 @@
+            # why: A low-relevance Chroma candidate can have a negative finite
+            # score. Preserve the raw value for gating instead of calling it an error.
@@ -216 +207,0 @@
-                and 0.0 <= float(raw_score) <= 1.0
@@ -347 +338,3 @@
-    except (TimeoutError, ConnectionError):
+    # api: OpenAI APITimeoutError inherits APIConnectionError; neither requires
+    # reading exception text or exposing provider details to classify the failure.
+    except (TimeoutError, ConnectionError, APIConnectionError, RateLimitError):
```

### `agentic/tools/ticket_tools.py`

```diff
--- OLD/agentic/tools/ticket_tools.py
+++ NEW/agentic/tools/ticket_tools.py
@@ -7 +7 @@
-from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
+from sqlalchemy.exc import IntegrityError, SQLAlchemyError
@@ -535,0 +536,11 @@
+                }
+
+            # why: Automated follow-up cannot undo an existing human handoff.
+            if metadata.status == "escalated" and case_outcome != "escalated":
+                return {
+                    "saved": False,
+                    "ticket_id": ticket_id,
+                    "turn_id": turn_id,
+                    "error_code": "ticket_already_escalated",
+                    "retryable": False,
+                    "reason": "The ticket is already assigned to human handling.",
@@ -598,13 +609,2 @@
-    except OperationalError as error:
-        error_text = str(error.orig).lower()
-        transient_markers = (
-            "database is locked",
-            "database is busy",
-            "deadlock",
-            "timeout",
-            "temporarily unavailable",
-            "connection reset",
-        )
-        retryable = error.connection_invalidated or any(
-            marker in error_text for marker in transient_markers
-        )
+    except SQLAlchemyError as error:
+        retryable = is_retryable_database_error(error)
@@ -619,9 +618,0 @@
-            "reason": "The support outcome could not be saved.",
-        }
-    except SQLAlchemyError:
-        return {
-            "saved": False,
-            "ticket_id": ticket_id,
-            "turn_id": turn_id,
-            "error_code": "database_error",
-            "retryable": False,
```

### `agentic/tools/reservation_tools.py`

```diff
--- OLD/agentic/tools/reservation_tools.py
+++ NEW/agentic/tools/reservation_tools.py
@@ -8 +8 @@
-from sqlalchemy.exc import  SQLAlchemyError
+from sqlalchemy.exc import SQLAlchemyError
```

### `tests/test_intake.py`

```diff
--- OLD/tests/test_intake.py
+++ NEW/tests/test_intake.py
@@ -151 +151,4 @@
-    assert result["technical_attempts"] == {}
+    assert result["technical_attempts"] == {
+        "turn-1:intake:ticket:ticket-1": 1,
+        "turn-1:intake:customer:customer-1": 1,
+    }
@@ -173,0 +177 @@
+                "lookup_attempts": {"ticket": 1, "customer": 1},
```

### `tests/test_foundation_fixes.py`

Old version: absent from the working tree.

```diff
--- /dev/null
+++ NEW/tests/test_foundation_fixes.py
@@ -0,0 +1,268 @@
+"""Regression tests for the September foundation fixes; no external services."""
+
+from contextlib import contextmanager
+import sqlite3
+from unittest.mock import Mock
+
+import httpx
+import pytest
+from langchain_core.documents import Document
+from langchain_core.messages import AIMessage, HumanMessage
+from langgraph.graph import END, START, StateGraph
+from langgraph.store.memory import InMemoryStore
+from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
+from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError
+
+from agentic import intake, persist
+from agentic.state import CaseState
+from agentic.tools import customer_tools, knowledge_tools, ticket_tools
+from data.models import udahub
+from utils import get_session
+from test_intake import seeded_resources, trusted_config
+from test_persist import resolved_state
+from test_review_regressions import ScoredStore, knowledge_config
+from test_ticket_tools import engine, resolved_outcome
+
+
+@pytest.fixture
+def intake_resources():
+    """Dispose both isolated databases after each Intake regression."""
+    support, business = seeded_resources()
+    yield trusted_config(support, business)
+    support.dispose()
+    business.dispose()
+
+
+def run_intake_graph(state, config):
+    """Apply Intake's delta through the real CaseState reducers."""
+    def prepare(current: CaseState) -> CaseState:
+        return intake.prepare_case(current, config, store=InMemoryStore())
+
+    graph = StateGraph(CaseState)
+    graph.add_node("prepare", prepare)
+    graph.add_edge(START, "prepare")
+    graph.add_edge("prepare", END)
+    return graph.compile().invoke(state)
+
+
+def test_invalid_intake_cannot_reuse_saved_reply(engine, monkeypatch):
+    """A failed preparation cannot release an older reply even if Persist runs."""
+    state = resolved_state()
+    state.update(persistence_result={"saved": True}, messaging_disabled=False)
+    state["messages"].append(AIMessage(content="Old reply", id="old-reply"))
+    lookup = Mock(side_effect=AssertionError("No lookup permitted"))
+    monkeypatch.setattr(intake, "load_ticket_context", lookup)
+    config = {"configurable": {"thread_id": "ticket-1"}}
+    merged = run_intake_graph(state, config)
+    assert lookup.call_count == 0
+    assert merged["preparation_result"]["outcome"] == "failed"
+    assert merged["approved_response"] is None
+    assert merged["persistence_result"] is None
+    assert merged["outcome"] is None
+    assert merged["handoff"] is None
+    assert merged["messaging_disabled"] is True
+    result = persist.persist_case(merged, config, udahub_engine=engine, store=InMemoryStore())
+    assert result["persistence_result"]["saved"] is False
+    assert "messages" not in result
+    with get_session(engine) as session:
+        assert session.query(udahub.TicketMessage).count() == 0
+
+
+def test_same_turn_keeps_budgets_and_terminal_state(intake_resources, monkeypatch):
+    """Reentry must neither spend another lookup nor renew any allowance."""
+    initial = {"messages": [HumanMessage(content="Help", id="turn-1")]}
+    merged = run_intake_graph(initial, intake_resources)
+    spent = {**merged["technical_attempts"], "operations:lookup": 2}
+    controls = {
+        "technical_attempts": spent, "workflow_steps": 12, "model_calls": 8,
+        "original_review_attempts": 2, "acknowledgement_review_attempts": 2,
+        "classification_status": "completed", "classification": {"urgency": "urgent"},
+        "messaging_disabled": True, "handoff": {"message_id": "handoff-1"},
+        "outcome": "escalated",
+    }
+    merged.update(controls)
+    lookup = Mock(side_effect=AssertionError("Same turn is already prepared"))
+    monkeypatch.setattr(intake, "load_ticket_context", lookup)
+    second = run_intake_graph(merged, intake_resources)
+    assert lookup.call_count == 0
+    for field, value in controls.items():
+        assert second[field] == value
+    assert second["trace"] == merged["trace"]
+
+
+@pytest.mark.parametrize("boundary", ["load_ticket_context", "lookup_customer"])
+@pytest.mark.parametrize("failure_count,retryable,expected_calls", [(1, True, 2), (2, True, 2), (1, False, 1)])
+def test_intake_retry_budget(boundary, failure_count, retryable, expected_calls, intake_resources, monkeypatch):
+    """Only a transient failure gets a second attempt; reentry cannot add a third."""
+    actual = getattr(intake, boundary)
+    calls = []
+
+    def failing_then_real(*args, **kwargs):
+        calls.append((args, kwargs))
+        if len(calls) <= failure_count:
+            return {
+                "outcome": "database_error" if boundary == "load_ticket_context" else "technical_failure",
+                "records": [], "error_code": "transient_database_error" if retryable else "database_error",
+                "reason": "Controlled database failure.", "retryable": retryable,
+            }
+        return actual(*args, **kwargs)
+
+    monkeypatch.setattr(intake, boundary, failing_then_real)
+    state = {"messages": [HumanMessage(content="Help", id="turn-1")]}
+    result = run_intake_graph(state, intake_resources)
+    assert len(calls) == expected_calls
+    if expected_calls == 2:
+        assert calls[0] == calls[1]
+    expected = "ready" if retryable and failure_count == 1 else "failed"
+    assert result["preparation_result"]["outcome"] == expected
+    key = "ticket" if boundary == "load_ticket_context" else "customer"
+    assert result["preparation_result"]["lookup_attempts"][key] == expected_calls
+    assert result["trace"][-1]["payload"]["lookup_attempts"][key] == expected_calls
+    repeated = run_intake_graph(result, intake_resources)
+    assert len(calls) == expected_calls
+    assert repeated["technical_attempts"] == result["technical_attempts"]
+
+
+@pytest.mark.parametrize("boundary,outcome", [("load_ticket_context", "missing_ticket"), ("lookup_customer", "not_found")])
+def test_intake_never_retries_completed_absence(boundary, outcome, intake_resources, monkeypatch):
+    """Even an inconsistent retry flag cannot turn absence into a retry loop."""
+    lookup = Mock(return_value={"outcome": outcome, "records": [], "retryable": True})
+    monkeypatch.setattr(intake, boundary, lookup)
+    result = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
+    assert lookup.call_count == 1
+    assert result["preparation_result"]["outcome"] != "ready"
+
+
+@pytest.mark.parametrize("boundary", ["load_ticket_context", "lookup_customer"])
+def test_intake_unknown_exception_is_sanitized_without_retry(boundary, intake_resources, monkeypatch):
+    """Unclassified exceptions must not leak details or receive a retry."""
+    lookup = Mock(side_effect=RuntimeError("secret database endpoint"))
+    monkeypatch.setattr(intake, boundary, lookup)
+    result = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
+    assert lookup.call_count == 1
+    assert result["preparation_result"]["retryable"] is False
+    assert "secret database endpoint" not in str(result)
+
+
+@pytest.mark.parametrize("score", [-0.02, 1.01, float("nan"), float("inf"), True])
+def test_relevance_score_validity_and_gate(knowledge_config, score):
+    """Finite raw scores are evidence; non-finite scores are technical failures."""
+    document = Document(page_content="Stored policy.", metadata={"article_id": "article1", "account_id": "account1"})
+    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([(document, score)])
+    result = knowledge_tools.search_knowledge("policy", knowledge_config)
+    if score in (-0.02, 1.01):
+        assert result["status"] == "completed"
+        assert result["gate_records"][0]["score"] == score
+        assert result["gate_records"][0]["gate_pass"] is (score >= 0.7)
+    else:
+        assert result["error_code"] == "invalid_relevance_score"
+        assert result["retryable"] is False
+
+
+@pytest.mark.parametrize("boundary", ["customer", "save"])
+@pytest.mark.parametrize("failure,expected", [
+    (OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: timeout")), False),
+    (OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: deadlock")), False),
+    (OperationalError("SELECT", {}, sqlite3.OperationalError("database is locked")), True),
+    (PoolTimeoutError("pool exhausted"), True),
+])
+def test_shared_database_retry_policy(boundary, failure, expected, monkeypatch):
+    """Database names and generic wording cannot classify permanent errors as transient."""
+    @contextmanager
+    def failing_session(_engine):
+        raise failure
+        yield
+
+    module = customer_tools if boundary == "customer" else ticket_tools
+    monkeypatch.setattr(module, "get_session", failing_session)
+    result = (module.lookup_customer(object(), customer_id="customer-1") if boundary == "customer"
+              else module.save_ticket_outcome("ticket-1", resolved_outcome(), object()))
+    assert result["retryable"] is expected
+    assert "SELECT" not in result["reason"]
+
+
+@pytest.mark.parametrize("error_kind,expected", [("timeout", True), ("connection", True), ("rate_limit", True), ("authentication", False)])
+def test_provider_retry_policy(knowledge_config, error_kind, expected):
+    """Classify installed embedding-provider exceptions without invoking the provider."""
+    request = httpx.Request("POST", "https://example.invalid/embeddings")
+    if error_kind == "timeout":
+        error = APITimeoutError(request=request)
+    elif error_kind == "connection":
+        error = APIConnectionError(request=request)
+    else:
+        status = 429 if error_kind == "rate_limit" else 401
+        cls = RateLimitError if status == 429 else AuthenticationError
+        error = cls("private provider error", response=httpx.Response(status, request=request), body=None)
+    knowledge_config["configurable"]["knowledge_store"] = ScoredStore(error=error)
+    result = knowledge_tools.search_knowledge("policy", knowledge_config)
+    assert result["retryable"] is expected
+    assert "private provider error" not in str(result)
+
+
+@pytest.mark.parametrize("outcome", ["resolved", "information_requested"])
+def test_escalated_ticket_cannot_be_reopened(engine, outcome):
+    """A terminal human handoff is protected before any new support writes."""
+    with get_session(engine) as session:
+        session.get(udahub.TicketMetadata, "ticket-1").status = "escalated"
+    payload = resolved_outcome()
+    payload["outcome"] = outcome
+    payload["approved_response"]["outcome"] = outcome
+    result = ticket_tools.save_ticket_outcome("ticket-1", payload, engine)
+    assert result["saved"] is False
+    assert result["error_code"] == "ticket_already_escalated"
+    assert result["retryable"] is False
+    with get_session(engine) as session:
+        assert session.get(udahub.TicketMetadata, "ticket-1").status == "escalated"
+        assert session.query(udahub.TicketMessage).count() == 0
+        assert session.query(udahub.TraceEvent).count() == 0
+
+
+def test_fallback_handoff_is_returned_after_real_save(engine):
+    """The checkpoint must contain the same handoff that the support database saved."""
+    state = resolved_state()
+    state.update(outcome="escalated", approved_response=None, review_result=None, handoff=None)
+    result = persist.persist_case(state, {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
+    assert result["persistence_result"]["saved"] is True
+    assert result["handoff"]["handoff_source"] == "deterministic_fallback"
+    with get_session(engine) as session:
+        import json
+        saved = session.get(udahub.TicketMessage, result["handoff"]["message_id"])
+        assert json.loads(saved.content) == {**result["handoff"], "message_type": "human_handoff"}
+
+
+def test_new_turn_resets_only_current_turn_allowances(intake_resources):
+    """A different stable human ID gets new counters while ticket history survives."""
+    state = run_intake_graph({"messages": [HumanMessage(content="Help", id="turn-1")]}, intake_resources)
+    state.update(original_review_attempts=2, messaging_disabled=True,
+                 identity_correction={"asked": True}, technical_attempts={"spent-old-key": 2})
+    state["messages"].append(HumanMessage(content="New question", id="turn-2"))
+    result = run_intake_graph(state, intake_resources)
+    assert result["original_review_attempts"] == 0
+    assert result["messaging_disabled"] is False
+    assert result["identity_correction"] == {"asked": True}
+    assert result["technical_attempts"] == {
+        "turn-2:intake:ticket:ticket-1": 1,
+        "turn-2:intake:customer:customer-1": 1,
+    }
+    assert len(result["trace"]) == 2
+
+
+def test_memory_summary_is_checkpoint_only_after_core_commit(engine):
+    """Document the supported audit boundary with real database assertions."""
+    result = persist.persist_case(resolved_state(), {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
+    assert result["persistence_result"]["saved"] is True
+    assert result["persistence_result"]["memory"]["status"] == "succeeded"
+    assert result["trace"][-1]["event_type"] == "memory_write_summary"
+    with get_session(engine) as session:
+        assert session.get(udahub.TraceEvent, "turn-1:persist:outcome") is not None
+        assert session.get(udahub.TraceEvent, "turn-1:persist:memory") is None
+
+
+def test_persist_rejects_unhashable_outcome_without_sql(engine):
+    """Malformed outcome values produce a controlled failure instead of TypeError."""
+    state = resolved_state()
+    state["outcome"] = []
+    result = persist.persist_case(state, {"configurable": {"thread_id": "ticket-1"}}, udahub_engine=engine, store=InMemoryStore())
+    assert result["persistence_result"]["error_code"] == "invalid_case_outcome"
+    assert result["persistence_result"]["attempts"] == 0
+    assert "messages" not in result
```

### `tests/test_tool_interfaces.py`

Old version: absent from the working tree.

```diff
--- /dev/null
+++ NEW/tests/test_tool_interfaces.py
@@ -0,0 +1,51 @@
+"""Model-facing schemas and actual SQL checks for the Operations tool set."""
+
+import pytest
+from sqlalchemy import event
+
+from agentic.tools.customer_tools import lookup_customer_tool
+from agentic.tools.experience_tools import lookup_experiences
+from agentic.tools.reservation_tools import lookup_reservations
+from agentic.tools.subscription_tools import lookup_subscription
+from test_review_regressions import business_engine, config
+
+
+TOOLS = [lookup_customer_tool, lookup_subscription, lookup_reservations, lookup_experiences]
+
+
+@pytest.mark.parametrize("tool", TOOLS)
+def test_model_schema_hides_trusted_runtime_context(tool):
+    """The model can supply lookup arguments, but cannot replace trusted scope."""
+    properties = tool.tool_call_schema.model_json_schema()["properties"]
+    assert not ({"config", "cultpass_engine", "trusted_customer_id", "turn_id",
+                 "issue_id", "lookup_key", "attempt", "lookup_mode"} & properties.keys())
+
+
+def test_decorated_tools_have_stable_distinct_names():
+    """Operations can dispatch each declared tool without a name collision."""
+    assert {tool.name for tool in TOOLS} == {
+        "lookup_customer_tool", "lookup_subscription", "lookup_reservations", "lookup_experiences",
+    }
+
+
+@pytest.mark.parametrize("tool,args", [
+    (lookup_customer_tool, {"customer_id": "c1"}),
+    (lookup_subscription, {"customer_id": "c1"}),
+    (lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
+    (lookup_experiences, {"experience_id": "e1"}),
+])
+def test_operations_tools_issue_only_reads(business_engine, tool, args):
+    """Observe executed statements so a successful lookup cannot hide business writes."""
+    statements = []
+
+    def record_statement(connection, cursor, statement, parameters, context, executemany):
+        statements.append(statement)
+
+    event.listen(business_engine, "before_cursor_execute", record_statement)
+    try:
+        result = tool.invoke(args, config=config(business_engine))
+    finally:
+        event.remove(business_engine, "before_cursor_execute", record_statement)
+    assert result["outcome"] == "found"
+    assert statements
+    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
```

### `.env.example`

Old version: absent from the working tree.

```diff
--- /dev/null
+++ NEW/.env.example
@@ -0,0 +1,3 @@
+# Copy this file to .env and fill in your Vocareum key locally.
+OPENAI_BASE_URL=https://openai.vocareum.com/v1
+VOCAREUM_API_KEY=
```

### `TOOL_CORRECTIONS.md`

```diff
--- OLD/TOOL_CORRECTIONS.md
+++ NEW/TOOL_CORRECTIONS.md
@@ -0,0 +1,5 @@
+> **Superseded review status — 2026-09-17:** the confirmed foundation defects,
+> including section 12's Intake findings, have now been corrected and regression
+> tested. See [the actual before/after change log](FOUNDATION_FIXES_2026-09-17.md).
+> Older observations and proposed snippets below are retained as study history.
+
```

## Original-file integrity evidence

Every listed file retained the same bytes before and after the fixes.

| Protected file | SHA-256 before = after |
|---|---|
| `01_external_db_setup.ipynb` | `dfcc4eac62cc70d6bc57755338c1e00138c6da63b512ae8bd2297072d679d60a` |
| `02_core_db_setup.ipynb` | `da91c1f1032329b8a9fd1cc488ada291488f13449f33824462613da7ca6b08a7` |
| `03_agentic_app.ipynb` | `4fe1c39374507b7f522e4290da75902524172a5b91cc9a512e0ca5394bd4610d` |
| `data/core/README.md` | `db93d55ddd2c65cb34be6d6678678e9f73d61ac7937da6898d52e5d04dd3610b` |
| `data/external/README.md` | `bc41bbf6cc6e32da7e34edf1046c3fbc93561792b5c92c3a686c6771da084200` |
| `data/external/cultpass_articles.jsonl` | `01ccd89c29562e20da490e59ee5bc0f56fe5b8c4c3e45832380e2afb362e2bef` |
| `data/external/cultpass_experiences.jsonl` | `4c1707f8ed9b94637bff3fcc442cce4f44c48edbb1e9cdb4f2fcd92079c21ce8` |
| `data/external/cultpass_users.jsonl` | `b9c5f95a40b275aefff783b339a364ae7cc6d38b03a656388cfbda6ccd9d657f` |
| `data/models/cultpass.py` | `afea9b9a618e0a82ed4b680ace7cdfb08954725daff44eaa45cad29f355aaf13` |
| `data/models/udahub.py` | `d13f37c3c916c9faf2f05b499bdb7009c2a4c62a3c83769dbc75b68e56d495c1` |
| `requirements.txt` | `ef3013ddd2dcae2a490271ba16483de94a6b871354347147465c5cffa4687a95` |
| `utils.py` | `9102fbd3f5b608da68e2dfbc660b041a9687338223ae785638f80cb33809951a` |

## Appended only: `knowledge/errors.md`

Old entries unchanged; the following is new.

```markdown


## 2026-09-17 · nd901 · autonomous-knowledge-agent / Intake lifecycle and retries
- **Symptom**: Same-turn reentry reset spent budgets; invalid input retained an older approved reply; classified transient lookups were attempted only once.
- **Cause**: The state reset was unconditional, the invalid-input delta omitted release indicators, and the retry flag had no executor.
- **Fix**: Clear release indicators on invalid input, initialize controls only for a new turn, preserve completed same-turn preparation, and execute at most two lookup attempts with stable keys and trace counts. Real StateGraph and database regression tests pass.
- **Misconception**:
- **Card**: no

## 2026-09-17 · nd901 · autonomous-knowledge-agent / finite retrieval scores
- **Symptom**: A valid low-relevance candidate with score -0.02 became a technical failure instead of failing the relevance gate.
- **Cause**: Candidate validation incorrectly required every raw score to be between zero and one.
- **Fix**: Retain finite numeric validation and compare the unchanged raw score to the threshold. Negative scores, above-one scores and invalid numeric values have regression coverage.
- **Misconception**:
- **Card**: no

## 2026-09-17 · nd901 · autonomous-knowledge-agent / consistent retry classification
- **Symptom**: A missing table named timeout or deadlock was retryable, while connection-pool and embedding-provider timeouts were not.
- **Cause**: Customer/read and support/write boundaries used substring matching, and retrieval recognized only built-in connection exceptions.
- **Fix**: Use the existing SQLite/pool classifier at both database boundaries; explicitly recognize installed OpenAI connection, timeout and rate-limit exceptions. Authentication and unknown errors remain non-retryable.
- **Misconception**:
- **Card**: no

## 2026-09-17 · nd901 · autonomous-knowledge-agent / persistence consistency
- **Symptom**: A saved fallback handoff was absent from the checkpoint delta; another automated outcome could reopen an escalated ticket; an unhashable outcome could raise TypeError.
- **Cause**: Persist omitted the generated handoff, the write boundary lacked an escalation guard, and enum membership ran before type validation.
- **Fix**: Return the saved handoff, reject non-escalated writes on an escalated ticket before mutation, and validate outcome type. Verify with in-memory transactions and regression tests.
- **Misconception**:
- **Card**: no

## 2026-09-17 · nd901 · autonomous-knowledge-agent / missing version-controlled dependencies
- **Symptom**: The working solution imported files that were absent from the committed tree.
- **Cause**: Runtime source, the existing error classifier and regression tests had remained untracked.
- **Fix**: Include required existing source and tests unchanged where no correction was needed; restore the original credential-free environment template. The exported Git index passes 125 tests with 12 planned skips using the existing environment.
- **Misconception**:
- **Card**: no
```

## Appended only: `knowledge/idioms.md`

Old entries unchanged; the following is new.

```markdown


### `Callable[[], dict[str, Any]]` with `lambda: lookup(...)`
**Met in**: nd901 / autonomous-knowledge-agent / Intake      **Date**: 2026-09-17
**Does**: describes a no-argument function returning a dictionary; the lambda delays a trusted lookup until the bounded retry helper calls it.
**vs. the thing I'd have reached for**: passing `lookup(...)` executes immediately and passes its result, so the retry helper cannot call it again.
**Mine now**: no

### `except (APIConnectionError, RateLimitError)`
**Met in**: nd901 / autonomous-knowledge-agent / knowledge retrieval      **Date**: 2026-09-17
**Does**: catches the installed OpenAI SDK's named failure classes; `APITimeoutError` is a subclass of `APIConnectionError`.
**vs. the thing I'd have reached for**: looking for the word timeout in exception text can misclassify a permanent error, and built-in `TimeoutError` does not catch the SDK's timeout class.
**Mine now**: no
```
