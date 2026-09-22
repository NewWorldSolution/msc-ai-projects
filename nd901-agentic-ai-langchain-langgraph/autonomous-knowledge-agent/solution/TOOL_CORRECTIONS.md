> **Superseded review status — 2026-09-17:** the confirmed foundation defects,
> including section 12's Intake findings, have now been corrected and regression
> tested. See [the actual before/after change log](FOUNDATION_FIXES_2026-09-17.md).
> Older observations and proposed snippets below are retained as study history.

# Tool corrections: exact replacements

> Internal study aid. Exclude this file from the final submission package.
> The persistent cleanup checklist is in the repository AGENTS.md, section 8.

This guide accompanies the function docstrings updated on 2026-09-14.

**Latest readiness result, 2026-09-14:** the experience date guard and both
memory test helpers are now corrected. The full existing suite reports
**81 passed, 12 skipped**, with three existing dependency deprecation warnings.
The write-failure fake provides get returning None, so the test reaches put and
verifies put_called. The recall-failure fake provides search raising the expected
RuntimeError. Test assertions were preserved; no production memory behavior was
changed to satisfy the tests.

All active tests pass, but a subsequent focused Intake review found three
untested correctness gaps. Address section 12 before treating Intake as ready
for agent integration. Add the missing agent/tool-binding and workflow assertions
as those components are built. The
12 skipped scenarios are still incomplete, and these checks do not establish
live-model, live-retrieval or end-to-end workflow correctness. Existing syntax,
module-import and model-facing tool-schema checks passed during the prior recheck.

**Original guide scope:** the exact before/after blocks below are retained for
reference. Most are now applied; do not paste the whole guide a second time.

The replacements follow the behavior already specified in
[ARCHITECTURE.md](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/design/ARCHITECTURE.md) and your review checklist.
These are expression-level corrections, not a new agent architecture.

## How to use this guide

1. Add the shared database classifier in section 4 before applying its imports.
2. In each named file, find the **Replace** block and use the corresponding
   **With** block. Preserve indentation. The blocks match the source after the
   docstring update; they do not depend on old line numbers.
3. For a block explicitly marked "both occurrences", replace both.
4. Add the regression test file from section 10, fix the existing memory test
   in section 8, then run the command in section 11.
5. Update the docstrings' known-limitations notes as each correction is applied,
   following section 9. Do not leave a docstring describing an already fixed bug.

The business tools continue to perform no business writes. The shared classifier
adds no package dependency: it uses standard-library SQLite and existing SQLAlchemy.

## 1. Make customer records JSON serializable

### [agentic/tools/customer_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/customer_tools.py)

Preserve timestamps and None values while allowing customer results to enter JSON handoffs and trace payloads.

**Replace:**

```python
            customer_record = model_to_dict(customer)
```

**With:**

```python
            customer_record = model_to_dict(customer)
            for field in ("created_at", "updated_at"):
                value = customer_record[field]
                # api: isoformat() converts a stored datetime into JSON-safe text.
                customer_record[field] = value.isoformat() if value is not None else None
```

## 2. Distinguish experience single and list modes

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

Use the trusted assignment mode, as reservations already do. It stays outside model-facing arguments.

**Replace:**

```python
    cultpass_engine = configurable.get("cultpass_engine")
```

**With:**

```python
    lookup_mode = configurable.get("lookup_mode")
    if lookup_mode not in ("single", "list"):
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "Trusted lookup mode must be single or list.",
            "error_code": "invalid_lookup_mode",
            "retryable": False,
        }

    cultpass_engine = configurable.get("cultpass_engine")
```

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

The ambiguity check follows the empty-result branch and excludes exact-ID lookups.

**Replace:**

```python
    return {
        **provenance,
        "outcome": "found",
        "records": records,
        "reason": "CultPass experiences found.",
        "retryable": False,
    }
```

**With:**

```python
    if not usable_experience_id and lookup_mode == "single" and len(records) > 1:
        return {
            **provenance,
            "outcome": "ambiguous",
            "records": records,
            "reason": "Multiple experiences matched; more identifying details are required.",
            "retryable": False,
        }

    return {
        **provenance,
        "outcome": "found",
        "records": records,
        "reason": "CultPass experiences found.",
        "retryable": False,
    }
```

## 3. Preserve Intake facts in the handoff

### [agentic/tools/handoff_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/handoff_tools.py)

Copy existing trusted Intake facts and identifiers without inventing a tool call. account_context follows CaseState's dictionary contract and must contain serializable current facts.

**Replace:**

```python
        "ticket_id": ticket_id,
        "turn_id": turn_id,
```

**With:**

```python
        "ticket_id": ticket_id,
        "account_id": state.get("account_id"),
        "customer_id": state.get("customer_id"),
        "account_context": dict(state.get("account_context", {})),
        "turn_id": turn_id,
```

## 4. Retry only recognized transient failures

Create **`agentic/tools/database_errors.py`** inside the solution with this complete content:

```python
"""Classify transient failures for the project's SQLite database engines."""

import sqlite3

from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError


def is_retryable_database_error(error: SQLAlchemyError) -> bool:
    """Recognize SQLite lock/busy failures and connection-pool timeouts.

    Args:
        error: SQLAlchemy exception caught at a database boundary.

    Returns:
        True only for recognized transient conditions. Permanent or unknown
        errors, including errors from unrecognized backends, return False.

    Notes:
        Performs no retries, logging or database access. Exact-message
        fallback supports test doubles without a SQLite error code.
    """
    if isinstance(error, SQLAlchemyTimeoutError):
        return True
    if not isinstance(error, OperationalError):
        return False

    original = error.orig
    if not isinstance(original, sqlite3.OperationalError):
        return False

    # api: sqlite_errorcode exposes SQLite's numeric result code on real errors.
    code = getattr(original, "sqlite_errorcode", None)
    if isinstance(code, int):
        # api: & 0xFF extracts the primary code from an extended SQLite code.
        return (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)

    # why: Exact known messages support test doubles without guessing from keywords.
    return str(original).lower() in {
        "database is locked",
        "database is busy",
        "database table is locked",
        "database schema is locked",
    }
```

The classifier is intentionally scoped to this project's SQLite engines. Unknown backend/provider failures are non-retryable until an explicit supported classification exists. It does not consume an attempt or retry an operation.

### [agentic/tools/subscription_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/subscription_tools.py)

Import the new helper shown at the beginning of this section.

**Replace:**

```python
from agentic.state import LookupResult
```

**With:**

```python
from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
```

### [agentic/tools/subscription_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/subscription_tools.py)

Replace both existing database-error handlers. Keep the subscription tool's earlier MultipleResultsFound handler.

**Replace:**

```python
    except OperationalError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass subscription query could not be completed.",
            "error_code": "cultpass_operational_error",
            "retryable": True,
        }
    except SQLAlchemyError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass subscription query failed.",
            "error_code": "cultpass_query_error",
            "retryable": False,
        }
```

**With:**

```python
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass subscription query could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }
```

### [agentic/tools/reservation_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/reservation_tools.py)

Import the new helper shown at the beginning of this section.

**Replace:**

```python
from agentic.state import LookupResult
```

**With:**

```python
from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
```

### [agentic/tools/reservation_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/reservation_tools.py)

Replace both existing database-error handlers. Keep the subscription tool's earlier MultipleResultsFound handler.

**Replace:**

```python
    except OperationalError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass reservation query could not be completed.",
            "error_code": "cultpass_operational_error",
            "retryable": True,
        }

    except SQLAlchemyError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass reservation query failed.",
            "error_code": "cultpass_query_error",
            "retryable": False,
        }
```

**With:**

```python
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass reservation query could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }
```

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

Import the new helper shown at the beginning of this section.

**Replace:**

```python
from agentic.state import LookupResult
```

**With:**

```python
from agentic.state import LookupResult
from agentic.tools.database_errors import is_retryable_database_error
```

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

Replace both existing database-error handlers. Keep the subscription tool's earlier MultipleResultsFound handler.

**Replace:**

```python
    except OperationalError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": (
                "The CultPass experience query could not be completed."
            ),
            "error_code": "cultpass_operational_error",
            "retryable": True,
        }

    except SQLAlchemyError:
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass experience query failed.",
            "error_code": "cultpass_query_error",
            "retryable": False,
        }
```

**With:**

```python
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            **provenance,
            "outcome": "technical_failure",
            "records": [],
            "reason": "The CultPass experience query could not be completed.",
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
        }
```

### [agentic/tools/knowledge_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/knowledge_tools.py)

Share the SQLite classification for authoritative article reads.

**Replace:**

```python
from data.models import udahub
```

**With:**

```python
from data.models import udahub
from agentic.tools.database_errors import is_retryable_database_error
```

### [agentic/tools/knowledge_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/knowledge_tools.py)

Unknown exception text no longer authorizes retries. Provider-specific errors stay non-retryable until their actual exception types/codes are classified; no live-service coverage is claimed.

**Replace:**

```python
    except OperationalError as error:
        error_text = str(error.orig).lower()
        retryable = error.connection_invalidated or any(
            marker in error_text for marker in _TRANSIENT_MARKERS
        )
        return technical_failure(
            (
                "transient_retrieval_error"
                if retryable
                else "retrieval_database_error"
            ),
            "The knowledge search could not be completed.",
            retryable=retryable,
        )
    except SQLAlchemyError:
        return technical_failure(
            "retrieval_database_error",
            "The knowledge search could not be completed.",
        )
    except (TimeoutError, ConnectionError):
        return technical_failure(
            "transient_retrieval_error",
            "The knowledge search could not be completed.",
            retryable=True,
        )
    except Exception as error:
        error_text = str(error).lower()
        retryable = any(
            marker in error_text for marker in _TRANSIENT_MARKERS
        )
        return technical_failure(
            (
                "transient_retrieval_error"
                if retryable
                else "knowledge_retrieval_error"
            ),
            "The knowledge search could not be completed.",
            retryable=retryable,
        )
```

**With:**

```python
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return technical_failure(
            "transient_retrieval_error" if retryable else "retrieval_database_error",
            "The knowledge search could not be completed.",
            retryable=retryable,
        )
    except (TimeoutError, ConnectionError):
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
```

Once the replacements are applied, `OperationalError` is unused in the three business lookup modules and Knowledge. Knowledge's `_TRANSIENT_MARKERS` is also unused. Those imports/constants can be removed; that cleanup is not required for correctness. Customer lookup and support persistence retain their existing narrower classifiers; the proposed shared helper is used at the boundaries shown above.

## 5. Reject malformed enum values safely

### [agentic/tools/memory_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/memory_tools.py)

Python short-circuits or: an invalid type is rejected before set membership can raise.

**Replace:**

```python
        or memory_type not in _MEMORY_TYPES
```

**With:**

```python
        or not isinstance(memory_type, str)
        or memory_type not in _MEMORY_TYPES
```

### [agentic/tools/ticket_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/ticket_tools.py)

Unhashable values now produce invalid_case_outcome instead of escaping validation.

**Replace:**

```python
    if case_outcome not in allowed_case_outcomes:
```

**With:**

```python
    if not isinstance(case_outcome, str) or case_outcome not in allowed_case_outcomes:
```

### [agentic/tools/ticket_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/ticket_tools.py)

Apply the same guard to handoff source validation.

**Replace:**

```python
            or handoff_source not in allowed_handoff_sources
```

**With:**

```python
            or not isinstance(handoff_source, str)
            or handoff_source not in allowed_handoff_sources
```

## 6. Make missing input and boundary dates controlled

### [agentic/tools/subscription_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/subscription_tools.py)

An omitted customer ID reaches the existing missing_input branch. Any supplied ID must still match trusted customer scope.

**Replace:**

```python
def lookup_subscription(customer_id: str, config: RunnableConfig) -> LookupResult:
```

**With:**

```python
def lookup_subscription(
    config: RunnableConfig, *, customer_id: str | None = None
) -> LookupResult:
```

### [agentic/tools/reservation_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/reservation_tools.py)

Named .invoke arguments stay compatible. The future Operations caller must still translate other Pydantic ValidationError cases.

**Replace:**

```python
def lookup_reservations(
    customer_id: str,
    config: RunnableConfig,
    *,
    reservation_id: str | None = None,
```

**With:**

```python
def lookup_reservations(
    config: RunnableConfig,
    *,
    customer_id: str | None = None,
    reservation_id: str | None = None,
```

### [agentic/tools/reservation_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/reservation_tools.py)

Keep this inside the existing try/except ValueError, before database access. Reject noncanonical formats and the date whose following day overflows.

**Replace:**

```python
            parsed_event_date = date.fromisoformat(event_date)
```

**With:**

```python
            parsed_event_date = date.fromisoformat(event_date)
            if parsed_event_date.isoformat() != event_date or parsed_event_date == date.max:
                raise ValueError("Event date must be a supported YYYY-MM-DD date.")
```

### [agentic/tools/reservation_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/reservation_tools.py)

Replace both occurrences to explain the accepted date range.

**Replace:**

```python
"reason": "Event date must use YYYY-MM-DD.",
```

**With:**

```python
"reason": "Event date must use YYYY-MM-DD and be earlier than 9999-12-31.",
```

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

Keep this inside the existing try/except ValueError, before database access. Reject noncanonical formats and the date whose following day overflows.

**Replace:**

```python
            parsed_event_date = date.fromisoformat(event_date)
```

**With:**

```python
            parsed_event_date = date.fromisoformat(event_date)
            if parsed_event_date.isoformat() != event_date or parsed_event_date == date.max:
                raise ValueError("Event date must be a supported YYYY-MM-DD date.")
```

### [agentic/tools/experience_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/experience_tools.py)

Replace both occurrences to explain the accepted date range.

**Replace:**

```python
"reason": "Event date must use YYYY-MM-DD.",
```

**With:**

```python
"reason": "Event date must use YYYY-MM-DD and be earlier than 9999-12-31.",
```

## 7. Validate and classify ticket loading

### [agentic/tools/ticket_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/ticket_tools.py)

Use the SQLite classifier from section 4.

**Replace:**

```python
from data.models import udahub
```

**With:**

```python
from data.models import udahub
from agentic.tools.database_errors import is_retryable_database_error
```

### [agentic/tools/ticket_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/ticket_tools.py)

Unusable IDs must never reach SQL. Intake must recognize the additional missing_input outcome.

**Replace:**

```python
    # api: get_session opens a session and guarantees rollback/cleanup on errors.
    try:
```

**With:**

```python
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        return {
            "outcome": "missing_input",
            "ticket_id": None,
            "error_code": "invalid_ticket_id",
            "retryable": False,
            "reason": "A non-empty ticket ID is required.",
        }

    # api: get_session opens a session and guarantees rollback/cleanup on errors.
    try:
```

### [agentic/tools/ticket_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/tools/ticket_tools.py)

Preserve the existing database_error outcome and add deterministic retry metadata.

**Replace:**

```python
    except SQLAlchemyError:
        return {
            "outcome": "database_error",
            "ticket_id": ticket_id,
            "reason": "The UDA-Hub database query could not be completed.",
        }
```

**With:**

```python
    except SQLAlchemyError as error:
        retryable = is_retryable_database_error(error)
        return {
            "outcome": "database_error",
            "ticket_id": ticket_id,
            "error_code": (
                "transient_database_error" if retryable else "database_error"
            ),
            "retryable": retryable,
            "reason": "The UDA-Hub database query could not be completed.",
        }
```

## 8. Make the memory failure test reach the write

### [tests/test_memory_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/tests/test_memory_tools.py)

Return an absent record from get so execution actually reaches put.

**Replace:**

```python
    class FailingStore:
        def put(self, namespace, key, value):
            raise RuntimeError("backend details must not escape")
```

**With:**

```python
    class FailingStore:
        def __init__(self):
            self.put_called = False

        def get(self, namespace, key):
            return None

        def put(self, namespace, key, value):
            self.put_called = True
            raise RuntimeError("backend details must not escape")
```

### [tests/test_memory_tools.py](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/tests/test_memory_tools.py)

Assert that the failing operation advertised by the test was attempted.

**Replace:**

```python
    result = save_customer_memory(
        "account-1",
        "customer-1",
        resolved_memory(),
        store=FailingStore(),
    )

    assert result == {
```

**With:**

```python
    store = FailingStore()
    result = save_customer_memory(
        "account-1",
        "customer-1",
        resolved_memory(),
        store=store,
    )

    assert store.put_called is True
    assert result == {
```

## 9. Docstrings after applying the corrections

The docstrings currently tell the truth about the uncorrected implementation.
Update these statements when their replacement code is in place:

| Function | Documentation update after applying the code |
|---|---|
| `lookup_customer` | Say that datetime columns become ISO strings and nullable timestamps remain None. Remove the note that JSON conversion is still needed. |
| `lookup_customer_tool` | Replace the retained-datetime note with "Found records contain JSON-safe customer values." |
| `lookup_experiences` | Add trusted `lookup_mode` (`single` or `list`) to configuration. Describe descriptive ambiguity in single mode and found lists in list mode. Remove the maximum-date and always-retryable limitations. |
| `lookup_reservations` | Say that an omitted customer ID returns missing_input. Remove the maximum-date and always-retryable limitations. |
| `lookup_subscription` | Say that an omitted customer ID returns missing_input. Describe recognized transient database failures rather than all OperationalError instances. |
| `assemble_fallback_handoff` | State that account_context and existing account/customer IDs are preserved. Copied values still need to be serializable before handoff assembly. |
| `save_customer_memory`, `save_ticket_outcome` | Remove the unhashable enum-value limitation after the type guards are applied. |
| `load_ticket_context` | Add missing_input to outcomes and describe pre-query ID validation plus error_code/retryable on database failures. |
| `search_knowledge` | Replace the generic substring note with explicit recognized-failure classification and non-retryable unknown exceptions. |

## 10. Add executable regression tests

Create **`tests/test_review_regressions.py`** inside the solution with the following
content. These tests cover the supplied replacements using in-memory databases
and a controlled scored store. The existing backend-write test is corrected
separately in section 8.

```python
"""Regression checks for the tool corrections; only in-memory resources are used."""

from contextlib import contextmanager
from datetime import datetime
import json
import sqlite3
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from agentic.tools import customer_tools as customer
from agentic.tools import experience_tools as experience
from agentic.tools import knowledge_tools as knowledge
from agentic.tools import reservation_tools as reservation
from agentic.tools import subscription_tools as subscription
from agentic.tools import ticket_tools as ticket
from agentic.tools.database_errors import is_retryable_database_error
from agentic.tools.handoff_tools import assemble_fallback_handoff
from agentic.tools.memory_tools import save_customer_memory
from data.models import cultpass, udahub
from utils import get_session


@pytest.fixture
def business_engine():
    engine = create_engine("sqlite://")
    cultpass.Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add_all([
            cultpass.User(user_id="c1", full_name="One", email="one@example.com"),
            cultpass.User(user_id="c2", full_name="Two", email="two@example.com"),
            cultpass.Subscription(subscription_id="s1", user_id="c1",
                                  status="active", tier="basic", monthly_quota=3),
        ])
        for number in (1, 2):
            session.add(cultpass.Experience(
                experience_id=f"e{number}", title="Concert", description="Stored",
                location="Warsaw", when=datetime(2026, 9, 15, 10),
                slots_available=5, is_premium=False,
            ))
        session.add_all([
            cultpass.Reservation(reservation_id="r1", user_id="c1",
                                 experience_id="e1", status="confirmed"),
            cultpass.Reservation(reservation_id="r2", user_id="c1",
                                 experience_id="e2", status="confirmed"),
            cultpass.Reservation(reservation_id="other", user_id="c2",
                                 experience_id="e1", status="confirmed"),
        ])
    yield engine
    engine.dispose()


def config(engine, mode="single"):
    return {"configurable": {
        "cultpass_engine": engine, "trusted_customer_id": "c1",
        "turn_id": "turn1", "issue_id": "issue1", "lookup_key": "lookup1",
        "attempt": 1, "lookup_mode": mode,
    }}


def test_customer_result_and_fallback_are_json_serializable(business_engine):
    result = customer.lookup_customer_tool.invoke(
        {"customer_id": "c1"}, config=config(business_engine),
    )
    assert isinstance(result["records"][0]["created_at"], str)
    json.dumps(result)
    handoff = assemble_fallback_handoff({
        "ticket_id": "ticket1", "turn_id": "turn1", "account_id": "account1",
        "customer_id": "c1", "messages": [HumanMessage(content="Help", id="turn1")],
        "tool_results": [result], "account_context": {"user_id": "c1", "is_blocked": True},
    })
    assert handoff["account_context"]["is_blocked"] is True
    assert handoff["customer_id"] == "c1"
    json.dumps(handoff)


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_unique_lookup_success_and_provenance(business_engine, tool, args):
    result = tool.invoke(args, config=config(business_engine))
    assert result["outcome"] == "found"
    assert len(result["records"]) == 1
    assert result["turn_id"] == "turn1"
    assert result["issue_id"] == "issue1"
    assert result["lookup_key"] == "lookup1"
    assert result["attempt"] == 1
    json.dumps(result)


@pytest.mark.parametrize("mode,expected", [("single", "ambiguous"), ("list", "found")])
def test_descriptive_cardinality_and_customer_scope(business_engine, mode, expected):
    experiences = experience.lookup_experiences.invoke(
        {"title": "Concert"}, config=config(business_engine, mode),
    )
    reservations = reservation.lookup_reservations.invoke(
        {"customer_id": "c1", "event_name": "Concert"}, config=config(business_engine, mode),
    )
    assert experiences["outcome"] == reservations["outcome"] == expected
    assert len(experiences["records"]) == len(reservations["records"]) == 2
    assert {row["user_id"] for row in reservations["records"]} == {"c1"}


def test_no_customer_required_for_experience_and_foreign_reservation_hidden(business_engine):
    trusted = config(business_engine)
    trusted["configurable"].pop("trusted_customer_id")
    assert experience.lookup_experiences.invoke(
        {"experience_id": "e1"}, config=trusted,
    )["outcome"] == "found"
    assert reservation.lookup_reservations.invoke(
        {"customer_id": "c1", "reservation_id": "other"}, config=config(business_engine),
    )["outcome"] == "not_found"


@pytest.mark.parametrize("tool,args", [
    (customer.lookup_customer_tool, {"customer_id": "c2"}),
    (subscription.lookup_subscription, {"customer_id": "c2"}),
    (reservation.lookup_reservations, {"customer_id": "c2"}),
])
def test_customer_switch_rejected_without_database_query(business_engine, tool, args):
    statements = []
    # api: before_cursor_execute observes SQL actually sent to the database.
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    result = tool.invoke(args, config=config(business_engine))
    assert result["error_code"] == "customer_scope_mismatch"
    assert statements == []


@pytest.mark.parametrize("tool", [
    subscription.lookup_subscription, reservation.lookup_reservations,
])
def test_omitted_customer_returns_missing_input(business_engine, tool):
    assert tool.invoke({}, config=config(business_engine))["outcome"] == "missing_input"


@pytest.mark.parametrize("tool,args", [
    (reservation.lookup_reservations, {"customer_id": "c1"}),
    (experience.lookup_experiences, {}),
])
@pytest.mark.parametrize("date_text", ["9999-12-31", "2026-02-30", "20260915"])
def test_invalid_date_rejected_before_sql(business_engine, tool, args, date_text):
    statements = []
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    result = tool.invoke({**args, "event_date": date_text}, config=config(business_engine))
    assert result["outcome"] == "missing_input"
    assert statements == []


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_missing_schema_is_nonretryable(tool, args):
    engine = create_engine("sqlite://")
    try:
        result = tool.invoke(args, config=config(engine))
        assert result["outcome"] == "technical_failure"
        assert result["retryable"] is False
    finally:
        engine.dispose()


@pytest.mark.parametrize("module,tool,args", [
    (subscription, subscription.lookup_subscription, {"customer_id": "c1"}),
    (reservation, reservation.lookup_reservations, {"customer_id": "c1", "reservation_id": "r1"}),
    (experience, experience.lookup_experiences, {"experience_id": "e1"}),
])
def test_locked_database_is_retryable(business_engine, module, tool, args):
    @contextmanager
    def locked(_engine):
        raise OperationalError("SELECT", {}, sqlite3.OperationalError("database is locked"))
        yield

    with patch.object(module, "get_session", locked):
        result = tool.invoke(args, config=config(business_engine))
    assert result["outcome"] == "technical_failure"
    assert result["retryable"] is True


def test_unknown_database_error_is_not_transient():
    assert is_retryable_database_error(SQLAlchemyError("timeout in configuration")) is False
    error = OperationalError("SELECT", {}, sqlite3.OperationalError("no such table: timeout"))
    assert is_retryable_database_error(error) is False


def test_bad_memory_type_and_ticket_outcome_are_controlled(business_engine):
    result = save_customer_memory("account1", "c1", {
        "memory_id": "m1", "memory_type": [], "content": "Stored",
        "ticket_id": "ticket1", "turn_id": "turn1", "outcome": "resolved",
        "source_refs": ["article1"],
    }, store=InMemoryStore())
    assert result["error_code"] == "invalid_memory_payload"
    result = ticket.save_ticket_outcome(
        "ticket1", {"turn_id": "turn1", "outcome": []}, business_engine,
    )
    assert result["error_code"] == "invalid_case_outcome"


def test_invalid_ticket_id_performs_no_sql(business_engine):
    statements = []
    event.listen(business_engine, "before_cursor_execute",
                 lambda *args: statements.append(args[2]))
    assert ticket.load_ticket_context("", business_engine)["outcome"] == "missing_input"
    assert statements == []


class ScoredStore:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates
        self.error = error

    def similarity_search_with_relevance_scores(self, query, *, k, filter):
        assert filter == {"account_id": "account1"}
        if self.error is not None:
            raise self.error
        return self.candidates


@pytest.fixture
def knowledge_config():
    engine = create_engine("sqlite://")
    udahub.Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add(udahub.Account(account_id="account1", account_name="CultPass"))
        session.add(udahub.Knowledge(article_id="article1", account_id="account1",
                                    title="Policy", content="Stored policy.", tags="policy"))
    yield {"configurable": {
        "udahub_engine": engine, "trusted_account_id": "account1",
        "turn_id": "turn1", "issue_id": "issue1",
        "retrieval_settings": {"threshold": 0.7, "candidate_count": 3,
                               "embedding_model": "test",
                               "score_convention": "higher_is_more_relevant"},
    }}
    engine.dispose()


@pytest.mark.parametrize("account,text,expected", [
    ("account1", "Stored policy.", None),
    ("account2", "Stored policy.", "invalid_source_metadata"),
    ("account1", "Invented policy.", "source_content_mismatch"),
])
def test_knowledge_sources(knowledge_config, account, text, expected):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([
        (Document(page_content=text, metadata={"article_id": "article1", "account_id": account}), 0.9),
    ])
    result = knowledge.search_knowledge("policy", knowledge_config)
    assert result["error_code"] == expected
    if expected is None:
        assert result["status"] == "completed"
        assert result["articles"][0]["content"] == "Stored policy."
        assert result["gate_records"][0]["gate_pass"] is True
    else:
        assert result["status"] == "technical_failure"
        assert result["articles"] == []


def test_empty_and_missing_knowledge_are_distinct(knowledge_config):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore([])
    empty = knowledge.search_knowledge("policy", knowledge_config)
    missing = knowledge.search_knowledge("", knowledge_config)
    assert empty["status"] == "completed"
    assert empty["gate_records"][0]["score"] is None
    assert missing["status"] == "missing_input"


@pytest.mark.parametrize("error,retryable", [
    (ValueError("timeout must be positive"), False),
    (TimeoutError("temporary timeout"), True),
])
def test_retrieval_retry_classification(knowledge_config, error, retryable):
    knowledge_config["configurable"]["knowledge_store"] = ScoredStore(error=error)
    result = knowledge.search_knowledge("policy", knowledge_config)
    assert result["status"] == "technical_failure"
    assert result["retryable"] is retryable


@pytest.mark.parametrize("tool,args", [
    (subscription.lookup_subscription, {"customer_id": "c2"}),
    (reservation.lookup_reservations, {"customer_id": "c2", "reservation_id": "absent"}),
    (experience.lookup_experiences, {"experience_id": "absent"}),
])
def test_completed_absence_is_not_a_failure(business_engine, tool, args):
    trusted = config(business_engine)
    trusted["configurable"]["trusted_customer_id"] = "c2"
    result = tool.invoke(args, config=trusted)
    assert result["outcome"] == "not_found"
    assert result["records"] == []
    assert result["retryable"] is False


def test_bad_memory_after_core_success_keeps_the_reply(monkeypatch):
    import agentic.persist as persist
    state = {
        "ticket_id": "ticket1", "turn_id": "turn1", "account_id": "account1",
        "customer_id": "c1", "outcome": "resolved",
        "messages": [HumanMessage(content="Help", id="turn1")],
        "approved_response": {"message_id": "reply1", "draft_id": "draft1",
                              "outcome": "resolved", "content": "Approved reply"},
        "review_result": {"draft_id": "draft1", "verdict": "approved"},
        "memory_updates": [{"memory_id": "m1", "memory_type": [], "content": "Stored",
                            "source_refs": ["article1"]}],
    }
    calls = []

    def saved(*args):
        calls.append(args)
        return {"saved": True, "approved_response": state["approved_response"]}

    monkeypatch.setattr(persist, "save_ticket_outcome", saved)
    result = persist.persist_case(
        state, {"configurable": {"thread_id": "ticket1"}},
        udahub_engine=object(), store=InMemoryStore(),
    )
    assert len(calls) == 1
    assert result["persistence_result"]["saved"] is True
    assert result["persistence_result"]["memory"]["status"] == "failed"
    assert result["messages"][0].content == "Approved reply"
```

## 11. Validation and remaining workflow work

From the solution directory, run:

```sh
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

The proposed implementation blocks, the corrected memory test and the new
regression test file were applied together in a temporary copy using the existing
project interpreter. Result: **74 passed, 12 skipped**. This is not the result for
your still-uncorrected implementation, whose existing suite has **37 passed,
12 skipped**.

The skipped scenario files contain TODO-only tests. Removing their skip markers
alone would produce meaningless passes because their functions have no assertions.
Keep them marked as pending until executable assertions exist.

### Integration is a separate implementation task

The `raise NotImplementedError(...)` bodies in
[Operations](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/agents/operations.py),
[Knowledge](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/agents/knowledge.py),
[Intake](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/intake.py), and
[build_workflow](/Users/burakkilic/msc-ai/curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/solution/agentic/workflow.py) cannot be repaired by importing the tools
or deleting the exception. Their complete behavior remains to be implemented.

Use their existing architecture/docstring contracts as the implementation checklist:

- Operations binds the four permitted tools, validates model tool names/arguments,
  supplies trusted engines/customer scope/turn/issue/lookup key/attempt, and invokes
  the selected tool. It must catch Pydantic ValidationError, since optional
  customer arguments only solve the omitted-input case. It must supply the trusted
  single/list lookup_mode to both reservations and experiences.
- Operations constructs normalized lookup keys that include trusted scope and
  matching mode, reuses only completed same-turn results, and permits one
  deterministic transient retry per scoped lookup.
- Knowledge calls search_knowledge, retains source/gate provenance, assesses
  issue coverage, and enforces independent technical and targeted-search budgets.
- Intake derives trusted account/customer identity from the support ticket,
  handles missing_input and classified database failures, converts customer facts
  before placing them in account_context, and handles memory recall failure
  separately from empty memory.
- The factory creates and compiles the agreed graph, binds resources and limits,
  and copies rather than mutates caller configuration.
- Escalation and Supervisor still need their existing routing/fallback behavior;
  unit tests of Persist do not establish a functioning full workflow.
- Replace skipped scenario bodies with real assertions for these behaviors.
  No live model, embedding service or Chroma retrieval-quality claim is made here.

No complete replacement agent/workflow body is presented as a finished fix in this
guide: those are the unfinished implementation work identified by the review.


## 12. Intake review — 2026-09-14

The seven existing Intake tests pass. The full project suite remains **81 passed,
12 skipped**, with three existing dependency deprecation warnings. Intake source
and test syntax checks pass. Additional in-memory probes expose these gaps; the
implementation and tests were not changed during this review.

### P1 — Reset controls only for a new turn

Location: `agentic/intake.py:65` (especially counters at 75–89).
`prepare_case` resets classification, technical/review budgets and terminal state
on every invocation, even when the incoming human-message ID equals the existing
state turn_id. A probe with both review counters at 2 and messaging_disabled=True
returned counters at 0, messaging_disabled=False and handoff=None for the same turn.

Reset turn-scoped fields only when entering a new human-message ID. Preserve the
current turn's spent allowances and terminal decisions on reentry. Add a regression
that merges a prepared turn, consumes its budgets, re-enters Intake with the same
message ID, and verifies that counters and terminal state remain unchanged. The
existing same-turn test checks only duplicate trace suppression.

### P1 — Reject invalid input without exposing previous success

Location: `agentic/intake.py:51`.
The invalid-current-message branch returns only preparation_result. A minimal
StateGraph using the actual CaseState reducers retains a previous approved_response,
persistence_result={saved: True}, outcome=resolved and messaging_disabled=False
alongside the new failed preparation result. This is stale state, not proof that
this invocation produced or saved a response.

Make invalid input an explicit application failure before routing onward, or ensure
the controlled failure delta invalidates terminal output/save indicators and cannot
release the previous response. Add a test starting from a previously completed
turn and assert the merged result cannot be mistaken for current success.

### P2 — Execute the allowed transient lookup retry

Locations: `agentic/intake.py:118` and `agentic/intake.py:186`.
The ticket and customer helpers perform no retries; Intake calls each once and
copies retryable=True into a failed preparation result. For each boundary, a fake
that fails transiently on the first call and succeeds on the second was called
only once. The agreed architecture allows one retry of the failed operation.

Implement bounded retry handling in Intake for recognized transient ticket/customer
lookup failures, with stable current-turn attempt accounting and trace provenance.
Do not retry not_found, validation failures or unknown/non-retryable errors. A
second failure must return a controlled terminal preparation failure. Add tests for
transient-then-success, two transient failures, non-retryable failure, and same-turn
reentry without a renewed allowance.

### Checks that passed

The existing tests verify ticket-derived customer scope, rejection of a different
customer record, customer-only facts, new-turn reset behavior, preserved conversation
and identity-correction history, historical memory separated from current facts,
controlled recall failure, explicit missing-ticket/customer outcomes and duplicate
trace suppression. These remain useful checks; they do not cover the gaps above.
