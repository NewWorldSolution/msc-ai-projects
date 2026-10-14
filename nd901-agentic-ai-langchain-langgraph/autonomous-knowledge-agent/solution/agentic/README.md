# UDA-Hub application

This package implements [Part A of the architecture](design/ARCHITECTURE.md): the
state contract, the six agents, the read-only tools, the workflow factory and the
persistence helpers, with a passing local test suite.

## Workflow and responsibilities

- Intake prepares the existing ticket and customer-only facts, then returns to Supervisor.
- Supervisor assigns Classifier, delegates further work and reviews exact drafts.
- Classifier, Knowledge, Operations, Response and Escalation all return to Supervisor.
- Specialists select their permitted tools; Operations is read-only.
- Response prepares complete channel-aware customer wording.
- Escalation prepares a generated or deterministic handoff, never customer wording.
- Supervisor alone selects Persist, with an approved reply or a handoff without one.
- Persist saves support records, returns the approved-message delta and writes eligible memory.

## Lesson patterns to follow

[Architecture 5.9](design/ARCHITECTURE.md#59-implementation-style-from-the-lessons)
maps the actual notebooks to this scaffold. LangGraph connects nodes through
StateGraph, explicit edges and Supervisor's Command, with reducers and SQLite
checkpoint/store persistence. Supervisor routes in code and calls a model only
for draft review. LangChain supplies ChatPromptTemplate, structured model results,
Operations tool binding and scored article retrieval inside the nodes.

ticket_tools.py starts with load_ticket_context using notebook 02's supplied
utils.get_session and mapped SQLAlchemy models. Database helpers are plain functions.

## State and update contracts

Call with a stable-ID HumanMessage and configurable.thread_id for an existing
UDA-Hub ticket. Intake derives account/customer identity and current turn_id.

messages uses add_messages: repeated stable IDs merge instead of adding duplicates.
trace and tool_results use add and receive only new entries. retrieved_articles
replaces the list, so Knowledge returns the complete current-turn merged evidence.
There is no second evidence-pack store or global relevance score.

original_review_attempts and acknowledgement_review_attempts are separate per-turn
counters. New evidence never resets them. classification_status distinguishes
pending, completed, failed and skipped, preventing repeated classification loops.
Identity-correction history remains ticket-scoped across turns. Old trace/tool
results retain turn IDs and are not reused as current business evidence.

## Tools and resources

Only tools/ helpers access CultPass and UDA-Hub business/support tables. Customer
lookup accepts exactly one ID or stored email; subscription lookup is separate.
Reservation and experience tools support the agreed unique/descriptive lookup modes.
The deterministic fallback handoff helper is shared by Escalation and Persist.

Application setup passes both engines, AgentModels, a prepared scored article
knowledge_store, RetrievalSettings, WorkflowLimits, the checkpointer and store to
build_workflow. Values for retrieval thresholds and execution limits must be chosen
from the planned demonstrations. Resources are never constructed at module import.
Model objects can be shared across roles; this interface does not require six clients.

The factory binds models and prepares local tool-call config with trusted resources,
preserving the caller's thread_id. Nodes receive the compiled long-term store by
its named store parameter. Resource objects do not belong in CaseState.

## Status and running

Node, tool and factory bodies are implemented; `trace_events` records every routing
decision, retrieval gate, tool call, review and final outcome, and `utils.routing_path`
renders a turn's routing events as one readable line. Notebook 02 seeds 14
articles and seven demo tickets; notebook 03 builds the Knowledge store, compiles the
graph and processes every open ticket with `utils.batch_interface`, presenting only the
approved reply or an explicit handoff report after a successful save.

Configure credentials locally from `.env.example` (OPENAI_API_KEY or VOCAREUM_API_KEY,
and OPENAI_BASE_URL). Run notebooks 01, 02 and 03 in order from the `solution` folder.
No model or database is called by importing this package.
