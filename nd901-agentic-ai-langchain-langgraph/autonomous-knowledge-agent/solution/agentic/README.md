# UDA-Hub application scaffold

This is the L1 scaffold for [Part A of the architecture](design/ARCHITECTURE.md).
Interfaces, schemas, reducers and TODOs are aligned; node/tool/factory bodies still
raise NotImplementedError. Scenario tests are skipped placeholders, not pass claims.

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

Start ticket_tools.py with load_ticket_context using notebook 02's supplied
utils.get_session and mapped SQLAlchemy models. Keep database helpers as plain
functions and all unfinished bodies as L1 TODOs until implementing them.

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

## Implementation still required

Fill node/tool/factory bodies and the eight scenario groups using the agreed
contracts. Implement searchable trace_events and atomic support writes. Expand the
article set and evaluate retrieval thresholds before a live demonstration.

Notebook 03 is a setup template. Its existing chat_interface still needs the
architecture 5.8 presentation guard: display only the explicit current approved
response after successful graph completion/persistence, never messages[-1].
Persist owns the conversation delta; the application performs no second state write.

Configure credentials locally using OPENAI_API_KEY and OPENAI_BASE_URL; explicitly
map a Vocareum-named key at setup if needed. No model or database is called by
importing these scaffolds, and the unfinished notebook is not ready to run end to end.
