"""L1 workflow factory contract; importing this module creates no resources."""

from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from sqlalchemy import Engine
from math import isfinite
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agentic.agents.supervisor import supervise_case
from agentic.state import CaseState

class AgentModels(TypedDict):
    """Caller-supplied models; roles may share the same configured instance."""

    supervisor: BaseChatModel
    classifier: BaseChatModel
    knowledge: BaseChatModel
    operations: BaseChatModel
    response: BaseChatModel
    escalation: BaseChatModel


class RetrievalSettings(TypedDict):
    """Values selected and recorded from the practical retrieval evaluation."""

    threshold: float
    candidate_count: int
    embedding_model: str
    score_convention: str


class WorkflowLimits(TypedDict):
    """Finite limits chosen from demo traces; no unvalidated default is supplied."""

    max_graph_steps: int
    max_model_calls: int
    call_timeout_seconds: float

def validate_workflow_limits(limits: WorkflowLimits) -> WorkflowLimits:
    """Validate limits and return a separeate normalized dictionary."""

    if not isinstance(limits, dict):
        raise ValueError("Limits must be provided as a dictionary.")
    for name in ("max_graph_steps", "max_model_calls"):
        value = limits.get(name)

        if type(value) is not int or value <= 0:
            raise ValueError(f"Limit '{name}' must be a positive integer.")
    timeout = limits.get("call_timeout_seconds")

    if type(timeout) not in (int,float):
        raise ValueError("Limit 'call_timeout_seconds' must be a positive finite number.")
    try:
        timeout = float(timeout)
    except OverflowError:
        raise ValueError("call_timeout_seconds is too large") from None

    # api: isfinite rejects infinity and NaN.
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "call_timeout_seconds must be a positive finite number"
        )

    return {
        "max_graph_steps": limits["max_graph_steps"],
        "max_model_calls": limits["max_model_calls"],
        "call_timeout_seconds": timeout,
    }
def build_workflow(
    *,
    cultpass_engine: Engine,
    udahub_engine: Engine,
    models: AgentModels,
    knowledge_store: VectorStore,
    retrieval_settings: RetrievalSettings,
    limits: WorkflowLimits,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    """Bind prepared resources and compile the explicit graph; architecture 5.8.

    TODO: Validate retrieval settings, finite limits and the six model bindings.
    TODO: Follow the explicit lesson pattern: StateGraph(CaseState), add_node
    for Intake, Supervisor, five specialists and Persist, then add_edge for the
    eight fixed edges (START/intake, intake/supervisor, five specialist returns,
    persist/END). Import StateGraph, START and END from langgraph.graph.
    Bind model keyword arguments through wrappers at construction.
    TODO: Create node-local tool-call config from caller RunnableConfig with trusted
    engines, knowledge_store, retrieval_settings and limits; preserve thread_id
    without mutating the caller configuration. Resource
    objects remain in runtime configuration/closures, never checkpointed state.
    TODO: Connect START -> intake -> supervisor. Invalid tickets fail explicitly.
    TODO: Supervisor returns Command, including classifier as a destination.
    Never add an unconditional outgoing edge alongside its selected route.
    TODO: Classifier, Knowledge, Operations, Response and Escalation each have
    a fixed return to Supervisor. Supervisor alone selects Persist; Persist ends.
    TODO: Use deterministic entry/terminal guards, including fallback -> Supervisor
    -> Persist without another model call. Keep the existing retry allowances.
    TODO: Enforce graph/model-call limits and timeouts; preserve the handoff path
    when a limit ends normal work, rather than only surfacing a recursion error.
    TODO: Use graph.compile(checkpointer=checkpointer, store=store); named store
    parameters are injected. Inspect the compiled get_graph().draw_mermaid()
    against architecture section 3. A returned save failure still follows
    persist -> END; the application checks persistence_result before presentation.
    TODO: Return the compiled graph with the configured run limits.

    Input is messages plus configurable.thread_id (an existing ticket ID).
    Caller owns connections, credentials and resource cleanup. The prepared
    VectorStore is the article retrieval resource and must provide scores as well
    as source documents; a plain document-only retriever does not meet that contract.
    """

    limits = validate_workflow_limits(limits)

    def supervisor_node(
            state: CaseState,
            config: RunnableConfig,
    ) -> Command:
        """Supply trusted settings and models to Supervisor."""

        supervisor_config: RunnableConfig = {
            **config,
            "configurable": {
                **(config.get("configurable") or {}),
                "limits": limits,
            },
        }

        return supervise_case(
            state,
            supervisor_config,
            model = models["supervisor"],
            classifier_model = models["classifier"],
        )

    raise NotImplementedError("L1 scaffold: implement from architecture section 5.8.")