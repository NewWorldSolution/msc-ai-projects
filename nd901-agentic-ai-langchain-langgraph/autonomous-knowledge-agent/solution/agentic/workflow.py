"""Compile the explicit support graph; importing this module creates no resources."""

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
from agentic.agents.classifier import classify_ticket
from agentic.agents.knowledge import retrieve_knowledge
from agentic.agents.operations import handle_operations
from agentic.agents.response import draft_response
from agentic.agents.escalation import prepare_escalation
from agentic.intake import prepare_case
from agentic.persist import persist_case
from agentic.model_budget import build_model_call_guard
from agentic.state import CaseState
from copy import deepcopy
from langgraph.graph import StateGraph, START, END
from agentic.tools.handoff_tools import assemble_fallback_handoff


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
    """Validate supplied resources and compile the support workflow.

    Args:
        cultpass_engine: Prepared engine for read-only customer and business
            lookups.
        udahub_engine: Prepared engine for support tickets, messages and logs.
        models: Chat-model bindings for Supervisor, Classifier, Knowledge,
            Operations, Response and Escalation.
        knowledge_store: Prepared vector store exposing scored article retrieval.
        retrieval_settings: Relevance threshold, candidate count, embedding-model
            identifier and higher-is-more-relevant score convention.
        limits: Graph-step allowance, model-call allowance and call timeout.
            The graph-step allowance must permit at least three nodes.
        checkpointer: Saver for active ticket conversation and workflow state.
        store: Long-term memory store injected into Intake and Persist.

    Returns:
        A compiled CaseState graph with the supplied checkpointer and store,
        default recursion_limit of max_graph_steps + 1, and max_concurrency
        of 1. Explicit invocation settings can override these defaults.

    Raises:
        ValueError: A supplied resource, model binding, retrieval setting or
            execution limit fails construction-time validation.

    Notes:
        Invoke with a HumanMessage carrying a stable message ID and
        configurable.thread_id identifying an existing ticket.

        Intake enters Supervisor. Registered specialists return to Supervisor,
        which selects the next destination through Command. Persist ends the
        automated turn.

        With insufficient steps for another specialist round, Supervisor
        prepares a deterministic handoff and routes to Persist without a
        model call.

        Trusted resources and copied settings override matching values in
        per-run configurable settings. Resource objects remain outside
        checkpointed case state. The caller owns resource creation and cleanup.

        The application must check persistence_result before presenting the
        explicit approved_response to the customer.
    """

    limits = validate_workflow_limits(limits)
    if limits["max_graph_steps"] < 3:
        raise ValueError("max_graph_steps must allow at least 3 nodes.")
    # Validate the supplied runtime resources.
    # Both database engines must be supplied by the application.
    for name, engine in (
        ("cultpass_engine", cultpass_engine),
        ("udahub_engine", udahub_engine),
    ):
        if not isinstance(engine, Engine):
            raise ValueError(
                f"{name} must be a SQLAlchemy Engine."
            )

    # Every agent needs an explicit model binding.
    if not isinstance(models, dict):
        raise ValueError("models must be a dictionary.")

    required_roles = (
        "supervisor",
        "classifier",
        "knowledge",
        "operations",
        "response",
        "escalation",
    )

    for role in required_roles:
        if not isinstance(models.get(role), BaseChatModel):
            raise ValueError(
                f"Missing or invalid chat model for '{role}'."
            )

    # Copy the mapping so changing the caller's dictionary later
    # cannot replace the models bound to this workflow.
    # The model objects themselves remain shared.
    models = dict(models)

    # Knowledge requires source documents together with relevance scores.
    if not isinstance(knowledge_store, VectorStore):
        raise ValueError(
            "knowledge_store must be a LangChain VectorStore."
        )

    if not callable(
        getattr(
            knowledge_store,
            "similarity_search_with_relevance_scores",
            None,
        )
    ):
        raise ValueError(
            "knowledge_store must provide scored retrieval."
        )

    # Checkpointing preserves the active ticket conversation.
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise ValueError(
            "checkpointer must be a LangGraph checkpoint saver."
        )

    # The store holds permitted memory across conversations.
    if not isinstance(store, BaseStore):
        raise ValueError(
            "store must be a LangGraph BaseStore."
        )

    # Validate and copy retrieval settings.

    if not isinstance(retrieval_settings, dict):
        raise ValueError("retrieval_settings must be a dictionary.")

    threshold = retrieval_settings.get("threshold")

    # Exact types exclude bool: True must not become a threshold of 1.
    if type(threshold) not in (int, float):
        raise ValueError("threshold must be numeric.")

    try:
        threshold = float(threshold)
    except OverflowError:
        raise ValueError("threshold is too large.") from None

    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1.")

    candidate_count = retrieval_settings.get("candidate_count")
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("candidate_count must be a positive integer.")

    embedding_model = retrieval_settings.get("embedding_model")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string.")

    score_convention = retrieval_settings.get("score_convention")
    if score_convention != "higher_is_more_relevant":
        raise ValueError(
            "score_convention must be 'higher_is_more_relevant'."
        )

    # Keep a separate dictionary containing only validated settings.
    retrieval_settings = {
        "threshold": threshold,
        "candidate_count": candidate_count,
        "embedding_model": embedding_model,
        "score_convention": score_convention,
    }

    # Build trusted configuration for each node invocation

    def build_node_config(config: RunnableConfig) -> RunnableConfig:
        """Combine the current run's configuration with trusted resources."""

        if not isinstance(config, dict):
            raise ValueError("Node configuration must be a dictionary.")

        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            raise ValueError("Configurable section must be a dictionary.")

        return {
            **config,
            "configurable": {
                **configurable,
                "cultpass_engine": cultpass_engine,
                "udahub_engine": udahub_engine,
                "knowledge_store": knowledge_store,
                "retrieval_settings": dict(retrieval_settings),
                "limits": dict(limits),
            },
        }


    # 4. BIND THE GRAPH NODES TO TRUSTED RESOURCES AND MODELS.
    def intake_node(
        state: CaseState, config: RunnableConfig, *, store: BaseStore
    ) -> CaseState:
        """Prepare the case using trusted engines and the injected memory store."""
        # api: LangGraph injects the compiled store into the named store parameter.
        return prepare_case(state, build_node_config(config), store=store)

    def supervisor_node(
        state: CaseState,
        config: RunnableConfig,
    ) -> Command:
        """Route normally, or prepare a handoff before steps run out."""
        supervisor_config = build_node_config(config)

        # Validate ticket scope before allowing the early handoff route.
        preparation = state.get("preparation_result") or {}
        thread_id = supervisor_config["configurable"].get("thread_id")

        if (
            preparation.get("ticket_lookup_outcome") != "success"
            or preparation.get("error_code") == "invalid_ticket_context"
            or not isinstance(thread_id, str)
            or not thread_id.strip()
            or state.get("ticket_id") != thread_id
            or preparation.get("outcome")
            not in ("ready", "customer_not_found", "failed")
        ):
            raise RuntimeError(
                preparation.get("reason")
                or "Case preparation did not establish a valid ticket."
            )

        remaining = state.get("remaining_steps")
        if type(remaining) is not int or remaining < 2:
            raise RuntimeError(
                "The graph must retain enough steps for Supervisor and Persist."
            )

        # why: Another specialist assignment needs four steps:
        # this Supervisor, specialist, returning Supervisor, then Persist.
        if remaining <= 3:
            update: CaseState = {
                "outcome": "escalated",
                "escalation_reason": "Workflow step allowance exhausted.",
                "messaging_disabled": True,
                "acknowledgement_permitted": False,
                "approved_response": None,
                "assignment": None,
            }

            # Reuse the deterministic helper without spending a model call.
            update["handoff"] = assemble_fallback_handoff({**state, **update})

            event_id = f"{state['turn_id']}:workflow_step_limit"
            if not any(
                event.get("event_id") == event_id
                for event in state.get("trace", [])
            ):
                update["trace"] = [{
                    "event_id": event_id,
                    "turn_id": state["turn_id"],
                    "issue_id": None,
                    "node": "supervisor",
                    "event_type": "workflow_step_limit",
                    "outcome": "escalated",
                    "reason": update["escalation_reason"],
                    "payload": {
                        "remaining_steps": remaining,
                        "max_graph_steps": limits["max_graph_steps"],
                    },
                }]

            return Command(update=update, goto="persist")

        return supervise_case(
            state,
            supervisor_config,
            model=models["supervisor"],
        )

    def classifier_node(state: CaseState, config: RunnableConfig) -> CaseState:
        """Classify an assigned case and return its shared model-call count."""
        node_config = build_node_config(config)
        # why: Reservations change this copy, never the graph's input state.
        working_case = deepcopy(state)
        reserve_model_call = build_model_call_guard(
            working_case,
            max_model_calls=node_config["configurable"]["limits"]["max_model_calls"],
        )
        update = classify_ticket(
            working_case,
            model=models["classifier"],
            reserve_model_call=reserve_model_call,
            call_timeout_seconds=node_config["configurable"]["limits"]["call_timeout_seconds"],
        )
        return {**update, "model_calls": working_case["model_calls"]}

    def knowledge_node(state: CaseState, config: RunnableConfig) -> CaseState:
        """Retrieve and assess evidence with the bound Knowledge model."""
        return retrieve_knowledge(
            state, build_node_config(config), model=models["knowledge"]
        )

    def operations_node(state: CaseState, config: RunnableConfig) -> CaseState:
        """Obtain read-only business facts with the bound Operations model."""
        return handle_operations(
            state, build_node_config(config), model=models["operations"]
        )

    def response_node(state: CaseState, config: RunnableConfig) -> CaseState:
        """Draft a response for Supervisor review with the bound Response model."""
        return draft_response(
            state, build_node_config(config), model=models["response"]
        )

    def escalation_node(state: CaseState, config: RunnableConfig) -> CaseState:
        """Prepare a factual handoff with the bound Escalation model."""
        return prepare_escalation(
            state, build_node_config(config), model=models["escalation"]
        )

    def persist_node(
        state: CaseState, config: RunnableConfig, *, store: BaseStore
    ) -> CaseState:
        """Save Supervisor's final decision using trusted support storage."""
        return persist_case(
            state, build_node_config(config),
            udahub_engine=udahub_engine, store=store,
        )

    # Create the graph and register its node
    #api: StateGraph defines a workflow using CaseState and its reducers.

    graph= StateGraph(CaseState)

    graph.add_node("intake", intake_node)

    graph.add_node("supervisor", supervisor_node,
                   destinations = ("classifier",
                                    "knowledge",
                                    "operations",
                                    "response",
                                    "escalation",
                                    "persist"),
                                    )

    graph.add_node("classifier", classifier_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("operations", operations_node)
    graph.add_node("response", response_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("persist", persist_node)

    # Connect the fixed ages

    graph.add_edge(START, "intake")

    graph.add_edge("intake", "supervisor")

    graph.add_edge ("classifier", "supervisor")
    graph.add_edge ("knowledge", "supervisor")
    graph.add_edge ("operations", "supervisor")
    graph.add_edge ("response", "supervisor")
    graph.add_edge ("escalation", "supervisor")

    graph.add_edge ("persist", END)

    compiled = graph.compile(checkpointer = checkpointer, store = store)

    return compiled.with_config({
        "recursion_limit": limits["max_graph_steps"] + 1,
        "max_concurrency": 1
    })
