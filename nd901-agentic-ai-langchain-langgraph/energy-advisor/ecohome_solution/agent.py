import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools import TOOL_KIT

load_dotenv()


class AgentState(MessagesState):
    """Conversation state shared by the agent and tool nodes."""


class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o-mini"):
        """Create an Energy Advisor with an explicit LangGraph workflow."""

        self.system_message = SystemMessage(content=instructions)
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("VOCAREUM_API_KEY"),
        )

        # api: bind_tools gives the chat model the schemas it can request.
        self.llm_with_tools = llm.bind_tools(TOOL_KIT)

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", self._call_model)
        graph_builder.add_node("tools", ToolNode(TOOL_KIT))

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "agent")

        # api: compile turns the declared schema, nodes, and edges into a runnable graph.
        self.graph = graph_builder.compile()

    def _call_model(self, state: AgentState) -> dict[str, list[AIMessage]]:
        """Call the tool-enabled LLM with the system prompt and graph messages."""

        response = self.llm_with_tools.invoke(
            [self.system_message, *state["messages"]]
        )
        return {"messages": [response]}

    @staticmethod
    def _route_after_agent(
        state: AgentState,
    ) -> Literal["tools", "__end__"]:
        """Continue to tools when requested; otherwise finish the graph."""

        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    def invoke(
        self,
        question: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Ask the Energy Advisor a question about energy optimization.

        Args:
            question: The user's question about energy optimization.
            context: Optional trusted context such as location and current date.

        Returns:
            The graph result containing the complete message trace. If graph
            execution fails, the result also contains an ``error`` field.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        messages = []
        if context:
            messages.append(("system", context))
        messages.append(("user", question))

        try:
            return self.graph.invoke({"messages": messages})
        except Exception as exc:
            # why: Preserve a message-shaped result so evaluation can record a
            # failed agent run instead of terminating the whole test suite.
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not complete this request because the "
                            "Energy Advisor encountered an internal error."
                        )
                    )
                ],
                "error": str(exc),
            }

    def get_agent_tools(self) -> list[str]:
        """Get the names of tools available to the Energy Advisor."""

        return [tool.name for tool in TOOL_KIT]
