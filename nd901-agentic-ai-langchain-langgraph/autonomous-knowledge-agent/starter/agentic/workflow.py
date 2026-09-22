from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
)
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# TODO: Import agents from agentic/agents
# TODO: Import tools from agentic/tools
# TODO: Modify orchestrator to orchestrate your agents. Now it's just sample code!

orchestrator = create_react_agent(
    model = ChatOpenAI(model="gpt-4o-mini"),
    checkpointer = MemorySaver(),
    tools = [
        Tool(
            name="get_length", 
            func=lambda x: len(x), 
            description="Returns the length of a string."
        )
    ],
    prompt = SystemMessage(
        content=(
            "You're an engaging and helpful assistant. " 
        )
    ),
)