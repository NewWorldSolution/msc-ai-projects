# Rubric — nd901 · Course 3 · Project: Autonomous Knowledge Agent

> **PASTED VERBATIM. DO NOT PARAPHRASE, SUMMARISE, OR REWORD.**
> AGENTS.md §7.1 — the rubric-reviewer quotes these criteria exactly. Reviewers
> reject against this wording, so the wording is the asset.

**Source**: classroom → nd901 → *Course 3: Advanced Agentic AI Techniques* → **Project: Autonomous Knowledge Agent** → Rubric tab
**Captured**: 2026-09-07 — pasted as text, then verified against screenshots of the
rubric tab criterion by criterion. Not OCR'd.
**Status**: classroom shows "Overdue"; Burak has said to disregard it. Course 3 was 42 % complete at last check.

Implementation instructions: [`INSTRUCTIONS.md`](./INSTRUCTIONS.md).
Starter scaffold: [`starter/`](./starter/). Submission goes in `solution/`.
The other two nd901 projects have their own folders under `../` — never review
against a sibling's rubric.

> The classroom calls this project **"Autonomous Knowledge Agent"**; the scenario
> and the scaffold call the product **"UDA-Hub"** (a Universal Decision Agent),
> with **CultPass** as its first customer account. Same project.

**Structure note**: in the classroom each Submission Requirements cell is one
description sentence followed by a bulleted list. That structure is preserved
below with `•` markers. Only line breaks were added — no word was changed,
including the spelling `accross` in the memory criterion.

---

Use this project rubric to understand and assess the project criteria.

## Data Setup and Knowledge Base Preparation

| Criteria | Submission Requirements |
|---|---|
| Set up the database and knowledge base infrastructure | Successfully set up the database infrastructure and populate the knowledge base with comprehensive support articles.<br><br>• Successfully run the database management notebook to initialize the databases<br>• Database contains the required tables (Account, User, Ticket, TicketMetadata, TicketMessage, Knowledge)<br>• Knowledge base includes at least 10 additional support articles beyond the provided 4<br>• New articles cover different categories (technical issues, billing, account management, etc.)<br>• All database operations complete without errors<br>• Can demonstrate successful data retrieval from the database |

## Multi-Agent Architecture with LangGraph

| Criteria | Submission Requirements |
|---|---|
| Design and document multi-agent architecture | Design and document a comprehensive multi-agent architecture before implementation.<br><br>• Submit a detailed architecture design document in Markdown format<br>• Include a visual diagram showing the multi-agent architecture (can use ASCII art, Mermaid, or similar)<br>• Document the role and responsibilities of each agent in the system<br>• Explain the flow of information and decision-making between agents<br>• Describe how the system handles different types of inputs and expected outputs<br>• Architecture should be based on one of the standard patterns (Supervisor, Hierarchical, Network, etc.) |
| Implement the designed multi-agent architecture using LangGraph | Implement the designed multi-agent architecture using LangGraph with specialized agents for different tasks.<br><br>• Implementation matches the documented architecture design<br>• Project includes at least 4 specialized agents<br>• Each agent has a clearly defined role and responsibility as documented<br>• Agents are properly connected using LangGraph's graph structure<br>• Code demonstrates proper agent state management and message passing |
| Implement task routing and role assignment across agents | Implement intelligent task routing and role assignment across agents based on ticket characteristics.<br><br>• System can classify incoming tickets and route them to appropriate agents<br>• Routing logic considers ticket content and metadata (e.g. date, urgency, complexity...)<br>• At least one routing decision is made based on ticket classification<br>• Code includes routing logic that can be demonstrated with sample tickets<br>• Routing follows the architecture design principles |

## Knowledge Retrieval and Tool Usage

| Criteria | Submission Requirements |
|---|---|
| Implement knowledge-based response system with escalation logic | Implement a knowledge retrieval system that provides responses based on articles and escalates when no relevant knowledge is found.<br><br>• System retrieves relevant knowledge base articles based on ticket content<br>• All responses are based on the content of knowledge base articles<br>• System can demonstrate retrieval of appropriate articles for different ticket types<br>• Implements escalation logic when no relevant knowledge base article is found<br>• System includes confidence scoring to determine when to escalate<br>• Can demonstrate both successful knowledge retrieval and escalation scenarios |
| Implement support operation tools with database abstraction | Create and implement at least 2 tools that perform support operations with proper database abstraction.<br><br>• Implement at least 2 functional tools for support operations (e.g., account lookup, subscription management, refund processing)<br>• Tools abstract the interaction with the CultPass database<br>• Tools can be invoked by agents and return structured responses<br>• Tools include proper error handling and validation<br>• Can demonstrate tool usage with sample operations<br>• Tools are properly integrated into the agent workflow |

## Memory and State Management

| Criteria | Submission Requirements |
|---|---|
| Persist customer interaction history to enable personalized, context-aware support | Implement persistent memory to store and retrieve customer interaction history.<br><br>• System stores conversation history in a persistent database<br>• Can retrieve previous interactions for returning customers<br>• Uses historical context to provide personalized responses<br>• Demonstrates memory retrieval with sample customer interactions |
| Implement state, session and long-term memory in agent workflows | Implement different types of memory in agent workflows.<br><br>• Agents maintain state during multi-step interactions in one execution.<br>• Based on the appropriate scope (like thread_id or session_id), it's possible to inspect the workflow (e.g. messages, tool_usage)<br>• Short-term memory is used as context to keep conversation running during the same session<br>• Long-term memory is used to store resolved issues and customer preferences accross different sessions<br>• Memory is properly integrated into agent decision-making |

## Integration and Testing

| Criteria | Submission Requirements |
|---|---|
| Demonstrate end-to-end ticket processing workflow with proper logging | Demonstrate a complete end-to-end workflow for processing customer support tickets.<br><br>• System can process a ticket from initial submission to resolution/escalation<br>• Workflow includes classification, routing, knowledge retrieval, tool usage, resolution attempt, and final action<br>• Demonstrates the complete flow with sample tickets<br>• Includes proper error handling and edge cases<br>• System logs agent decisions, routing choices, tool usage and outcomes<br>• All logs are structured and searchable<br>• Shows both successful resolution and escalation scenarios<br>• Demonstrates tool integration in the workflow |

## Suggestions to Make Your Project Stand Out

- Advanced Knowledge Retrieval: Implement sophisticated semantic search using embeddings or vector databases for better article matching
- Multi-Channel Support: Extend the system to handle tickets from different channels (email, chat, social media)
- Sentiment Analysis: Add sentiment analysis to prioritize urgent or frustrated customer tickets
- A/B Testing Framework: Implement a framework to test different routing strategies and measure their effectiveness
- MCP: Create specialized tools for common support operations (refunds, account modifications, etc.) using FastMCP
