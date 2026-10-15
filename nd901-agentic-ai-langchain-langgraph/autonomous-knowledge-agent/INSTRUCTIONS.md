# Project Instructions: nd901, Course 3, Autonomous Knowledge Agent (UDA-Hub)

**Source**: classroom, nd901, *Course 3: Advanced Agentic AI Techniques*, **Project: Autonomous Knowledge Agent**.
Copied on 2026-09-07, the day I inspected the starter.
**Rubric**: [`rubric.md`](./rubric.md), copied word for word.

> The classroom calls this project **"Autonomous Knowledge Agent"**; the scenario
> calls the product **"UDA-Hub"**, a Universal Decision Agent, with **CultPass**
> as its first paying account. Same project.
> Udacity supplied a numbered instruction list here (Design / Setup / Agentic
> Workflow / Run), reproduced word for word in Part 1. The work breakdown in Part 2
> is mine: I derived it from those instructions, the `# TODO:` markers in the
> starter, and the rubric.

---

# Part 1: Udacity's text, word for word

## Project Scenario

You've joined a fast-growing AI startup building the next frontier in customer support automation.
Your team is responsible for building UDA-Hub, a Universal Decision Agent designed to plug into existing customer support systems (Zendesk, Intercom, Freshdesk, internal CRMs) and intelligently resolve tickets. But this isn't just another FAQ bot.
The goal? Build an agentic system that reads, reasons, routes, and resolves, acting as the operational brain behind support teams.
You'll need to design an agent system that can:

* Understand customer tickets across channels
* Decide which agent or tool should handle each case
* Retrieve or infer answers when possible
* Escalate or summarize issues when necessary
* Learn from interactions by updating long-term memory

Your agent should not only automate, it should decide how to automate!

## Project Introduction

In this project, you will develop UDA-Hub, an intelligent, multi-agent decision suite capable of resolving customer support tickets across multiple platforms.

Key Capabilities:

1. Multi-Agent Architecture with LangGraph Design and orchestrate specialized agents (e.g., Supervisor, Classifier, Resolver, Escalation…).
2. Input Handling Accept incoming support tickets in natural language with metadata (e.g., platform, urgency, history).
3. Decision Routing and Resolution
   * Route tickets to the right agent based on classification
   * Retrieve relevant knowledge via RAG if needed
   * Resolve or escalate based on confidence and context
4. Memory Integration
   * Maintain state during steps of the execution
   * Short-term memory is used as context to keep conversation running during the same session
   * Store and recall long-term memory for preferences, as an example

## Project Summary

Inputs:

* Incoming support ticket (text + metadata)
* Internal knowledge base (FAQ, previous tickets)
* Optional internal tool (e.g., refund)
* Memory store (for prior conversations and resolutions)

Deliverables:
A LangGraph-powered multi-agent system that:

* Understands tickets
* Routes to correct agent with tools
* Resolves or escalates based on decision logic
* Uses memory appropriately

## Project Instructions

Your starter folder looks like the following structure:

```
starter/
├── agentic/
│   ├── agents/
│   ├── design/
│   ├── tools/
│   └── workflow.py
├── data/
│   ├── core/
│   ├── external/
│   └── models/
├── .env
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
└── utils.py
```

### Design

1. Start by designing the solution. Your implementation will follow it.
2. Place all the documentation and diagrams about the design of your agentic system inside `agentic/design`

### Setup

1. Run notebook `01_external_db_setup.ipynb` in order to have all the data related to the account Cultpass. It's the first customer that has purchased Uda-hub
2. Run notebook `02_core_db_setup.ipynb` in order to have all the data related to Uda-hub application, including the files "received" from Cultpass like `cultpass_articles.jsonl`
3. You need to expand cultpass_articles form 4 to at least 14 articles. Make sure you have diverse topics for your agentic system.

### Agentic Workflow

1. Develop your agents inside `agentic/agents` and your tools inside `agentic/tools` . This will help you with modularity.
2. Develop your workflow orchestration in the file `workflow.py` . There's already a sample for you, but donot use it, create the graph from scratch. Do not use the prebuilt workflow.
3. When developing tools that abstract the database both for retrieval or for actions, please mind the relative/absolute paths. I strongly recommend you to use something like MCP servers for the tools.
4. If you're using RAG for retrieval, make sure you have documented how it works.
5. For short-term memory (session), you can use thread_id. For long-term memory, you're free to use semantic search.

### Run

1. There's a `chat_interface()` function inside `utils.py`. It's just a simple while True block. Starter code imports this inside the notebook. Feel free to improve it!
2. You're not forced to use `03_agentic_app.ipynb` , you can develop inside a .py module, but please name it as `03_agentic_app.py` and make it explicit how to run your project.
3. You must create test cases to pass the project!

## Submission Instructions

As mentioned above, you're receiving the starter code, but please submit your project with all artifacts under `solution/` . We'll not look into `starter/` . Make sure you are copying and pasting the code from `starter/` to `solution/` if you're not modifying it.
If you have installed a package, share the name and version in the documentation. Ideally share your `requirements.txt` and Python version, If you're developing locally.

### DON'Ts

* import or reference a folder outside `solution/` !
* share your `.env` file
* submit large `.db` files
* submit only the notebooks without the other artifacts

---

# Part 2: My work breakdown

Order matters: the rubric grades that the implementation *matches* the design
document, so the design is written first and nothing else starts until it exists.

| # | Step | Rubric criterion it serves |
|---|---|---|
| 1 | Architecture design doc + diagram in `agentic/design/` | Design and document multi-agent architecture |
| 2 | Copy `starter/` → `solution/`; env + `.env` with Vocareum `base_url` | Submission Instructions |
| 3 | Run notebook 01; expand articles 4 → 14+; run notebook 02 | Set up the database and knowledge base infrastructure |
| 4 | ≥2 CultPass DB tools in `agentic/tools/` with validation + structured errors | Implement support operation tools with database abstraction |
| 5 | Retrieval over the Knowledge table + confidence score → escalate | Implement knowledge-based response system with escalation logic |
| 6 | ≥4 specialized agents in `agentic/agents/` | Implement the designed multi-agent architecture using LangGraph |
| 7 | Hand-wired `StateGraph` in `workflow.py`; routing on content **and** metadata | Implement task routing and role assignment across agents |
| 8 | Checkpointer on `thread_id`, long-term store, messages persisted to `ticket_messages` | Both Memory and State Management criteria |
| 9 | Structured logging + test cases (resolution, escalation, tool use, returning customer) | Demonstrate end-to-end ticket processing workflow with proper logging |
| 10 | Rubric review, then README with package names + versions and Python version | Submission Instructions |

## Constraints that are easy to forget

- **`create_react_agent` is banned for the orchestrator.** "create the graph from
  scratch. Do not use the prebuilt workflow." This is the same call I made for the
  Energy Advisor project.
  It does *not* obviously ban it for the leaf agents; decide that in step 1 and
  log the decision.
- **No imports reaching outside `solution/`.** Every path inside the solution is
  relative to it. This is why the instructions flag relative/absolute paths.
- **`.db` files are not submitted**: the setup notebooks must rebuild them from
  the `.jsonl` files on a clean clone.
- **`.env` is not submitted**; ship an `.env.example` carrying
  `OPENAI_BASE_URL=https://openai.vocareum.com/v1`, or a reviewer running from a
  clean clone gets an auth error. This exact gap was caught once already.
- **FastMCP is a recommendation, not a requirement**: "I strongly recommend".
  It appears again under Suggestions to Make Your Project Stand Out.
