# Autonomous Knowledge Agent (UDA-Hub)

Udacity Agentic AI nanodegree, course 3 project. A multi-agent customer support system
built with LangGraph and LangChain for a fictional subscription service, CultPass.

The system reads support tickets, classifies them, looks up business facts in a read-only
database, retrieves knowledge base articles, drafts a reply, reviews it against the
evidence, and either answers the customer, asks a clarifying question, or hands the case
to a human with a factual summary. Every routing decision is logged, and each result in
the demonstration prints the route the message took through the agents.

## Where to look

- **[`solution/`](solution/)** is the submission. Start with
  [`solution/README.md`](solution/README.md): what the system does, how to run it, what
  you will see, how it decides, and its known limitations.
- [`solution/agentic/design/ARCHITECTURE.md`](solution/agentic/design/ARCHITECTURE.md)
  is the full design document.
- `starter/` is the unmodified Udacity starter, kept for reference.
- `INSTRUCTIONS.md` holds Udacity's project brief (Part 1, verbatim) and my own work
  breakdown (Part 2). `rubric.md` is the rubric as captured from the classroom.

## Running it

From the `solution` folder: install `requirements.txt` with Python 3.13, copy
`.env.example` to `.env` and fill in the API key and base URL, then run the notebooks
`01`, `02` and `03` in order. The databases are created by the notebooks and are not part
of this repository. Details are in `solution/README.md`.

## Tests

From the `solution` folder: `python -m pytest -q tests`. The suite uses fake models and
makes no live calls.

## Why this project is bigger than the brief, and how I built it

The rubric asks for four agents, two tools, retrieval with a confidence gate, memory and
logging. This project has all of that, and more: every customer reply is drafted, checked
against its evidence and reviewed before it is released; rules the model ignored in
testing are enforced in code; an escalated customer still receives a short receipt; every
routing decision is logged and printed as the route the case took; and a test suite of
about 600 tests runs without any live model call.

I built it this way on purpose. My background is process optimisation, and I plan to
apply this kind of support automation in my own work. So I treated the project as a
prototype I would have to defend in front of colleagues and auditors: what happens when
the model is wrong, what the customer sees when the system gives up, and how a human can
trace any decision afterwards. Those questions shaped the design more than the rubric did.

I designed the system myself: the Supervisor pattern with five specialists, the read-only
business tools, the one-repair rule, the receipt rules for escalations, the memory scope,
and the decision to keep memory across runs. Those choices, and the trade-offs behind
them, are mine.

I used AI assistants (Claude Code and Codex) the way I would use a pair programmer and a
proofreader. They helped me translate my design into Python, correct code, write and
repair tests, write docstrings and comments, and tidy the wording of this README and the
design document. They also reviewed the finished project against the rubric so I could
fix what they found. I ran every demonstration myself, read every reply the system
produced, and decided what to change after each run.
