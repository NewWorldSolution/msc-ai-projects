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
