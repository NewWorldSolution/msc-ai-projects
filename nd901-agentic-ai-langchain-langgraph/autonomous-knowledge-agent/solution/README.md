# UDA-Hub: an autonomous support agent for CultPass

UDA-Hub reads customer support tickets, decides what each one needs, and either answers
from the company's knowledge base, asks the customer a precise question, or hands the
case to a human with a factual summary. It never sends a reply that has not been checked
against the evidence it used, and it writes down every decision it makes.

The demo customer is CultPass, a subscription card for cultural experiences. The system
has read-only access to CultPass business data (customers, subscriptions, reservations,
experiences) and to a small knowledge base of 14 support articles.

## What you need

- Python 3.13 and the packages in `requirements.txt`.
- A credentials file. Copy `.env.example` to `.env` next to the notebooks and fill in the
  API key and the base URL you were given. The file is never committed.
- Nothing else. The databases are created by the notebooks.

## How to run

Run the three notebooks in order, from this folder, top to bottom:

1. `01_external_db_setup.ipynb` creates the CultPass business database with sample
   customers, subscriptions and experiences.
2. `02_core_db_setup.ipynb` creates the support database, loads the 14 articles and seeds
   seven demo tickets.
3. `03_agentic_app.ipynb` builds the article index, assembles the agents and processes
   every open ticket. It prints each result as it is saved, then an inspection of the
   stored conversation, decisions and memory for each ticket.

Running notebook 03 again processes only tickets that are still open. To repeat the full
demonstration, run notebook 02 again first.

## What you will see

Seven tickets, eight incoming messages, and one of three outcomes for each:

| Ticket | Outcome |
|---|---|
| "How do I reset my password? Please keep replies concise." | Answered from the articles; the preference is remembered |
| "I need the password-reset steps again." | Answered; the earlier help is recalled |
| "What is my current subscription status?" | Answered with the customer's actual status and tier from the database |
| "Can you renew my passport?" | Handed to a human; the customer gets a short receipt |
| "I need help with CultPass, but I have not described the problem yet." | Asked to describe the problem |
| "My problem is signing in. How do I reset my password?" (same ticket, next message) | Answered |
| "How do I update the email address registered on my CultPass account?" | Handed to a human, because no article covers it |
| "I can't log in to my Cultpass account." (a blocked customer) | Handed to a human; the customer gets a receipt |

## How it decides

Six specialised agents work under one supervisor, and the supervisor alone decides where a
ticket goes next.

1. **Intake** finds the ticket and the customer, and recalls what is remembered about them.
2. **Classifier** works out what the customer is asking, how urgent it is, and whether a
   business fact has to be looked up.
3. **Operations** performs read-only lookups in the business database when a fact is needed.
4. **Knowledge** searches the articles, keeps only those above a relevance threshold, and
   judges whether they actually answer the question.
5. **Response** writes the reply, and ties every factual sentence to the article or record
   it came from.
6. **Escalation** writes the factual summary a human receives when the system stops.

The supervisor reviews every draft before it is released. A reply goes out only if it is
grounded in the evidence, complete, in the right form for the channel, and matches the
decided outcome. A draft gets one chance to be repaired; after that the case goes to a
human.

The system escalates when the customer is blocked, when the request is outside support
scope, when no article is relevant enough, or when the evidence does not support an
answer. Escalation is a normal outcome, not a failure: the customer receives a short
receipt and the human receives the facts.

## Why it is built this way

These are the main design choices, taken while building and recorded as they were made.

- **A supervisor, not a free conversation between agents.** One place decides routing,
  so every route can be tested and every decision appears in the log.
- **Evidence before wording.** No reply may state a business fact that is not in an
  article or a database record. The reviewer rejects invented consequences, even
  plausible ones.
- **Rules in code where the model was unreliable.** Several things the model was asked
  to do in a prompt, such as never asking for a password or never promising a response
  time, are enforced by the application instead. Prompts request; code guarantees.
- **One repair, then a human.** Each agent may correct one invalid output. Retrying
  endlessly hides problems and spends money; a human handoff is cheaper and safer.
- **Receipts, not explanations, when escalating.** A customer whose case goes to a human
  gets a short confirmation and no business facts, so nothing unverified reaches them.
- **Memory with limits.** Resolved issues and stated preferences are remembered across
  tickets and used for tone and context, never as proof of a current fact.

## What is logged

Every step writes a structured event to the `trace_events` table: classification results,
each routing decision with its destination, each article's relevance score against the
threshold, each database lookup, each review verdict with its reasons, and the final
outcome. The events are searchable by ticket, turn, agent and event type. Conversation
state is checkpointed per ticket, so any ticket can be inspected after the fact.

## Known limitations

- The reviewer is a language model. It catches most unsupported statements, but not all.
  In one demo run it accepted a sentence about what a cancelled subscription "means"
  that no article states. The code cannot detect an invented inference on its own.
- Results vary slightly between runs even at temperature 0, because the recalled memory
  and the regenerated business data differ.
- The knowledge base is small, so relevance thresholds were chosen on a handful of test
  questions and would need recalibration for a real corpus.
- The system is read-only. It cannot cancel, pause or change anything; such requests go
  to a human.
- When a message contains one clear request and a second, vague remark, the system
  answers the clear request and does not act on the vague remark in that turn. The
  customer can raise it again in a follow-up. This was a deliberate choice: asking the
  model to decide case by case blocked clear answers behind unnecessary questions.

## Folder map

```
solution/
  01_external_db_setup.ipynb   business database
  02_core_db_setup.ipynb       support database, articles, demo tickets
  03_agentic_app.ipynb         the running system
  utils.py                     database helpers and the batch runner
  agentic/
    workflow.py                graph assembly
    state.py                   the shared case state
    agents/                    the six agents
    tools/                     read-only database tools
    design/ARCHITECTURE.md     the full design document
  data/                        created by the notebooks
  tests/                       local test suite, no live model calls
```

Run the tests from this folder with `python -m pytest -q tests`.
