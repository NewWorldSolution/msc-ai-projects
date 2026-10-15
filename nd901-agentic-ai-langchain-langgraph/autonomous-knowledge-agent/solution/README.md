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
demonstration, run notebook 02 again first. The long-term customer memory in `data/memory/`
is not reset by notebook 02 and persists across runs on purpose: a customer helped in an
earlier run is recognised as a returning customer. Delete the two files in that folder
for a run that starts with no history.

## What you will see

Seven tickets, eight incoming messages, and one of three outcomes for each. Every result
shows the outcome, the route the message took through the agents, and the reply:

| Ticket | Outcome |
|---|---|
| "How do I reset my password? Please keep replies concise." | Answered from the articles in a short reply; the customer is already known from earlier runs, so the reply opens with a welcome back |
| "I need the password-reset steps again." | Answered; the earlier help is recalled, acknowledged, and the reply stays short as the stored preference asks |
| "What is my current subscription status?" | Answered with the customer's actual status and tier from the database |
| "Can you renew my passport?" | Handed to a human; the customer gets a short receipt, which also greets them as a returning customer |
| "I need help with CultPass, but I have not described the problem yet." | Asked to describe the problem |
| "My problem is signing in. How do I reset my password?" (same ticket, next message) | Answered |
| "How do I update the email address registered on my CultPass account?" | Handed to a human, because no article covers it |
| "I can't log in to my Cultpass account." (a blocked customer) | Handed to a human; the customer gets a receipt |

## How it decides

An intake step finds the ticket and the customer and recalls what is remembered about
them. After that, five specialist agents work under one supervisor, and the supervisor
alone decides where a ticket goes next.

1. **Classifier** works out what the customer is asking, how urgent it is, and whether a
   business fact has to be looked up.
2. **Operations** performs read-only lookups in the business database when a fact is needed.
3. **Knowledge** searches the articles, keeps only those above a relevance threshold, and
   judges whether they actually answer the question.
4. **Response** writes the reply, ties every factual sentence to the article or record it
   came from, and opens with a short acknowledgement when the customer has been helped
   before.
5. **Escalation** writes the factual summary a human receives when the system stops.

The supervisor reviews every draft before it is released. Exact-text checks run in code
first: every factual sentence must be copied from the draft and tied to a real article or
record, receipts may not promise timing or state account values, and the email framing
must be complete. Then a model reviewer judges whether the draft is grounded in the
evidence, complete, in the right form for the channel, and matches the decided outcome.
A draft gets one chance to be repaired; after that the case goes to a human.

The system escalates when the customer is blocked, when the request is outside support
scope, when no article is relevant enough, or when the evidence does not support an
answer. Escalation is a normal outcome, not a failure: the human receives the facts, and
the customer receives a short receipt whenever one can still be written and approved. If
generation itself fails or the call budget is spent, the handoff is saved without a
customer reply and the batch output says so.

## Why it is built this way

These are the main design choices, taken while building and recorded as they were made.

- **A supervisor, not a free conversation between agents.** One place decides routing,
  so every route can be tested and every decision appears in the log.
- **Evidence before wording.** No reply may state a business fact that is not in an
  article or a database record. The reviewer is asked to reject invented consequences,
  even plausible ones, and to list every sentence it cannot support; the code fails the
  draft when that list names a sentence outside every cited claim.
- **Rules in code where the model was unreliable.** Where the model ignored a prompt
  rule, the application enforces the rule instead: credential requests are removed from
  what the classifier and the knowledge agent may ask for, receipts lose their promise
  phrases and account values, every claim is checked against the draft text and its
  source, and the returning-customer acknowledgement is supplied when the model forgets
  it. Prompts request; code guarantees what it can check.
- **One repair, then a human.** Each agent may correct one invalid output. Retrying
  endlessly hides problems and spends money; a human handoff is cheaper and safer.
- **Receipts, not explanations, when escalating.** A customer whose case goes to a human
  gets a short confirmation and no business facts, so nothing unverified reaches them.
- **Memory with limits.** Resolved issues and stated preferences are remembered across
  tickets and used for tone and context, never as proof of a current fact. A returning
  customer's answer opens by acknowledging the earlier help and follows a stored
  preference such as "keep replies concise"; a preference already in memory is not
  stored again.

## What is logged

Every step writes a structured event to the `trace_events` table: classification results,
each routing decision with its destination, each article's relevance score against the
threshold, each database lookup, each review verdict with its reasons, and the final
outcome. The events are searchable by ticket, turn, agent and event type. Conversation
state is checkpointed per ticket, so any ticket can be inspected after the fact. Notebook
03 prints each turn's route as one line built from the routing events, for example
`email -> intake -> supervisor -> classifier -> supervisor -> knowledge -> supervisor ->
response -> supervisor -> persist -> customer (answer)`.

## Known limitations

- The reviewer is a language model. It catches most unsupported statements, but not all.
  In one demo run it accepted a sentence about what a cancelled subscription "means"
  that no article states. The code cannot detect an invented inference on its own.
- Results vary slightly between runs even at temperature 0, because the recalled memory
  and the regenerated business data differ.
- The knowledge base is small, so relevance thresholds were chosen on a handful of test
  questions and would need recalibration for a real corpus.
- The system is read-only. It can explain from the articles how to cancel, pause or change
  a subscription, but it never performs a change itself; nothing in the business data is
  ever written by the agents.
- The returning-customer acknowledgement is guaranteed by code, but its wording comes
  from the model only when the model writes one; otherwise a fixed sentence is used.
  Because the wording is the model's, two answers to the same question from the same
  customer can still come out identical.
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
