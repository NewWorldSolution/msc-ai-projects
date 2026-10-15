"""Returning-customer acknowledgement and name greeting in Response; fake models, no provider calls."""

from copy import deepcopy

from agentic.agents import response
from test_response_agent import call, case, config, draft  # noqa: F401
from test_response_repair import variant

MEMORY = [
    {"memory_id": "m1", "memory_type": "resolved_issue", "outcome": "resolved",
     "content": "Guidance provided for: Explain cancellation."},
    {"memory_id": "m2", "memory_type": "preference", "outcome": "resolved",
     "content": "Please keep replies concise."},
]
FORM = "Use the cancellation form."


def returning(case, memory=MEMORY):
    case["customer_memory"] = deepcopy(memory)
    return case


def drafted(case, config, value):
    result, model = call(case, config, value)
    assert model.with_structured_output.return_value.invoke.call_count == 1
    return result["response_draft"]


def test_model_acknowledgement_in_the_body_is_kept(case, config):
    opening = "Welcome back! Since we helped you with your cancellation before, here are the steps again."
    value = variant(body=f"{opening} {FORM}", history_acknowledgement=opening)
    result = drafted(returning(case), config, value)
    assert result["personalization"] == {"acknowledgement": opening, "source": "model", "memory_ids": ["m1"]}
    assert result["history_acknowledgement"] == opening
    assert result["body"] == f"{opening} {FORM}"


def test_body_opening_counts_when_the_field_is_empty(case, config):
    value = variant(body=f"Welcome back, Ada. {FORM}")
    result = drafted(returning(case), config, value)
    assert result["personalization"]["source"] == "body"
    assert result["personalization"]["acknowledgement"] == "Welcome back, Ada."
    assert result["body"] == f"Welcome back, Ada. {FORM}"


def test_application_opens_with_the_same_topic_sentence_when_the_model_offers_none(case, config):
    result = drafted(returning(case), config, draft())
    assert result["personalization"]["source"] == "application"
    assert result["body"] == f"{response.HISTORY_ACKNOWLEDGEMENTS['same_topic']}\n\n{FORM}"
    assert result["claim_to_evidence"][0]["claim"] == FORM


def test_other_topic_history_gets_the_neutral_sentence(case, config):
    memory = [{**MEMORY[0], "content": "Guidance provided for: Explain login recovery."}]
    result = drafted(returning(case, memory), config, draft())
    assert result["body"].startswith(response.HISTORY_ACKNOWLEDGEMENTS["other_topic"])


def test_offered_sentence_that_does_not_acknowledge_history_is_not_accepted(case, config):
    # A copied guidance sentence carries no history marker; a promise phrase is never an acknowledgement.
    result = drafted(returning(case), config, variant(history_acknowledgement=FORM))
    assert result["personalization"]["source"] == "application"
    promise = "Welcome back, we will contact you shortly."
    result = drafted(returning(case), config, variant(body=f"{promise} {FORM}", history_acknowledgement=promise))
    assert result["personalization"]["source"] == "application"
    assert result["body"].startswith(response.HISTORY_ACKNOWLEDGEMENTS["same_topic"])


def test_first_time_customer_gets_no_acknowledgement(case, config):
    result = drafted(case, config, variant(history_acknowledgement="Welcome back."))
    assert result["personalization"] is None and result["history_acknowledgement"] is None
    assert result["body"] == FORM
    preference_only = [MEMORY[1]]
    result = drafted(returning(case, preference_only), config, draft())
    assert result["personalization"] is None and result["body"] == FORM


def test_acknowledgement_applies_to_answers_only(case, config):
    case.update(outcome="information_requested")
    returning(case)
    value = {"draft": dict(outcome="information_requested", channel="chat", subject=None, greeting=None,
                           body="Which plan do you mean?", closing=None, addressed_issue_ids=["issue1"],
                           claim_to_evidence=[], clarification_questions=[dict(
                               question="Which plan do you mean?", issue_ids=["issue1"], basis_source_ids=["assignment"])],
                           history_acknowledgement=None),
             "blocker": None}
    result = drafted(case, config, value)
    assert result["personalization"] is None and result["body"] == "Which plan do you mean?"


def email(case, **framing):
    case["channel"] = "email"
    value = variant(channel="email", subject="Cancellation guidance", greeting="Hello,", closing="Kind regards")
    value["draft"].update(framing)
    return value


def test_verified_name_personalises_a_generic_greeting(case, config):
    case["account_context"]["full_name"] = "Ada Lovelace"
    result = drafted(case, config, email(case))
    assert result["greeting"] == "Hello Ada," and result["personalized_greeting"] is True
    assert result["body"].startswith("Hello Ada,\n\n")
    assert result["framing_defaults"] == []


def test_greeting_that_already_names_the_customer_stays(case, config):
    case["account_context"]["full_name"] = "Ada Lovelace"
    result = drafted(case, config, email(case, greeting="Dear Ada,"))
    assert result["greeting"] == "Dear Ada," and result["personalized_greeting"] is False


def test_unknown_name_keeps_the_default_greeting(case, config):
    result = drafted(case, config, email(case, greeting=None))
    assert result["greeting"] == "Hello," and result["personalized_greeting"] is False
    assert result["framing_defaults"] == ["greeting"]


def test_chat_has_no_greeting_to_personalise(case, config):
    case["account_context"]["full_name"] = "Ada Lovelace"
    result = drafted(case, config, draft())
    assert result["greeting"] is None and result["personalized_greeting"] is False


def test_returning_customer_gets_the_instruction_as_its_own_turn(case, config):
    returning(case)
    _, model = call(case, config, draft())
    messages = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()
    assert len(messages) == 3 and messages[-1].type == "human"
    assert '"returning_customer": true' in messages[1].content
    assert "copy that sentence into history_acknowledgement" in messages[0].content
    assert messages[-1].content.startswith("This customer has been helped before.")
    assert "Guidance provided for: Explain cancellation." in messages[-1].content
    assert "Please keep replies concise." in messages[-1].content
    assert "history_acknowledgement" in messages[-1].content


def test_first_time_customer_gets_no_extra_turn(case, config):
    _, model = call(case, config, draft())
    messages = model.with_structured_output.return_value.invoke.call_args.args[0].to_messages()
    assert len(messages) == 2 and '"returning_customer": false' in messages[1].content


def test_acknowledgement_field_is_required_in_the_schema():
    # A required field is filled far more reliably than an optional one by the structured-output call.
    assert "history_acknowledgement" in response.ResponseDraft.model_json_schema()["required"]


def test_declared_sentence_outside_the_opening_does_not_count(case, config):
    # The customer reads the opening first; a greeting buried at the end is not an opening.
    result = drafted(returning(case), config, variant(body=f"{FORM} Welcome back!", history_acknowledgement="Welcome back!"))
    assert result["personalization"]["source"] == "application"
    assert result["body"].startswith(response.HISTORY_ACKNOWLEDGEMENTS["same_topic"])
    assert result["body"].endswith("Welcome back!")


def test_first_time_customer_loses_any_claim_of_earlier_contact(case, config):
    result = drafted(case, config, variant(body=f"Welcome back! {FORM}", history_acknowledgement="Welcome back!"))
    assert result["body"] == FORM and result["removed_history_sentences"] == 1
    assert result["history_acknowledgement"] is None and result["personalization"] is None
    declared = "As you asked last time, here it is."
    result = drafted(case, config, variant(body=f"{FORM} {declared}", history_acknowledgement=declared))
    assert result["body"] == FORM and result["removed_history_sentences"] == 1
    result = drafted(case, config, variant(body=f"Thanks for reaching out. {FORM}"))
    assert result["body"] == f"Thanks for reaching out. {FORM}" and result["removed_history_sentences"] == 0


def test_first_time_reply_that_is_only_a_history_claim_goes_to_repair(case, config):
    value = variant(body="Welcome back!", claim_to_evidence=[], history_acknowledgement="Welcome back!")
    result, model = call(case, config, value, value)
    assert model.with_structured_output.return_value.invoke.call_count == 2
    assert result["trace"][-1]["payload"]["detail"].startswith("A first-time customer's reply must not refer to earlier contact")


def test_first_time_receipt_and_clarification_are_cleaned_too(case, config):
    from test_response_agent import acknowledgement
    value = acknowledgement(case)
    value["draft"].update(body="Welcome back! Your request requires human assistance.", history_acknowledgement="Welcome back!")
    result = drafted(case, config, value)
    assert result["body"] == "Your request requires human assistance." and result["removed_history_sentences"] == 1
