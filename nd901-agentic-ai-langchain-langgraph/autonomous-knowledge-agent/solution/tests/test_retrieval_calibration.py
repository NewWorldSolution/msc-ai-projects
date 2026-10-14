"""Knowledge setup recommendations and the notebook's recorded retrieval settings."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agentic.agents import knowledge


@pytest.fixture
def observations():
    article = {
        "article_id": "login-article", "title": "Login recovery",
        "content": "Use Forgot Password and check your spam folder.",
    }
    return [
        {**article, "test_case": "login", "query": "My reset email is missing.", "score": 0.29},
        {**article, "test_case": "passport", "query": "Renew my passport.", "score": -0.06},
    ]


def scripted_model(threshold=0.27):
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = {
        "threshold": threshold,
        "candidate_count": 1 if threshold is not None else None,
        "rationale": "Login guidance is relevant; passport guidance is absent.",
        "limitations": ["Two pilot questions do not establish general accuracy."],
    }
    return model


def test_recommendation_uses_article_text_and_observed_scores(observations):
    before = deepcopy(observations)
    model = scripted_model()
    result = knowledge.recommend_retrieval_threshold(observations, model=model)
    assert result.threshold == 0.27
    assert observations == before
    calls = model.with_structured_output.return_value.invoke
    assert calls.call_count == 1
    prompt = calls.call_args.args[0]
    context = json.loads(prompt.messages[-1].content.split("\n", 1)[1])
    assert len(context["articles"]) == 1
    assert context["articles"][0]["content"] == observations[0]["content"]
    assert context["queries"][1]["candidates"][0]["score"] == -0.06
    assert "not instructions" in prompt.messages[0].content


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "0.27"])
def test_invalid_ai_threshold_is_not_accepted(observations, threshold):
    model = scripted_model(threshold)
    with pytest.raises(ValidationError):
        knowledge.recommend_retrieval_threshold(observations, model=model)
    assert model.with_structured_output.return_value.invoke.call_count == 1


def test_missing_article_content_prevents_model_call(observations):
    observations[0].pop("content")
    model = Mock()
    with pytest.raises(ValueError, match="rerun the probe cell"):
        knowledge.recommend_retrieval_threshold(observations, model=model)
    assert not model.mock_calls


def notebook_cell(marker):
    notebook_path = Path(__file__).resolve().parents[1] / "03_agentic_app.ipynb"
    notebook = json.loads(notebook_path.read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if marker in "".join(cell["source"])
    )


def run_settings_cell(*, article_count=14, embedding_model="text-embedding-3-small"):
    def no_input(prompt):
        raise AssertionError("The automatic notebook must not ask for input.")

    namespace = {
        "documents": [object()] * article_count, "embedding_model_name": embedding_model,
        "RetrievalSettings": dict, "json": json, "input": no_input,
    }
    exec(compile(notebook_cell("retrieval_threshold = "), "retrieval-settings-cell", "exec"), namespace)
    return namespace


def test_notebook_uses_recorded_settings_without_input_or_model(capsys):
    namespace = run_settings_cell()
    assert namespace["retrieval_settings"] == {
        "threshold": 0.15, "candidate_count": 3,
        "embedding_model": "text-embedding-3-small",
        "score_convention": "higher_is_more_relevant",
    }
    assert "WARNING" not in capsys.readouterr().out
    assert "recommend_retrieval_threshold" not in notebook_cell("retrieval_threshold = ")


@pytest.mark.parametrize("changes", [{"article_count": 15}, {"embedding_model": "other-model"}])
def test_notebook_warns_when_calibration_conditions_change(changes, capsys):
    namespace = run_settings_cell(**changes)
    assert "WARNING" in capsys.readouterr().out
    assert namespace["retrieval_settings"]["threshold"] == 0.15


def test_calibration_report_applies_recorded_gate(capsys):
    namespace = run_settings_cell()
    namespace["retrieval_test_queries"] = {"login": "q", "passport": "q"}
    namespace["retrieval_observations"] = [
        {"test_case": "login", "score": 0.2979}, {"test_case": "login", "score": 0.1947},
        {"test_case": "login", "score": -0.05}, {"test_case": "login", "score": -0.1},
        {"test_case": "passport", "score": -0.0632},
    ]
    exec(compile(notebook_cell("calibration_rows = []"), "calibration-report-cell", "exec"), namespace)
    report = namespace["calibration_report"]
    assert report["calibration_conditions"]["article_count"] == 14
    # Login keeps three candidates and passport one; only the two login scores above 0.15 pass.
    assert [row["gate_pass"] for row in report["results"]] == [True, True, False, False]


@pytest.mark.parametrize("count", [0, 2, True, 1.5, "1"])
def test_invalid_recommended_candidate_count(observations, count):
    model = scripted_model()
    model.with_structured_output.return_value.invoke.return_value["candidate_count"] = count
    # why: Only a positive integer within the supplied article catalog is valid.
    with pytest.raises(ValueError):
        knowledge.recommend_retrieval_threshold(observations, model=model)
    assert model.with_structured_output.return_value.invoke.call_count == 1
