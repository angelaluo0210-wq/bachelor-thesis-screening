"""Synthetic offline tests for the sealed Qwen final-Test infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_qwen_final_test as runner
from src.llm_baseline.final_test import (
    ConservativeRateLimiter, FinalEvaluationGateError, FinalTestAuthorizationError,
    evaluate_final_gate, evaluate_qwen_final, execute_final_records, load_authorized_test,
    require_complete_prediction_set, validate_locked_test_population,
)
from src.llm_baseline.prompt import ParsedDecision
from src.llm_baseline.records import create_or_resume_run


REVIEWS = frozenset({"A", "B"})


def _row(review: str, pmid: str, label: int) -> dict[str, object]:
    return {"review_key": review, "pmid": pmid, "doi": "", "openalex_id": "",
            "label_included": label, "title": "title", "abstract": "abstract",
            "_inclusion_text": "include", "_exclusion_text": "exclude"}


def _frame() -> pd.DataFrame:
    return pd.DataFrame([_row("A", "1", 1), _row("B", "2", 0)])


class _Client:
    def __init__(self, outcomes, interrupt_at: int | None = None):
        self.outcomes = list(outcomes)
        self.interrupt_at = interrupt_at
        self.calls = 0

    def classify_with_retries(self, prompt, before_attempt):
        self.calls += 1
        before_attempt()
        if self.interrupt_at == self.calls:
            raise KeyboardInterrupt()
        status, decision = self.outcomes.pop(0)
        return ParsedDecision(status, decision, None), {"usage": {"total_tokens": 7}}, 0, None


@pytest.fixture(autouse=True)
def synthetic_reconstruction(monkeypatch):
    monkeypatch.setattr("src.llm_baseline.final_test.reconstruct_input", lambda *args: object())
    monkeypatch.setattr("src.llm_baseline.final_test.build_prompt", lambda _: "synthetic")


def _new_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    create_or_resume_run(run, {"phase": "final_test", "test_access": "explicitly_authorized"}, resume=False)
    return run


def test_final_population_requires_every_review_and_exact_count():
    assert len(validate_locked_test_population(_frame(), REVIEWS, 2)) == 2
    with pytest.raises(FinalEvaluationGateError, match="exactly 2"):
        validate_locked_test_population(_frame().iloc[:1], REVIEWS, 2)
    wrong_review = _frame().copy(); wrong_review.loc[1, "review_key"] = "C"
    with pytest.raises(FinalEvaluationGateError, match="membership"):
        validate_locked_test_population(wrong_review, REVIEWS, 2)


def test_final_execution_is_append_only_and_resumes_after_interrupt(tmp_path):
    frame, run = _frame(), _new_run(tmp_path)
    limiter = ConservativeRateLimiter(60, clock=lambda: 10.0, sleep=lambda _: None)
    with pytest.raises(KeyboardInterrupt):
        execute_final_records(frame, object(), 512, _Client([("completed", "include"), ("completed", "exclude")], 2),
                              run, limiter, REVIEWS, 2)
    stored = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    assert len(stored) == 1
    result = execute_final_records(frame, object(), 512, _Client([("completed", "exclude")]), run,
                                   limiter, REVIEWS, 2)
    assert result["completed_records"] == 2 and result["missing_records"] == 0
    assert len((run / "predictions.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="different prompt"):
        create_or_resume_run(run, {"phase": "final_test", "prompt": "changed"}, resume=True)


def test_final_execution_keeps_malformed_and_transport_failures(tmp_path):
    run = _new_run(tmp_path)
    result = execute_final_records(
        _frame(), object(), 512, _Client([("malformed", None), ("transport_error", None)]), run,
        ConservativeRateLimiter(60, clock=lambda: 10.0, sleep=lambda _: None), REVIEWS, 2,
    )
    assert result["outcome_counts"] == {"api_transport_failure": 1, "malformed_output": 1}


def test_duplicate_or_missing_qwen_ids_cannot_pass_final_gate():
    frame = _frame()
    incomplete = pd.DataFrame([{"record_id": "a|pmid=1"}])
    with pytest.raises(FinalEvaluationGateError, match="exactly equal"):
        require_complete_prediction_set(frame, incomplete, REVIEWS, 2)
    duplicate = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate stable"):
        validate_locked_test_population(duplicate, frozenset({"A"}), 2)


def _qwen_predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": "a|pmid=1", "review_name": "A", "pmid": "1", "doi": "", "openalex_id": "",
         "true_label": 1, "status": "completed", "decision": "include"},
        {"record_id": "b|pmid=2", "review_name": "B", "pmid": "2", "doi": "", "openalex_id": "",
         "true_label": 0, "status": "completed", "decision": "exclude"},
    ])


def _pubmed_predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"review_name": "A", "pmid": "1", "doi": "", "openalex_id": "", "true_label": 1, "predicted_label": 1},
        {"review_name": "B", "pmid": "2", "doi": "", "openalex_id": "", "true_label": 0, "predicted_label": 0},
    ])


def test_final_evaluation_requires_three_way_alignment_and_emits_hard_label_metrics():
    primary = evaluate_qwen_final(_frame(), _qwen_predictions(), REVIEWS, 2)
    assert primary["micro_global"]["pr_auc"] is None
    assert primary["micro_global"]["recall"] == 1.0
    wrong_qwen_labels = _qwen_predictions().copy(); wrong_qwen_labels.loc[0, "true_label"] = 0
    with pytest.raises(FinalEvaluationGateError, match="Qwen/gold alignment"):
        evaluate_qwen_final(_frame(), wrong_qwen_labels, REVIEWS, 2)
    metrics, paired = evaluate_final_gate(_frame(), _qwen_predictions(), _pubmed_predictions(), REVIEWS, 2)
    assert metrics["micro_global"]["pr_auc"] is None
    assert metrics["micro_global"]["recall"] == 1.0 and len(paired) == 2
    mismatched = _pubmed_predictions().iloc[:1]
    with pytest.raises(FinalEvaluationGateError, match="alignment"):
        evaluate_final_gate(_frame(), _qwen_predictions(), mismatched, REVIEWS, 2)


def test_unauthorized_loader_refuses_before_any_test_read(monkeypatch):
    called = False

    def forbidden_read(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Test read should not occur")

    monkeypatch.setattr("src.llm_baseline.final_test.data.read_split", forbidden_read)
    with pytest.raises(FinalTestAuthorizationError):
        load_authorized_test(Path("synthetic-test.csv"), "wrong")
    assert not called
    with pytest.raises(SystemExit):
        runner.parse_args(["--test-split", "x", "--criteria-path", "y", "--model-name-or-path", "m", "--output-dir", "o"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--test-split", "x", "--criteria-path", "y", "--model-name-or-path", "m",
                           "--output-dir", "o", "--execute", "--max-records", "1"])
    accepted = runner.parse_args(["--test-split", "x", "--criteria-path", "y", "--model-name-or-path", "m",
                                  "--output-dir", "o", "--execute", "--unlock-test", "synthetic"])
    assert accepted.pubmedbert_predictions is None


def test_rate_limiter_waits_between_attempts():
    time = [0.0]
    waits = []
    limiter = ConservativeRateLimiter(60, clock=lambda: time[0], sleep=lambda seconds: waits.append(seconds))
    limiter.wait(); limiter.wait()
    assert waits == [1.0]
