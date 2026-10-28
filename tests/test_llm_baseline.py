"""Synthetic offline tests for Qwen baseline infrastructure; no network or research data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.llm_baseline.alignment import validate_one_to_one_alignment
from src.llm_baseline.client import QwenClient, QwenConfig
from src.llm_baseline.evaluation import (apply_screening_policy, evaluate_policy_records,
                                         evaluate_resolved_records, hard_decision_metrics, wilson_interval)
from src.llm_baseline.prompt import build_prompt, parse_response, prompt_fingerprint
from src.llm_baseline.records import (append_prediction, completed_record_ids,
                                      configuration_fingerprint, create_or_resume_run,
                                      record_id_for_row, require_unique_record_ids, stable_record_id)
from src.llm_baseline.reconstruction import reconstruct_input
from src.llm_baseline.runner import reconstruction_audit
from scripts import run_qwen_baseline


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[", "]", "inclusion", "exclusion",
                  "title", "abstract", "review", "human", "trial", "animal", "study", "hello", "world",
                  "long", "text", "café", "?", "!", "."]
    raw = Tokenizer(WordPiece({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    raw.normalizer = BertNormalizer(lowercase=True)
    raw.pre_tokenizer = BertPreTokenizer()
    raw.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)


def row(**updates: str) -> dict[str, str]:
    value = {"review_key": "Review A", "_inclusion_text": "human trial", "_exclusion_text": "animal",
             "title": "Study title", "abstract": "hello world", "pmid": "1"}
    value.update(updates)
    return value


@pytest.mark.parametrize("updates", [
    {}, {"abstract": ""}, {"_inclusion_text": "", "_exclusion_text": ""},
    {"_inclusion_text": "long " * 300}, {"_exclusion_text": "long " * 300},
    {"title": "long " * 300}, {"abstract": "long " * 600},
    {"title": "Café?! — Unicode"},
])
def test_reconstruction_is_deterministic_and_respects_bert_sequence(tokenizer, updates):
    first = reconstruct_input(tokenizer, row(**updates), 512)
    second = reconstruct_input(tokenizer, row(**updates), 512)
    assert first == second
    active = [token for token in first.input_ids if token != tokenizer.pad_token_id]
    assert active[0] == tokenizer.cls_token_id and active[-1] == tokenizer.sep_token_id
    assert len(first.input_ids) == 512
    assert len(first.token_ids["title"]) <= 64
    assert sum(map(len, first.token_ids.values())) <= 510


def test_boundary_prompt_and_parser(tokenizer):
    content = reconstruct_input(tokenizer, row(abstract="long " * 190), 512)
    prompt = build_prompt(content)
    assert "Inclusion criteria:" in prompt and "Article abstract:" in prompt
    assert prompt_fingerprint() == prompt_fingerprint()
    assert parse_response('{"decision":"include"}').decision == "include"
    assert parse_response('{"decision":"uncertain"}').decision == "uncertain"
    assert parse_response('{"decision":"include", "reason":"x"}').status == "malformed"
    assert parse_response("exclude").status == "malformed"


def test_four_field_caps_fill_the_512_token_sequence(tokenizer):
    content = reconstruct_input(tokenizer, row(_inclusion_text="long " * 400,
                                                _exclusion_text="long " * 400,
                                                title="long " * 400,
                                                abstract="long " * 400), 512)
    assert len([token for token in content.input_ids if token != tokenizer.pad_token_id]) == 512
    assert len(content.token_ids["title"]) == 64
    assert len(content.token_ids["inclusion"]) == 128
    assert len(content.token_ids["exclusion"]) == 128
    assert len(content.token_ids["abstract"]) == 190


def test_reconstruction_audit_checks_every_pilot_record(tokenizer):
    frame = pd.DataFrame([row(pmid="1"), row(pmid="2", abstract="")])
    frame["_qwen_source_row_index"] = [5, 9]
    audit = reconstruction_audit(frame, tokenizer, 512)
    assert audit["records_checked"] == 2
    assert audit["token_id_sequence_match"] == 2
    assert [record["source_row_index"] for record in audit["records"]] == [5, 9]


class MockTransport:
    def __init__(self, values): self.values = list(values); self.payloads = []
    def post_json(self, url, headers, payload):
        self.payloads.append((url, headers, payload))
        value = self.values.pop(0)
        if isinstance(value, Exception): raise value
        return value


def test_mock_client_parses_and_retries_without_network():
    transport = MockTransport([OSError("temporary"), {"choices": [{"message": {"content": '{"decision":"exclude"}'}}], "usage": {"total_tokens": 7}}])
    client = QwenClient(QwenConfig(max_retries=1), transport, api_key="synthetic")
    result, response, retries, error = client.classify_with_retries("prompt", sleep=lambda _: None)
    assert result.decision == "exclude" and response["usage"]["total_tokens"] == 7
    assert retries == 1 and error is None
    assert transport.payloads[0][2]["enable_thinking"] is False
    assert transport.payloads[0][2]["response_format"] == {"type": "json_object"}


def test_client_retries_malformed_json_but_not_uncertain():
    transport = MockTransport([
        {"choices": [{"message": {"content": "not-json"}}]},
        {"choices": [{"message": {"content": '{"decision":"uncertain"}'}}]},
    ])
    result, _, retries, error = QwenClient(QwenConfig(max_retries=1), transport, api_key="synthetic").classify_with_retries(
        "prompt", sleep=lambda _: None
    )
    assert result.decision == "uncertain" and retries == 1 and error is None


def test_client_requires_environment_key_and_runner_has_no_test_argument(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        QwenClient(QwenConfig(), MockTransport([])).classify("prompt")
    with pytest.raises(SystemExit):
        run_qwen_baseline.parse_args(["--model-name-or-path", "synthetic", "--output-dir", "out", "--test"])


def test_transport_error_redacts_injected_key():
    transport = MockTransport([OSError("request used secret-value")])
    client = QwenClient(QwenConfig(max_retries=0), transport, api_key="secret-value")
    parsed, _, retries, error = client.classify_with_retries("prompt", sleep=lambda _: None)
    assert parsed.status == "transport_error" and retries == 0
    assert "secret-value" not in error and "[REDACTED]" in error


def test_resume_fingerprint_and_prediction_storage(tmp_path):
    config = {"prompt": prompt_fingerprint(), "model": "qwen3.8-flash"}
    run = tmp_path / "run"
    manifest = create_or_resume_run(run, config, resume=False)
    assert manifest["configuration_fingerprint"] == configuration_fingerprint(config)
    append_prediction(run, {"record_id": "review_a|pmid=1", "status": "completed"})
    assert completed_record_ids(run) == {"review_a|pmid=1"}
    assert create_or_resume_run(run, config, resume=True)["configuration"] == config
    with pytest.raises(ValueError, match="different prompt"):
        create_or_resume_run(run, {"prompt": "other"}, resume=True)
    with pytest.raises(ValueError, match="Duplicate stored"):
        append_prediction(run, {"record_id": "review_a|pmid=1", "status": "completed"})


def test_stable_ids_allow_same_article_in_different_reviews_and_reject_in_review_duplicate():
    rows = [row(review_key="A", pmid="9"), row(review_key="B", pmid="9")]
    assert len(require_unique_record_ids(rows)) == 2
    with pytest.raises(ValueError, match="Duplicate stable"):
        require_unique_record_ids([row(review_key="A", pmid="9"), row(review_key="A", pmid="9")])
    assert stable_record_id(row(pmid=""), 2).endswith("source_row_index=2")
    assert record_id_for_row(row(pmid="") | {"_qwen_source_row_index": 9}, 2).endswith("source_row_index=9")


def test_alignment_requires_identifiers_and_detects_mismatch():
    source = pd.DataFrame({"review_key": ["A", "B"], "pmid": ["1", "1"], "label_included": [0, 1]})
    predicted = pd.DataFrame({"review_name": ["A", "B"], "pmid": ["1", "1"], "true_label": [0, 1]})
    assert validate_one_to_one_alignment(source, predicted).status == "verified_synthetic"
    mismatch = validate_one_to_one_alignment(source, pd.DataFrame({"review_name": ["A"], "pmid": ["1"]}))
    assert mismatch.status == "mismatch" and mismatch.source_only == 1
    with pytest.raises(ValueError, match="title-based"):
        validate_one_to_one_alignment(pd.DataFrame({"review_key": ["A"]}), pd.DataFrame({"review_name": ["A"]}))


def test_policy_retains_uncertain_malformed_and_transport_failures():
    frame = pd.DataFrame({
        "review_key": ["A"] * 5, "label_included": [1, 0, 1, 0, 1],
        "status": ["completed", "completed", "completed", "malformed", "transport_error"],
        "decision": ["include", "exclude", "uncertain", None, None],
    })
    scored = apply_screening_policy(frame)
    assert scored["retained"].tolist() == [1, 0, 1, 1, 1]
    result = evaluate_policy_records(frame)
    assert result["outcome_counts"] == {"explicit_include": 1, "explicit_exclude": 1, "uncertain": 1,
                                        "malformed_output": 1, "api_transport_failure": 1}
    assert result["micro_global"]["fn"] == 0
    assert result["response_quality"]["valid_model_predictions"]["coverage"] == pytest.approx(3 / 5)
    assert result["response_quality"]["valid_model_predictions"]["uncertain_rate_among_valid"] == pytest.approx(1 / 3)
    assert result["response_quality"]["technical_failures"]["rate"] == pytest.approx(2 / 5)
    assert result["end_to_end_primary"]["denominator_records"] == 5
    assert result["supplementary_valid_responses"]["denominator_records"] == 3


def test_hard_decision_metrics_and_unresolved_rejection():
    metrics = hard_decision_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert metrics["recall"] == 0.5 and metrics["fnr"] == 0.5 and metrics["pr_auc"] is None
    assert wilson_interval(1, 2)["n"] == 2
    frame = pd.DataFrame({"review_key": ["A", "A"], "label_included": [1, 0],
                          "decision": ["include", "exclude"], "status": ["completed", "completed"]})
    assert evaluate_resolved_records(frame)["micro_global"]["recall"] == 1.0
    frame.loc[1, "status"] = "malformed"
    with pytest.raises(ValueError, match="unresolved"):
        evaluate_resolved_records(frame)


def test_pilot_selection_is_label_blind_and_seeded():
    frame = pd.DataFrame([row(pmid=str(index)) | {"label_included": index % 2} for index in range(8)])
    first = run_qwen_baseline._select_pilot(frame, max_records=3, sample_seed=7)
    changed_labels = frame.copy(); changed_labels["label_included"] = 1 - changed_labels["label_included"]
    second = run_qwen_baseline._select_pilot(changed_labels, max_records=3, sample_seed=7)
    assert first["pmid"].tolist() == second["pmid"].tolist()
