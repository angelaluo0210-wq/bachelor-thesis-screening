"""Offline CPU integration tests: real 12-block tiny BERT and real PEFT.

All checkpoints/tokenizers/data are synthetic and local. No Hub download or
project test split is needed. Hidden dimensions are reduced, not layer count.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, AutoModel
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from src import train_classifier_head as trainer
from scripts import run_training_strategy_comparison as runner


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("tiny_pubmedbert")
    torch.manual_seed(7)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[", "]",
                  "inclusion", "exclusion", "title", "abstract", "study", "trial",
                  "animal", "human", "review", "eligible", "exclude", "random"]
    tokenizer = Tokenizer(WordPiece({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                                   special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                     pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                     mask_token="[MASK]", model_max_length=512)
    wrapped.save_pretrained(path)
    BertModel(BertConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=12, num_attention_heads=2, max_position_embeddings=512,
                        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)).save_pretrained(path)
    return path


def args_for(base, strategy="classifier_head_only", *extra):
    return trainer.parse_args(["--model-name-or-path", str(base), "--training-strategy", strategy,
                               "--comparison-mode", *extra])


def make_model(args):
    return trainer.build_model(args.model_name_or_path, args.dropout, args.pooling,
                               **trainer.strategy_options(args))


def inputs():
    return {"input_ids": torch.tensor([[2, 11, 12, 3], [2, 13, 14, 3]]),
            "attention_mask": torch.ones((2, 4), dtype=torch.long)}


def step(model, args):
    model.train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = trainer.build_optimizer(model, args)
    for _ in range(2):  # LoRA A may have zero gradient on the first step (B starts zero).
        optimizer.zero_grad(set_to_none=True)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.]))(
            model(**inputs()), torch.tensor([0., 1.]))
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, name
            else:
                assert p.grad is not None, name
        optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert name not in changed, name
    assert "classifier.weight" in changed
    return optimizer, changed


@pytest.mark.parametrize("strategy", trainer.STRATEGIES)
def test_strategy_parameters_modes_real_update_and_roundtrip(tiny_base, tmp_path, strategy):
    args = args_for(tiny_base, strategy)
    model = make_model(args)
    report = trainer.parameter_report(model)
    assert report["training_strategy"] == strategy
    assert report["classifier_parameters"] == 17
    names = set(report["trainable_parameter_names"])
    model.train()
    assert model.dropout.training and model.classifier.training
    if strategy == "classifier_head_only":
        assert names == {"classifier.weight", "classifier.bias"}
        assert not any(m.training for m in model.encoder.modules())
    elif strategy == "partial_unfreezing":
        assert report["adapted_layer_indices"] == [10, 11]
        assert report["trainable_parameters"] == 4465
        assert all(not p.requires_grad for p in model.encoder.embeddings.parameters())
        assert all(not p.requires_grad for p in model.encoder.pooler.parameters())
        for index, layer in enumerate(model.encoder.encoder.layer):
            assert all(p.requires_grad == (index >= 10) for p in layer.parameters())
            assert all(m.training == (index >= 10) for m in layer.modules())
    else:
        assert report["adapter_parameters"] == 12 * 2 * 8 * 32
        assert len(report["lora_target_modules"]) == 24
        for name in report["lora_target_modules"]:
            assert name.endswith(("attention.self.query", "attention.self.value"))
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Dropout):
                assert module.training == ("lora_dropout" in name), name
        assert all(name.startswith("classifier.") or ".lora_A." in name or ".lora_B." in name for name in names)
    optimizer, changed = step(model, args)
    if strategy == "classifier_head_only":
        assert len(optimizer.param_groups) == 1
        assert changed == {"classifier.weight", "classifier.bias"}
    else:
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[1]["lr"] == (1e-5 if strategy == "partial_unfreezing" else 5e-5)
        assert any(name.startswith("encoder.") for name in changed)
    assert all(group["weight_decay"] == 0.01 for group in optimizer.param_groups)
    flat = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(flat) == len(set(flat)) == len(names)
    model.eval()
    assert not any(m.training for m in model.modules())
    with torch.no_grad():
        expected = model(**inputs())
    path = tmp_path / "best.pt"
    trainer.save_checkpoint(path, model, args, 1, {"pr_auc": 0.75}, 2.)
    restored, payload = trainer.load_classifier_checkpoint(path, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs()), expected, rtol=1e-5, atol=1e-6)
    for name, p in model.named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], p, rtol=0, atol=0)
    assert payload["training_strategy"] == strategy
    assert payload["training_arguments"]["head_learning_rate"] is None
    if strategy == "lora":
        assert payload["adapter_config"]["bias"] == "none"
        assert payload["adapter_state_dict"]
        assert any(".lora_A." in name for name in changed)
        assert any(".lora_B." in name for name in changed)
    if strategy == "partial_unfreezing":
        assert set(payload["upper_encoder_state_dicts"]) == {"10", "11"}


def test_legacy_head_numerics_and_old_checkpoint(tiny_base, tmp_path):
    class Legacy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(tiny_base)
            self.encoder.requires_grad_(False)
            self.encoder.eval()
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(16, 1)

        def forward(self, **batch):
            return self.classifier(self.dropout(self.encoder(**batch).last_hidden_state[:, 0])).squeeze(-1)

    args = args_for(tiny_base)
    torch.manual_seed(42)
    legacy = Legacy()
    torch.manual_seed(42)
    current = make_model(args)
    for model, optimizer in ((legacy, torch.optim.AdamW(legacy.classifier.parameters(), lr=1e-4)),
                             (current, trainer.build_optimizer(current, args))):
        torch.manual_seed(1337)
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.]))(model(**inputs()), torch.tensor([0., 1.])).backward()
        optimizer.step()
    for name, p in legacy.named_parameters():
        assert torch.equal(p, dict(current.named_parameters())[name]), name
    path = tmp_path / "legacy.pt"
    trainer.save_checkpoint(path, current, args, 1, {"pr_auc": .5}, 2.)
    payload = torch.load(path, weights_only=True)
    for key in ("checkpoint_version", "training_strategy", "strategy_config", "parameter_report"):
        payload.pop(key)
    torch.save(payload, path)
    restored, _ = trainer.load_classifier_checkpoint(path, torch.device("cpu"))
    current.eval()
    torch.testing.assert_close(restored(**inputs()), current(**inputs()), rtol=0, atol=0)


@pytest.mark.parametrize("strategy", trainer.STRATEGIES)
def test_common_head_initialization(tiny_base, strategy):
    torch.manual_seed(42)
    baseline = make_model(args_for(tiny_base))
    torch.manual_seed(42)
    candidate = make_model(args_for(tiny_base, strategy))
    assert torch.equal(baseline.classifier.weight, candidate.classifier.weight)


def test_ap_selection_and_threshold_independence():
    best, epoch = -float("inf"), 0
    for index, ap in enumerate([.3, .7, .7, .4, float("nan")], 1):
        if trainer.is_better_checkpoint(ap, best):
            best, epoch = ap, index
    assert (best, epoch) == (.7, 2)
    labels, probs = np.array([0, 1, 1, 0]), np.array([.1, .6, .3, .8])
    low = trainer.calculate_metrics(labels, probs, .2)
    high = trainer.calculate_metrics(labels, probs, .7)
    assert low["pr_auc"] == high["pr_auc"]
    assert low["recall"] != high["recall"]


@pytest.mark.parametrize("extra", [["--evaluate-test"], ["--pooling", "mean"],
                                   ["--threshold", "0.2"], ["--unfreeze-last-n", "1"]])
def test_comparison_guard(extra):
    with pytest.raises(SystemExit):
        trainer.parse_args(["--comparison-mode", *extra])


def test_runner_matrix_and_test_guard(tmp_path):
    args = runner.parse_args([])
    plan = runner.make_plan(args)
    assert len(plan) == 4
    assert [r.get("backbone_lr") for r in plan[:2]] == ["1e-5", "2e-5"]
    assert [r.get("adapter_lr") for r in plan[2:]] == ["5e-5", "1e-4"]
    for run in plan:
        command = runner.build_command(args, run, tmp_path / run["id"])
        parsed = trainer.parse_args(command[2:])
        assert not parsed.evaluate_test and parsed.comparison_mode
        assert parsed.pooling == "cls" and parsed.threshold == .5
    with pytest.raises(SystemExit):
        runner.parse_args(["--evaluate-test"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--threshold", "0.9"])


def test_runner_failure_is_recorded_and_stops(tmp_path, monkeypatch):
    calls = []
    def fail(command, check):
        calls.append(command)
        return type("Result", (), {"returncode": 9})()
    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert runner.main(["--execute", "--smoke", "--output-root", str(tmp_path)]) == 1
    assert len(calls) == 1
    manifest = json.loads(next(tmp_path.rglob("comparison_manifest.json")).read_text())
    assert manifest["status"] == "failed" and "9" in manifest["error"]


@pytest.mark.parametrize("strategy", trainer.STRATEGIES)
def test_offline_smoke_artifacts_and_test_never_opened(tiny_base, tmp_path, monkeypatch, strategy):
    split = tmp_path / "splits"
    split.mkdir()
    for name in ("train", "validation"):
        pd.DataFrame({"review_key": [name] * 4, "title": ["human trial", "animal study"] * 2,
                      "abstract": ["eligible trial", "exclude review"] * 2,
                      "label_included": [1, 0, 1, 0]}).to_csv(split / f"{name}.csv", index=False)
    criteria = tmp_path / "criteria.xlsx"
    pd.DataFrame({"review_key": ["train", "validation"], "inclusion_criteria": ["human", "trial"],
                  "exclusion_criteria": ["animal", "review"]}).to_excel(criteria, index=False)
    real_read = trainer.read_split
    reads = []
    def guarded(path, **kwargs):
        assert Path(path).name != "test.csv"
        reads.append(Path(path).name)
        return real_read(path, **kwargs)
    monkeypatch.setattr(trainer, "read_split", guarded)
    output = tmp_path / "outputs"
    argv = ["--model-name-or-path", str(tiny_base), "--split-dir", str(split),
            "--criteria-path", str(criteria), "--output-dir", str(output),
            "--comparison-mode", "--training-strategy", strategy, "--batch-size", "2",
            "--epochs", "1", "--max-length", "32", "--max-train-samples", "4",
            "--max-validation-samples", "4"]
    assert trainer.main(argv) == 0
    assert reads == ["train.csv", "validation.csv"]
    summary_path = next(output.rglob("run_summary.json"))
    assert strategy in summary_path.parts and "seed_42" in summary_path.parts
    summary = json.loads(summary_path.read_text())
    config = json.loads((summary_path.parent / "config.json").read_text())
    assert summary["checkpoint_roundtrip"]["passed"]
    assert summary["parameter_update_check"]["frozen_parameters_unchanged"]
    assert config["training_strategy"] == strategy and not config["test_evaluated"]
    assert config["optimizer"]["weight_decay"] == .01
    assert config["scheduler"] is None and config["gradient_clipping"] is None
    for name in ("training_history.csv", "validation_predictions.csv", "per_review_validation_metrics.csv",
                 "metrics.json", "parameter_report.json", "run.log", "best_pr_auc_classifier_head.pt"):
        assert (summary_path.parent / name).is_file()
    first = trainer.create_run_dir(output / strategy)
    second = trainer.create_run_dir(output / strategy)
    assert first != second
    # Close Windows log handles before pytest removes temporary artifacts.
    for handler in trainer.LOGGER.handlers[:]:
        handler.close()
        trainer.LOGGER.removeHandler(handler)


def test_trainable_audit_rejects_accidental_unfreeze(tiny_base):
    model = make_model(args_for(tiny_base, "partial_unfreezing"))
    model.encoder.embeddings.word_embeddings.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Unexpected trainable"):
        trainer.parameter_report(model)


@pytest.mark.parametrize("strategy", trainer.STRATEGIES)
def test_real_preflight_optimizer_step(tiny_base, strategy):
    args = args_for(tiny_base, strategy, "--batch-size", "2", "--max-length", "32")
    model = make_model(args)
    frame = pd.DataFrame({"review_key": ["a", "a"], "title": ["human", "animal"],
                          "abstract": ["trial", "study"], "label_included": [1, 0],
                          "_inclusion_text": ["trial"] * 2, "_exclusion_text": ["review"] * 2,
                          "_review_text": [""] * 2})
    tokenizer = trainer.AutoTokenizer.from_pretrained(tiny_base)
    result = trainer.one_step_preflight(model, frame, tokenizer, args,
                                        nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert result["frozen_parameters_unchanged"]
    assert result["effective_batch_size"] == result["microbatch_size"] == 2
    assert result["gradient_accumulation_steps"] == 1
    assert "gpu_name" not in result


@pytest.mark.parametrize("strategy", ["partial_unfreezing", "lora"])
def test_incomplete_adaptation_checkpoint_rejected(tiny_base, tmp_path, strategy):
    args = args_for(tiny_base, strategy)
    model = make_model(args)
    path = tmp_path / "incomplete.pt"
    trainer.save_checkpoint(path, model, args, 1, {"pr_auc": .5}, 1.)
    payload = torch.load(path, weights_only=True)
    key = "upper_encoder_state_dicts" if strategy == "partial_unfreezing" else "adapter_state_dict"
    payload[key].pop(next(iter(payload[key])))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete"):
        trainer.load_classifier_checkpoint(path, torch.device("cpu"))


def test_real_test_split_is_sealed():
    path = Path(__file__).resolve().parents[1] / "data/processed/splits/test.csv"
    with pytest.raises(PermissionError, match="sealed"):
        path.open("rb")  # audit hook raises before the file is opened
