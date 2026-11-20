"""Evaluate the saved weighted/unweighted BCE runs under one Validation rule.

The Validation phase uses saved scores only. The Test phase first reproduces
the unweighted checkpoint's complete Validation scores; it never trains or
selects a threshold or checkpoint from Test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_classifier_head as trainer

W_DIR = ROOT / "outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/lora/adapter_lr_5e-5/lora/seed_42/cls/run_20260910_191547"
U_DIR = ROOT / "supplementary_202610/cls_lora_unweighted_bce/20261028_210437_bdbd9750/seed_42/lora/seed_42/cls/run_20261028_210543"
W_CONFIG = ROOT / "results/formal_training_strategy_20260910/lora_5e-5/config.json"
U_CONFIG = U_DIR / "config.json"
W_THRESHOLD = ROOT / "results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json"
U_THRESHOLD = ROOT / "results/weighted_bce_ablation/unweighted_validation_threshold.json"
W_TEST = ROOT / "outputs/final_test/final_test_predictions.csv"
W_TEST_METRICS = ROOT / "outputs/final_test/final_test_metrics.json"
VAL_SPLIT = ROOT / "data/processed/splits/validation.csv"
TEST_SPLIT = ROOT / "data/processed/splits/test.csv"
MODEL = ROOT / "model/pubmedbert"
RESULT = ROOT / "results/weighted_bce_ablation"
IDENTITY = ("review_name", "title", "pmid", "doi", "openalex_id", "true_label")
CONFIG_FIELDS = (
    "training_strategy", "pooling", "seed", "batch_size", "epochs", "max_length",
    "dropout", "learning_rate", "head_learning_rate", "adapter_learning_rate",
    "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
    "checkpoint_selection_metric", "optimizer",
)
METRIC_FIELDS = ("tp", "fp", "tn", "fn", "recall", "precision", "f1", "fnr", "workload_reduction", "ap")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_identity(frame: pd.DataFrame, split: bool) -> pd.DataFrame:
    rename = {"review_key": "review_name", "label_included": "true_label"} if split else {}
    frame = frame.rename(columns=rename)
    return frame.loc[:, IDENTITY].reset_index(drop=True)


def check_identity(left: pd.DataFrame, right: pd.DataFrame, left_is_split: bool = False,
                   right_is_split: bool = False) -> int:
    a = normalized_identity(left, left_is_split)
    b = normalized_identity(right, right_is_split)
    if len(a) != len(b) or not a.equals(b):
        if len(a) == len(b):
            mismatch = np.flatnonzero((a != b).any(axis=1).to_numpy())
            location = int(mismatch[0]) if len(mismatch) else "unknown"
        else:
            location = "row count"
        raise ValueError(f"Prediction identity or gold-label mismatch at {location}")
    return len(a)


def scores(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame["true_label"].to_numpy(dtype=np.int64)
    p = frame["predicted_probability"].to_numpy(dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Scores must be finite probabilities with both binary labels")
    return y, p


def checked_threshold(path: Path, prediction_path: Path) -> float:
    record = read_json(path)
    if record.get("split") != "validation" or record.get("target_recall") != 0.95:
        raise ValueError(f"Invalid Validation threshold record: {path}")
    if record.get("selection_rule") != "largest_threshold_with_validation_recall_at_least_target":
        raise ValueError(f"Unexpected threshold rule: {path}")
    source = str(record.get("source", "")).replace("\\", "/")
    relative = prediction_path.relative_to(ROOT).as_posix()
    if not source.endswith(relative):
        raise ValueError(f"Threshold source differs from predictions: {path}")
    threshold = float(record["selected"]["threshold"])
    labels, probabilities = scores(read_csv(prediction_path))
    check = trainer.calculate_metrics(labels, probabilities, threshold)
    for field in ("tp", "fp", "tn", "fn"):
        if check[field] != record["selected"][field]:
            raise ValueError(f"Threshold record {field} disagrees with saved scores: {path}")
    if check["recall"] < 0.95:
        raise ValueError(f"Validation threshold fails the target: {path}")
    # Only scores of positive records can change Recall. This proves that the
    # selected value is the largest observed feasible threshold, including ties.
    positives_above = probabilities[labels == 1]
    next_higher = positives_above[positives_above > threshold]
    if len(next_higher) and (next_higher.min() > threshold):
        next_recall = float(np.mean(positives_above >= next_higher.min()))
        if next_recall >= 0.95:
            raise ValueError(f"Threshold is not the largest feasible value: {path}")
    return threshold


def result_row(group: str, split: str, frame: pd.DataFrame, threshold: float) -> dict:
    labels, probabilities = scores(frame)
    metric = trainer.calculate_metrics(labels, probabilities, threshold)
    if metric["tp"] + metric["fp"] + metric["tn"] + metric["fn"] != len(frame):
        raise ValueError("Confusion matrix does not sum to the data size")
    if not np.isclose(metric["workload_reduction"],
                      (metric["tn"] + metric["fn"]) / len(frame), atol=1e-15):
        raise ValueError("Workload reduction must be the predicted exclusion fraction")
    return {"group": group, "split": split, "n": len(frame), "positive": int(labels.sum()),
            "threshold": threshold, **{key: (metric["pr_auc"] if key == "ap" else metric[key])
                                   for key in METRIC_FIELDS}}


def check_config() -> dict:
    weighted, unweighted = read_json(W_CONFIG), read_json(U_CONFIG)
    differences = {key: [weighted.get(key), unweighted.get(key)] for key in CONFIG_FIELDS
                   if weighted.get(key) != unweighted.get(key)}
    if differences:
        raise ValueError(f"Substantive training configuration mismatch: {differences}")
    if weighted.get("pos_weight") != 62.38868894601542 or unweighted.get("pos_weight") is not None:
        raise ValueError("Class-weight provenance mismatch")
    if weighted.get("loss") != "weighted_BCEWithLogitsLoss" or unweighted.get("loss") != "BCEWithLogitsLoss":
        raise ValueError("Loss configuration mismatch")
    if weighted["data_qc"] != unweighted["data_qc"]:
        raise ValueError("Train/Validation data profiles differ")
    return {"matched_fields": list(CONFIG_FIELDS), "differences": {},
            "intentional_loss_difference": {"weighted_pos_weight": weighted["pos_weight"],
                                            "unweighted_pos_weight": None},
            "nontraining_record_differences": ["comparison_mode", "output_dir", "model_name_or_path",
                                               "split_dir", "criteria_path", "environment"]}


def load_unweighted_with_local_base(checkpoint_path: Path, device: torch.device):
    """Restore the original adapter and head on the local base, in memory only."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = trainer.build_model(
        str(MODEL), checkpoint["model_config"]["dropout"],
        checkpoint["model_config"]["pooling"],
        model_revision=checkpoint.get("model_revision"),
        **checkpoint["strategy_config"],
    )
    if model.training_strategy != "lora":
        raise ValueError("Restored model strategy is not LoRA")
    saved = checkpoint["adapter_state_dict"]
    expected = get_peft_model_state_dict(model.encoder)
    if set(saved) != set(expected) or any(saved[k].shape != expected[k].shape for k in expected):
        raise ValueError("LoRA checkpoint tensor names or shapes are incompatible")
    current_config = model.encoder.peft_config["default"].to_dict()
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "target_modules"):
        left, right = checkpoint["adapter_config"][key], current_config[key]
        if key == "target_modules":
            left, right = sorted(left), sorted(right)
        if left != right:
            raise ValueError(f"LoRA checkpoint configuration differs: {key}")
    set_peft_model_state_dict(model.encoder, saved)
    model.classifier.load_state_dict(checkpoint["classifier_head_state_dict"])
    trainer.parameter_report(model)
    model.to(device).eval()
    return model, checkpoint


def validation_phase() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    config_check = check_config()
    val = read_csv(VAL_SPLIT)
    w_val = read_csv(W_DIR / "validation_predictions.csv")
    u_val = read_csv(U_DIR / "validation_predictions.csv")
    check_identity(w_val, u_val)
    check_identity(w_val, val, right_is_split=True)
    if len(val) != 22397:
        raise ValueError("Unexpected Validation population")
    w_threshold = checked_threshold(W_THRESHOLD, W_DIR / "validation_predictions.csv")
    u_threshold = checked_threshold(U_THRESHOLD, U_DIR / "validation_predictions.csv")
    rows = [result_row("Weighted BCE", "validation", w_val, w_threshold),
            result_row("Unweighted BCE", "validation", u_val, u_threshold)]
    pd.DataFrame(rows).to_csv(RESULT / "validation_comparison.csv", index=False)
    write_json(RESULT / "validation_provenance.json", {
        "status": "complete", "selection_data": "validation_only", "target_recall": 0.95,
        "selection_rule": "largest_threshold_with_validation_recall_at_least_target",
        "population": len(val), "identity_and_labels_match_in_order": True,
        "ap_implementation": "sklearn.metrics.average_precision_score via trainer.calculate_metrics",
        "workload_reduction_definition": "(TN + FN) / N = predicted exclusion fraction",
        "configuration_check": config_check,
        "threshold_sources": {"weighted": str(W_THRESHOLD.relative_to(ROOT)),
                              "unweighted": str(U_THRESHOLD.relative_to(ROOT))},
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                         (VAL_SPLIT, W_CONFIG, U_CONFIG, W_DIR / "validation_predictions.csv",
                          U_DIR / "validation_predictions.csv", W_THRESHOLD, U_THRESHOLD,
                          W_DIR / "best_pr_auc_classifier_head.pt",
                          U_DIR / "best_pr_auc_classifier_head.pt")},
    })
    print((RESULT / "validation_comparison.csv").read_text(encoding="utf-8"))


def reproduce_validation(model: torch.nn.Module, tokenizer, checkpoint: dict, device,
                         batch_size: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    val = trainer.read_split(VAL_SPLIT)
    trainer.validate_labels(val, "validation")
    criteria_path = next((ROOT / "data").glob("eligibility_criteria*.xlsx"))
    criteria, schema = trainer.load_eligibility_criteria(criteria_path)
    val = trainer.join_criteria(val, criteria, schema, "validation")
    max_length = int(checkpoint["training_arguments"]["max_length"])
    if max_length != 512:
        raise ValueError("Unexpected checkpoint max_length")
    loader = trainer.make_loader(val, tokenizer, max_length, batch_size, False, 0, 42)
    saved = read_csv(U_DIR / "validation_predictions.csv")
    check_identity(read_csv(VAL_SPLIT), saved, left_is_split=True)
    stored = saved["predicted_probability"].to_numpy(dtype=np.float64)
    threshold = checked_threshold(U_THRESHOLD, U_DIR / "validation_predictions.csv")
    probability_chunks, label_chunks, index_chunks = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Reproducing Unweighted Validation"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                criteria_mask=batch["criteria_mask"].to(device),
                article_mask=batch["article_mask"].to(device),
            )
            current = torch.sigmoid(logits.cpu()).numpy().astype(np.float64)
            current_labels = batch["labels"].numpy().astype(np.int64)
            current_indices = batch["row_indices"].numpy().astype(np.int64)
            reference = stored[current_indices]
            close_here = np.isclose(current, reference, rtol=2e-4, atol=2e-5)
            decisions_here = (current >= threshold) == (reference >= threshold)
            if not close_here.all() or not decisions_here.all():
                bad = int(np.flatnonzero(~(close_here & decisions_here))[0])
                outcome = {"status": "failed", "reason": "probability or threshold decision differs",
                           "first_mismatch_row": int(current_indices[bad]),
                           "reproduced_probability": float(current[bad]),
                           "saved_probability": float(reference[bad]),
                           "probabilities_close": bool(close_here[bad]),
                           "threshold_decision_identical": bool(decisions_here[bad]),
                           "model_weight_sha256": sha256(MODEL / "pytorch_model.bin"),
                           "checkpoint_sha256": sha256(U_DIR / "best_pr_auc_classifier_head.pt")}
                write_json(RESULT / "validation_reproduction.json", outcome)
                raise RuntimeError("Validation probability reproduction failed; Test inference prohibited")
            probability_chunks.append(current)
            label_chunks.append(current_labels)
            index_chunks.append(current_indices)
    probabilities = np.concatenate(probability_chunks)
    labels = np.concatenate(label_chunks)
    indices = np.concatenate(index_chunks)
    if not np.array_equal(indices, np.arange(len(val))):
        raise ValueError("Validation inference row order changed")
    reproduced = trainer.prediction_frame(val, probabilities, labels, indices, 0.5)
    check_identity(reproduced.fillna("").astype(str), saved)
    difference = np.abs(probabilities.astype(np.float64) - stored)
    same_decisions = bool(np.array_equal(probabilities >= threshold, stored >= threshold))
    close = bool(np.allclose(probabilities, stored, rtol=2e-4, atol=2e-5))
    from sklearn.metrics import average_precision_score
    reproduced_ap = float(average_precision_score(labels, probabilities))
    stored_ap = float(average_precision_score(labels, stored))
    ap_close = abs(reproduced_ap - stored_ap) <= 1e-6
    outcome = {"status": "passed" if close and same_decisions and ap_close else "failed",
               "rows": len(val), "identity_and_labels_match_in_order": True,
               "max_absolute_probability_difference": float(difference.max()),
               "mean_absolute_probability_difference": float(difference.mean()),
               "probability_tolerance": {"rtol": 2e-4, "atol": 2e-5},
               "probabilities_close": close, "threshold_decisions_identical": same_decisions,
               "ap_reproduced": reproduced_ap, "ap_saved_scores": stored_ap,
               "ap_difference_within_1e-6": ap_close,
               "model_weight_sha256": sha256(MODEL / "pytorch_model.bin"),
               "model_config_sha256": sha256(MODEL / "config.json"),
               "tokenizer_config_sha256": sha256(MODEL / "tokenizer_config.json"),
               "vocab_sha256": sha256(MODEL / "vocab.txt"),
               "checkpoint_sha256": sha256(U_DIR / "best_pr_auc_classifier_head.pt"),
               "device": str(device), "batch_size": batch_size, "torch_threads": threads,
               "torch_version": torch.__version__}
    write_json(RESULT / "validation_reproduction.json", outcome)
    if outcome["status"] != "passed":
        raise RuntimeError("Validation probability reproduction failed; Test inference prohibited")
    return outcome


def test_phase(batch_size: int, threads: int) -> None:
    if not (RESULT / "validation_provenance.json").is_file():
        raise FileNotFoundError("Complete the Validation phase first")
    if any((RESULT / name).exists() for name in
           ("unweighted_test_predictions.csv", "test_comparison.csv", "test_provenance.json")):
        raise FileExistsError("Test outputs already exist; refusing to overwrite")
    check_config()
    checkpoint_path = U_DIR / "best_pr_auc_classifier_head.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("checkpoint_version") != 2 or checkpoint.get("epoch") != 1 or
            checkpoint.get("seed") != 42 or checkpoint.get("training_strategy") != "lora" or
            checkpoint.get("pos_weight") is not None or
            checkpoint.get("model_config", {}).get("pooling") != "cls"):
        raise ValueError("Unweighted checkpoint metadata is incompatible")
    expected_ap = read_json(U_DIR / "metrics.json")["overall_validation"]["pr_auc"]
    if abs(checkpoint["validation_metrics"]["pr_auc"] - expected_ap) > 1e-12:
        raise ValueError("Checkpoint selection AP disagrees with saved run")
    if not (MODEL / "pytorch_model.bin").is_file():
        raise FileNotFoundError("Local PubMedBERT base weights missing")
    trainer.set_seed(42)
    device = trainer.select_device()
    if device.type != "cuda":
        raise RuntimeError("The full reproduction and Test phase requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model, checkpoint = load_unweighted_with_local_base(checkpoint_path, device)
    reproduction = reproduce_validation(model, tokenizer, checkpoint, device, batch_size, threads)
    # No Test split or gold label is opened before the reproduction gate above.
    threshold = checked_threshold(U_THRESHOLD, U_DIR / "validation_predictions.csv")
    w_threshold = checked_threshold(W_THRESHOLD, W_DIR / "validation_predictions.csv")
    test = trainer.read_split(TEST_SPLIT)
    trainer.validate_labels(test, "test")
    if len(test) != 23592:
        raise ValueError("Unexpected Test population")
    criteria_path = next((ROOT / "data").glob("eligibility_criteria*.xlsx"))
    criteria, schema = trainer.load_eligibility_criteria(criteria_path)
    test = trainer.join_criteria(test, criteria, schema, "test")
    loader = trainer.make_loader(test, tokenizer, 512, batch_size, False, 0, 42)
    _, _, probabilities, labels, indices = trainer.evaluate(
        model, loader, nn.BCEWithLogitsLoss(), device, threshold, "Unweighted Test inference"
    )
    if not np.array_equal(indices, np.arange(len(test))):
        raise ValueError("Test inference row order changed")
    u_test = trainer.prediction_frame(test, probabilities, labels, indices, threshold)
    u_test_text = u_test.fillna("").astype(str)
    test_text = read_csv(TEST_SPLIT)
    w_test = read_csv(W_TEST)
    check_identity(u_test_text, test_text, right_is_split=True)
    check_identity(w_test, test_text, right_is_split=True)
    check_identity(u_test_text, w_test)
    rows = [result_row("Weighted BCE", "test", w_test, w_threshold),
            result_row("Unweighted BCE", "test", u_test_text, threshold)]
    official = read_json(W_TEST_METRICS)["at_locked_threshold"]
    for key in ("tp", "fp", "tn", "fn", "recall", "precision", "f1", "fnr", "workload_reduction"):
        if not np.isclose(rows[0][key], official[key], atol=1e-10):
            raise ValueError(f"Weighted Test {key} differs from frozen official result")
    if not np.isclose(rows[0]["ap"], official["pr_auc"], atol=1e-10):
        raise ValueError("Weighted Test AP differs from frozen official result")
    u_test.to_csv(RESULT / "unweighted_test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULT / "test_comparison.csv", index=False)
    write_json(RESULT / "test_provenance.json", {
        "status": "complete", "reproduction_gate": reproduction["status"],
        "test_row_identity_and_labels_match_in_order": True,
        "thresholds_selected_from": "Validation predictions only",
        "checkpoint_selection": "saved best Validation AP checkpoint for each group",
        "test_labels_used_for_selection": False,
        "weighted_official_result_unchanged_and_matched": True,
        "ap_implementation": "sklearn.metrics.average_precision_score via trainer.calculate_metrics",
        "workload_reduction_definition": "(TN + FN) / N = predicted exclusion fraction",
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                         (TEST_SPLIT, W_TEST, W_TEST_METRICS, W_THRESHOLD, U_THRESHOLD,
                          checkpoint_path, MODEL / "pytorch_model.bin", criteria_path)},
        "output_sha256": {"unweighted_test_predictions.csv":
                          sha256(RESULT / "unweighted_test_predictions.csv")},
    })
    print((RESULT / "test_comparison.csv").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.torch_threads < 1:
        parser.error("batch-size and torch-threads must be positive")
    if args.phase == "validation":
        validation_phase()
    else:
        test_phase(args.batch_size, args.torch_threads)


if __name__ == "__main__":
    main()
