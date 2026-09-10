"""Create or verify the canonical pre-AutoDL experiment manifest.

The utility hashes the immutable input files and the supplied local PubMedBERT
snapshot.  It never reads article text from test.csv and never trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
CRITERIA_PATH = ROOT / "data" / "eligibility_criteria(排纳标准).xlsx"
CANONICAL_SPLIT_PATH = ROOT / "config" / "canonical_split.json"
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_record(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["review_key", "label_included"], low_memory=False)
    labels = pd.to_numeric(frame["label_included"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path} has non-binary labels")
    return {
        "path": str(path), "sha256": sha256(path), "rows": int(len(frame)),
        "positive": int(labels.eq(1).sum()), "negative": int(labels.eq(0).sum()),
        "reviews": sorted(frame["review_key"].dropna().astype(str).unique().tolist()),
    }


def model_files(model_path: Path) -> list[dict[str, Any]]:
    required = ("config.json", "tokenizer_config.json", "vocab.txt")
    missing = [name for name in required if not (model_path / name).is_file()]
    weights = [p for p in model_path.iterdir() if p.name in {"pytorch_model.bin", "model.safetensors"}]
    if missing or len(weights) != 1:
        raise ValueError(
            f"Model snapshot must contain {required} and exactly one weight file; "
            f"missing={missing}, weights={[p.name for p in weights]}"
        )
    files = [model_path / name for name in required] + weights
    return [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)]


def build_manifest(model_path: Path) -> dict[str, Any]:
    canonical = json.loads(CANONICAL_SPLIT_PATH.read_text(encoding="utf-8"))
    dataset = {name: split_record(SPLIT_DIR / f"{name}.csv") for name in ("train", "validation", "test")}
    expected = canonical["review_membership"]
    if any(dataset[name]["reviews"] != sorted(expected[name]) for name in expected):
        raise ValueError("Active split review membership differs from config/canonical_split.json")
    seen = [review for name in expected for review in expected[name]]
    if len(seen) != len(set(seen)):
        raise ValueError("Canonical split review membership overlaps")
    return {
        "schema_version": 1,
        "purpose": "formal_training_strategy_comparison",
        "canonical_split": {"path": str(CANONICAL_SPLIT_PATH), "sha256": sha256(CANONICAL_SPLIT_PATH)},
        "dataset": dataset,
        "eligibility_criteria": {"path": str(CRITERIA_PATH), "sha256": sha256(CRITERIA_PATH)},
        "base_model": {"identity": MODEL_ID, "local_path": str(model_path), "files": model_files(model_path)},
        "experiment": {
            "pooling": "cls", "seed": 42, "batch_size": 32, "epochs": 5,
            "max_length": 512, "dropout": 0.1, "loss": "BCEWithLogitsLoss(pos_weight=train_negative/train_positive)",
            "optimizer": "AdamW(weight_decay=0.01)", "checkpoint_selection_metric": "validation_pr_auc",
            "strategies": {
                "classifier_head_only": {"head_learning_rate": 0.0001},
                "partial_unfreezing": {"unfreeze_last_n": 2, "head_learning_rate": 0.0001, "backbone_learning_rates": [0.00001, 0.00002]},
                "lora": {"target_modules": ["query", "value"], "r": 8, "alpha": 16, "dropout": 0.05, "head_learning_rate": 0.0001, "adapter_learning_rates": [0.00005, 0.0001]}
            },
            "test_isolation_rule": "Strategy comparison may open only train.csv and validation.csv; test.csv is reserved for one later final evaluation after researcher-approved validation-only threshold calibration."
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "formal_experiment_manifest.json")
    parser.add_argument("--verify", type=Path, help="Compare an existing manifest with the current inputs without writing.")
    args = parser.parse_args()
    manifest = build_manifest(args.model_path)
    if args.verify:
        saved = json.loads(args.verify.read_text(encoding="utf-8"))
        if saved != manifest:
            raise SystemExit("Manifest verification failed: current inputs differ from the saved manifest")
        print(f"Manifest verification passed: {args.verify}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
