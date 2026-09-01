"""Run the four frozen-backbone pooling strategies sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


POOLING_STRATEGIES = ("cls", "mean", "attention", "concat")
EXPECTED_TRAINABLE_PARAMETERS = {
    "cls": 769,
    "mean": 769,
    "attention": 1537,
    "concat": 2305,
}
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-rate",
        required=True,
        help="Shared LR selected manually from Stage 1 validation results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "pooling_comparison",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "splits",
    )
    parser.add_argument(
        "--criteria-path",
        type=Path,
        default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    )
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    try:
        learning_rate = float(args.learning_rate)
    except ValueError as error:
        parser.error(f"learning-rate must be numeric: {error}")
    if learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.epochs <= 0 or not 0.0 <= args.dropout < 1.0:
        parser.error("epochs must be positive and dropout must be in [0, 1)")
    if args.max_length < 16 or not 0.0 <= args.threshold <= 1.0:
        parser.error("max-length/threshold are outside their valid ranges")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    return args


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def create_manifest_path(output_root: Path) -> Path:
    primary = output_root / "pooling_comparison_manifest.json"
    if not primary.exists():
        return primary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"pooling_comparison_manifest_{timestamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"pooling_comparison_manifest_{timestamp}_{suffix:02d}.json"
        suffix += 1
    return candidate


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = create_manifest_path(args.output_root)
    environment = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "selected_shared_learning_rate": args.learning_rate,
        "pooling_strategies": list(POOLING_STRATEGIES),
        "shared_configuration": {
            "model_name_or_path": args.model_name_or_path,
            "train_path": str((args.split_dir / "train.csv").resolve()),
            "validation_path": str((args.split_dir / "validation.csv").resolve()),
            "criteria_path": str(args.criteria_path.resolve()),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "seed": args.seed,
            "max_length": args.max_length,
            "threshold": args.threshold,
            "optimizer": "AdamW",
            "scheduler": None,
            "gradient_clipping": None,
            "loss": "BCEWithLogitsLoss(pos_weight=train_negative/train_positive)",
            "test_evaluation": False,
        },
        "pos_weight": None,
        "expected_trainable_parameter_count_per_pooling": EXPECTED_TRAINABLE_PARAMETERS,
        "trainable_parameter_count_per_pooling": {
            pooling: None for pooling in POOLING_STRATEGIES
        },
        "backbone_frozen_per_pooling": {
            pooling: None for pooling in POOLING_STRATEGIES
        },
        "runtime_environment": environment,
        "runs": {},
        "experimental_difference": (
            "pooling strategy and its necessarily different classifier-side "
            "parameterization only"
        ),
    }
    write_manifest(manifest_path, manifest)

    print("=" * 72, flush=True)
    print("POOLING COMPARISON INVARIANT CHECK: PASS", flush=True)
    print(f"Shared learning rate: {args.learning_rate}", flush=True)
    print("Order: cls -> mean -> attention -> concat", flush=True)
    print("Test evaluation and automatic model/LR selection are disabled.", flush=True)
    print("=" * 72, flush=True)

    trainer = ROOT / "src" / "train_classifier_head.py"
    observed_pos_weight: float | None = None
    for pooling in POOLING_STRATEGIES:
        pooling_root = args.output_root / pooling
        existing_configs = set(pooling_root.rglob("config.json")) if pooling_root.exists() else set()
        command = [
            sys.executable,
            str(trainer),
            "--pooling", pooling,
            "--learning-rate", args.learning_rate,
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--dropout", str(args.dropout),
            "--seed", str(args.seed),
            "--max-length", str(args.max_length),
            "--threshold", str(args.threshold),
            "--num-workers", str(args.num_workers),
            "--split-dir", str(args.split_dir),
            "--criteria-path", str(args.criteria_path),
            "--model-name-or-path", args.model_name_or_path,
            "--output-dir", str(args.output_root),
            "--require-cuda",
        ]
        print(f"\n>>> STARTING pooling={pooling} | output_root={pooling_root}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_pooling"] = pooling
            manifest["exit_code"] = completed.returncode
            write_manifest(manifest_path, manifest)
            print(
                f"\n*** FAILED: pooling={pooling} (exit code {completed.returncode}) ***",
                file=sys.stderr,
                flush=True,
            )
            print("Remaining pooling runs were not started.", file=sys.stderr, flush=True)
            return completed.returncode

        new_configs = set(pooling_root.rglob("config.json")) - existing_configs
        if len(new_configs) != 1:
            manifest["status"] = "failed_manifest_audit"
            manifest["failed_pooling"] = pooling
            manifest["manifest_error"] = f"Expected one new config.json, found {len(new_configs)}"
            write_manifest(manifest_path, manifest)
            print(manifest["manifest_error"], file=sys.stderr, flush=True)
            return 1
        config_path = new_configs.pop()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["pooling"] != pooling or config["test_evaluated"]:
            raise RuntimeError(f"Run invariant violation in {config_path}")
        if not config["backbone_frozen"] or config["backbone_trainable_parameters"] != 0:
            raise RuntimeError(f"Backbone freeze violation in {config_path}")
        if config["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS[pooling]:
            raise RuntimeError(
                f"Unexpected trainable parameter count in {config_path}: "
                f"expected {EXPECTED_TRAINABLE_PARAMETERS[pooling]}, "
                f"found {config['trainable_parameters']}"
            )

        pos_weight = float(config["pos_weight"])
        if observed_pos_weight is None:
            observed_pos_weight = pos_weight
            manifest["pos_weight"] = pos_weight
        elif abs(pos_weight - observed_pos_weight) > 1e-12:
            raise RuntimeError("pos_weight differs across pooling runs")

        manifest["trainable_parameter_count_per_pooling"][pooling] = config[
            "trainable_parameters"
        ]
        manifest["backbone_frozen_per_pooling"][pooling] = True
        manifest["runs"][pooling] = {
            "run_directory": str(config_path.parent.resolve()),
            "config_path": str(config_path.resolve()),
            "trainable_parameters": config["trainable_parameters"],
            "frozen_parameters": config["frozen_parameters"],
            "backbone_trainable_parameters": 0,
            "completed": True,
        }
        write_manifest(manifest_path, manifest)
        print(f">>> COMPLETED pooling={pooling}", flush=True)

    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print("\nAll four pooling runs completed. No winner was selected.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
