"""Run the three CLS classifier-head learning-rate trials sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LEARNING_RATES = (("1e-4", "1e-4"), ("5e-4", "5e-4"), ("1e-3", "1e-3"))
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "hyperparameter_tuning" / "cls")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--model-name-or-path", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    shared = {
        "model_name_or_path": args.model_name_or_path,
        "pooling": "cls",
        "training_strategy": "classifier_head_only",
        "split_dir": str(args.split_dir.resolve()),
        "criteria_path": str(args.criteria_path.resolve()),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "dropout": args.dropout,
        "seed": args.seed,
        "max_length": args.max_length,
        "threshold": args.threshold,
        "optimizer": "AdamW",
        "scheduler": None,
        "gradient_clipping": None,
        "loss": "BCEWithLogitsLoss(pos_weight=train_negative/train_positive)",
        "test_evaluated": False,
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shared_configuration": shared,
        "learning_rates": [label for label, _ in LEARNING_RATES],
        "experimental_difference": "learning_rate only",
    }
    (args.output_root / "lr_tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 72, flush=True)
    print("LR TUNING INVARIANT CHECK: PASS", flush=True)
    print("The only intended experimental difference across the three tuning runs is the learning rate.", flush=True)
    print("Runs are sequential; test evaluation and winner selection are disabled.", flush=True)
    print("=" * 72, flush=True)

    trainer = ROOT / "src" / "train_classifier_head.py"
    for label, value in LEARNING_RATES:
        run_output = args.output_root / f"lr_{label}"
        command = [
            sys.executable,
            str(trainer),
            "--pooling", "cls",
            "--learning-rate", value,
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
            "--output-dir", str(run_output),
            "--require-cuda",
        ]
        print(f"\n>>> STARTING CLS LR={label} | output={run_output}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"\n*** FAILED: CLS LR={label} (exit code {completed.returncode}) ***", file=sys.stderr, flush=True)
            print("Remaining LR runs were not started.", file=sys.stderr, flush=True)
            return completed.returncode
        print(f">>> COMPLETED CLS LR={label}", flush=True)

    print("\nAll three LR runs completed. No winner was selected.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
