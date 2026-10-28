"""Plan or execute the Train/Validation-only Mean Pooling + LoRA supplement.

The runner deliberately has no Test option.  Each seed receives an isolated
directory and invokes the existing classifier trainer without comparison mode;
the legacy comparison mode remains locked to CLS.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "supplementary_202610" / "mean_lora")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(\u6392\u7eb3\u6807\u51c6).xlsx")
    parser.add_argument("--model-name-or-path", default=ROOT / "model" / "pubmedbert")
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be non-empty and unique")
    return args


def build_command(args, seed: int, output: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "src" / "train_classifier_head.py"),
               "--training-strategy", "lora", "--pooling", "mean", "--threshold", "0.5",
               "--learning-rate", "1e-4", "--head-learning-rate", "1e-4",
               "--adapter-learning-rate", "5e-5", "--lora-r", "8", "--lora-alpha", "16",
               "--lora-dropout", "0.05", "--lora-target-modules", "query", "value",
               "--batch-size", "2" if args.smoke else "32", "--epochs", "1" if args.smoke else "5",
               "--max-length", "512", "--dropout", "0.1", "--seed", str(seed),
               "--split-dir", str(args.split_dir), "--criteria-path", str(args.criteria_path),
               "--model-name-or-path", str(args.model_name_or_path), "--output-dir", str(output)]
    if args.smoke:
        command += ["--max-train-samples", "8", "--max-validation-samples", "8"]
    else:
        command.append("--require-cuda")
    assert "--evaluate-test" not in command and "--comparison-mode" not in command
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = {"pooling": "mean", "training_strategy": "lora", "test_disabled": True,
            "seeds": args.seeds, "hyperparameters": {"head_lr": 1e-4, "adapter_lr": 5e-5,
            "r": 8, "alpha": 16, "lora_dropout": .05, "targets": ["query", "value"],
            "batch_size": 2 if args.smoke else 32, "epochs": 1 if args.smoke else 5, "max_length": 512}}
    if not args.execute:
        print(json.dumps({"execute": False, **plan}, indent=2))
        return 0
    batch = args.output_root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    batch.mkdir(parents=True, exist_ok=False)
    manifest_path = batch / "mean_lora_manifest.json"
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
                **plan, "runs": []}
    def write(): manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write()
    try:
        for seed in args.seeds:
            output = batch / f"seed_{seed}"
            command = build_command(args, seed, output)
            manifest["active_seed"], manifest["active_command"] = seed, command
            write()
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"seed {seed} exited with code {completed.returncode}")
            summaries = list(output.rglob("run_summary.json"))
            if len(summaries) != 1:
                raise RuntimeError(f"seed {seed}: expected one completed run summary")
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            if summary["pooling"] != "mean" or summary["training_strategy"] != "lora" or summary["test_evaluated"]:
                raise ValueError(f"seed {seed}: unsafe child result")
            manifest["runs"].append({"seed": seed, "summary_path": str(summaries[0]),
                                     "best_validation_ap": summary["best_validation_pr_auc"],
                                     "checkpoint_path": summary["best_checkpoint_path"]})
            write()
        manifest["status"] = "completed"; manifest.pop("active_seed", None); manifest.pop("active_command", None); write()
        return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error)); write(); return 1


if __name__ == "__main__":
    raise SystemExit(main())
