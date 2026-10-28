"""Plan or execute sealed-Test supplementary CLS + LoRA experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MULTISEED_SEEDS = (123, 2026)  # Seed 42 is the existing frozen formal result.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", choices=("multiseed", "unweighted-ablation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Optional subset of new robustness seeds (123, 2026); unavailable for ablation.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "supplementary_202610")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(\u6392\u7eb3\u6807\u51c6).xlsx")
    parser.add_argument("--model-name-or-path", default=ROOT / "model" / "pubmedbert")
    return parser.parse_args(argv)


def seeds(args):
    if args.experiment == "unweighted-ablation":
        if args.seeds is not None:
            raise SystemExit("--seeds is not valid for unweighted-ablation; it is fixed at seed 42")
        return (42,)
    selected = tuple(args.seeds) if args.seeds is not None else MULTISEED_SEEDS
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(MULTISEED_SEEDS):
        raise SystemExit("multiseed --seeds must be a non-empty unique subset of 123 2026")
    return selected


def build_command(args, seed: int, output: Path) -> list[str]:
    weighting = "unweighted" if args.experiment == "unweighted-ablation" else "weighted"
    command = [sys.executable, str(ROOT / "src" / "train_classifier_head.py"),
               "--training-strategy", "lora", "--pooling", "cls", "--threshold", "0.5",
               "--class-weighting", weighting, "--learning-rate", "1e-4", "--head-learning-rate", "1e-4",
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
    # Multi-seed runs are exact repeats of the locked historical CLS+LoRA
    # protocol. The ablation intentionally cannot use comparison mode because
    # that mode protects the historical weighted-loss setting.
    if args.experiment == "multiseed":
        command.append("--comparison-mode")
    assert "--evaluate-test" not in command
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    run_seeds = seeds(args)
    root = args.output_root / ("cls_lora_multiseed" if args.experiment == "multiseed" else "cls_lora_unweighted_bce")
    plan = {"experiment": args.experiment, "pooling": "cls", "training_strategy": "lora", "test_disabled": True,
            "seeds": run_seeds, "comparison_mode": args.experiment == "multiseed",
            "class_weighting": "unweighted" if args.experiment == "unweighted-ablation" else "weighted",
            "hyperparameters": {"head_lr": 1e-4, "adapter_lr": 5e-5, "r": 8, "alpha": 16,
            "lora_dropout": .05, "targets": ["query", "value"], "batch_size": 2 if args.smoke else 32,
            "epochs": 1 if args.smoke else 5, "max_length": 512}}
    if not args.execute:
        print(json.dumps({"execute": False, **plan}, indent=2)); return 0
    batch = root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]); batch.mkdir(parents=True, exist_ok=False)
    manifest_path = batch / "manifest.json"; manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke, **plan, "runs": []}
    def write(): manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write()
    try:
        for seed in run_seeds:
            command = build_command(args, seed, batch / f"seed_{seed}"); manifest["active_seed"] = seed; manifest["active_command"] = command; write()
            if subprocess.run(command, check=False).returncode: raise RuntimeError(f"seed {seed} failed")
            summaries = list((batch / f"seed_{seed}").rglob("run_summary.json"))
            if len(summaries) != 1: raise RuntimeError(f"seed {seed}: expected one run summary")
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            if summary["test_evaluated"] or summary["pooling"] != "cls" or summary["training_strategy"] != "lora": raise ValueError("unsafe child result")
            manifest["runs"].append({"seed": seed, "summary_path": str(summaries[0]), "best_validation_ap": summary["best_validation_pr_auc"]}); write()
        manifest["status"] = "completed"; manifest.pop("active_seed", None); manifest.pop("active_command", None); write(); return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error)); write(); return 1


if __name__ == "__main__": raise SystemExit(main())
