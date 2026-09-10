"""Plan or execute the locked CLS training-strategy matrix; never evaluate test.

Default: print the plan only. --execute is required to launch subprocesses.
--smoke executes A/B/C on tiny train/validation subsets, one config each.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/training_strategy_comparison")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/processed/splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data/eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--model-name-or-path", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    args = parser.parse_args(argv)
    if args.train_baseline and args.baseline_summary:
        parser.error("Choose baseline reuse OR baseline training")
    if args.smoke and args.baseline_summary:
        parser.error("Smoke runs verify a fresh baseline; do not reuse formal results")
    if args.execute and not args.smoke and not (args.train_baseline or args.baseline_summary):
        parser.error("Formal execution requires --baseline-summary or --train-baseline")
    return args


def make_plan(args):
    runs = []
    if args.train_baseline or args.smoke:
        runs.append({"id": "head_lr_1e-4", "strategy": "classifier_head_only"})
    for rate in (["1e-5"] if args.smoke else ["1e-5", "2e-5"]):
        runs.append({"id": f"backbone_lr_{rate}", "strategy": "partial_unfreezing", "backbone_lr": rate})
    for rate in (["5e-5"] if args.smoke else ["5e-5", "1e-4"]):
        runs.append({"id": f"adapter_lr_{rate}", "strategy": "lora", "adapter_lr": rate})
    return runs


def build_command(args, run, output):
    command = [sys.executable, str(ROOT / "src/train_classifier_head.py"),
               "--comparison-mode", "--training-strategy", run["strategy"],
               "--pooling", "cls", "--threshold", "0.5",
               "--learning-rate", "1e-4", "--head-learning-rate", "1e-4",
               "--backbone-learning-rate", run.get("backbone_lr", "1e-5"),
               "--adapter-learning-rate", run.get("adapter_lr", "5e-5"),
               "--unfreeze-last-n", "2", "--lora-r", "8", "--lora-alpha", "16",
               "--lora-dropout", "0.05", "--lora-target-modules", "query", "value",
               "--batch-size", "2" if args.smoke else "32",
               "--epochs", "1" if args.smoke else "5", "--max-length", "512",
               "--dropout", "0.1", "--seed", str(args.seed),
               "--split-dir", str(args.split_dir), "--criteria-path", str(args.criteria_path),
               "--model-name-or-path", args.model_name_or_path, "--output-dir", str(output)]
    if args.smoke:
        command += ["--max-train-samples", "8", "--max-validation-samples", "8"]
    else:
        command.append("--require-cuda")
    assert "--evaluate-test" not in command
    assert command[command.index("--threshold") + 1] == "0.5"
    return command


def read_result(summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads((summary_path.parent / "config.json").read_text(encoding="utf-8"))
    if summary.get("test_evaluated") is not False or config.get("evaluate_test") is not False:
        raise ValueError("Result must explicitly attest test evaluation disabled")
    if any("test_metrics" in item for item in (summary, config)):
        raise ValueError("Test metrics are not accepted in a strategy comparison")
    if summary["pooling"] != "cls" or summary["threshold"] != 0.5:
        raise ValueError("Result violates CLS/threshold policy")
    if summary["checkpoint_selection_metric"] != "validation_pr_auc":
        raise ValueError("Result must use validation AP checkpoint selection")
    if not math.isfinite(summary["best_validation_pr_auc"]):
        raise ValueError("Non-finite best validation AP")
    checkpoint = Path(summary["best_checkpoint_path"])
    if not checkpoint.exists():
        checkpoint = summary_path.parent / checkpoint.name
    if not checkpoint.is_file():
        raise ValueError("Best checkpoint is unavailable locally")
    result = {"strategy": summary["training_strategy"], "seed": summary["seed"],
              "learning_rates": config["optimizer"], "trainable_parameters": summary["trainable_parameters"],
              "best_epoch": summary["best_pr_auc_epoch"],
              "best_validation_ap": summary["best_validation_pr_auc"],
              "checkpoint_path": str(checkpoint.resolve()), "run_path": str(summary_path.parent.resolve()),
              "summary_path": str(summary_path.resolve())}
    return result, config


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps({"execute": False, "test_disabled": True, "pooling": "cls",
                          "baseline": str(args.baseline_summary) if args.baseline_summary else "existing baseline requires verification",
                          "runs": plan}, indent=2))
        return 0
    batch = args.output_root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    batch.mkdir(parents=True, exist_ok=False)
    manifest_path = batch / "comparison_manifest.json"
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "smoke": args.smoke, "test_disabled": True, "pooling": "cls",
                "threshold": 0.5, "selection": "validation_AP", "seed": args.seed,
                "plan": plan, "runs": []}

    def write():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write()
    try:
        if args.baseline_summary:
            baseline, config = read_result(args.baseline_summary)
            required = {"training_strategy": "classifier_head_only", "smoke_test": False,
                        "batch_size": 32, "epochs": 5, "max_length": 512,
                        "dropout": 0.1, "seed": args.seed, "learning_rate": 1e-4,
                        "model_name_or_path": args.model_name_or_path,
                        "loss": "weighted_BCEWithLogitsLoss", "scheduler": None, "gradient_clipping": None}
            if any(config.get(key) != value for key, value in required.items()):
                raise ValueError("Existing baseline config does not match the comparison protocol")
            if baseline["trainable_parameters"] != 769 or not config.get("backbone_frozen"):
                raise ValueError("Baseline must have exactly 769 head-only trainable parameters")
            optimizer = config["optimizer"]
            if (optimizer.get("name") != "AdamW" or optimizer.get("learning_rate") != 1e-4
                    or optimizer.get("weight_decay", .01) != .01):
                raise ValueError("Baseline AdamW configuration differs")
            if not math.isclose(config["pos_weight"], 121346 / 1945, rel_tol=1e-10):
                raise ValueError("Baseline training class weight differs")
            baseline["reused"] = True
            baseline["provenance_note"] = "Historical data/criteria/base-model identity must be verified separately; paths alone are not hashes."
            manifest["runs"].append(baseline)
            write()
        for run in plan:
            manifest["active_run"] = run["id"]
            output = batch / run["strategy"] / run["id"]
            command = build_command(args, run, output)
            manifest["active_command"] = command
            write()
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"{run['id']} exited with code {completed.returncode}")
            summaries = list(output.rglob("run_summary.json"))
            if len(summaries) != 1:
                raise RuntimeError("Expected exactly one completed run summary")
            result, config = read_result(summaries[0])
            if result["strategy"] != run["strategy"] or result["seed"] != args.seed or not config["comparison_mode"]:
                raise ValueError("Child run strategy/seed/guard mismatch")
            if not args.smoke:
                expected = {"classifier_head_only": 769, "partial_unfreezing": 14176513, "lora": 295681}
                if result["trainable_parameters"] != expected[run["strategy"]]:
                    raise ValueError("Unexpected PubMedBERT-base trainable parameter count")
            if config["scheduler"] is not None or config["gradient_clipping"] is not None or config["warmup"] is not None:
                raise ValueError("Child run changed the shared optimizer policy")
            result["reused"] = False
            manifest["runs"].append(result)
            write()
        manifest["status"] = "completed"
        manifest.pop("active_run", None)
        manifest.pop("active_command", None)
        write()
        print(f"Completed: {manifest_path}")
        return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write()
        print(f"Comparison stopped: {error}; manifest={manifest_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
