"""Package existing thesis evidence without training, inference, or API calls.

Creates a new results/reproducibility directory and refuses to overwrite it.
The package contains public, field-minimized copies of the four pooling runs;
all original experiment files remain untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("REPRO_SOURCE_ROOT", ROOT)).resolve()
OUT = ROOT / "results/reproducibility"
POOL = Path("outputs/pooling_comparison/formal/20260905_122936")
STRAT = Path("results/formal_training_strategy_20260910")
WINNER = Path("outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/lora/adapter_lr_5e-5/lora/seed_42/cls/run_20260910_191547")
MEAN = Path("supplementary_202610/mean_lora/20261028_085641_aaeca089")
CLS = Path("supplementary_202610/cls_lora_multiseed/20261028_161144_dfbd2800")
UNWEIGHTED = Path("supplementary_202610/cls_lora_unweighted_bce/20261028_210437_bdbd9750")
TFIDF = Path("supplementary_202610/tfidf_logreg")
QWEN = Path("supplementary_202610/llm_baseline/qwen_final_test_20261029")
QLEN = Path("outputs/qwen_full_input_validation_comparison")
FULL = Path("outputs/pubmedbert_vs_qwen_full_test_subset")
INPUTS = ["data/processed/splits/train.csv", "data/processed/splits/validation.csv", "data/eligibility_criteria(排纳标准).xlsx"]


CRITERIA = INPUTS[-1]


def rel(path: Path | str) -> str:
    return Path(path).as_posix()


def actual(path: Path | str) -> Path:
    return SOURCE_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict:
    return json.loads(actual(path).read_text(encoding="utf-8"))


def scrub_string(value: str) -> str:
    value = re.sub(r"(?i)[A-Z]:\\Users\\[^\\]+\\Desktop\\[^\\]+", "<PROJECT_ROOT>", value)
    value = re.sub(r"/(?:root|home)/[^/\s]+/(?:thesis_project/)?", "<PROJECT_ROOT>/", value)
    value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]{16,}", "Bearer <REDACTED>", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}\b", "<REDACTED_TOKEN>", value)
    value = re.sub(r"(?i)(DASHSCOPE_API_KEY\s*=\s*)[^\s]+", r"\1<REDACTED>", value)
    return value


def scrub(value):
    if isinstance(value, str):
        return scrub_string(value)
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def run_dir(base: Path, seed: int, pool: str) -> Path:
    found = list(actual(base / f"seed_{seed}/lora/seed_{seed}/{pool}").glob("run_*"))
    if len(found) != 1:
        raise ValueError(f"Expected one historical run at {base}, seed={seed}, pooling={pool}; found {len(found)}")
    return found[0].relative_to(SOURCE_ROOT)


def environment(config: Path | None = None, *, python: str = "not_recorded") -> dict:
    names = ("torch", "transformers", "peft", "accelerate", "numpy", "pandas", "scikit_learn", "cuda", "gpu")
    result = {name: "not_recorded" for name in names}
    result["python"] = python
    if config is None or not actual(config).is_file():
        return result
    document = read_json(config)
    settings = document.get("environment", {})
    runtime = document.get("configuration", {}).get("runtime", {})
    packages = settings.get("packages", {})
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "pandas"):
        result[name] = packages.get(name, "not_recorded")
    result["scikit_learn"] = packages.get("scikit-learn", "not_recorded")
    result["torch"] = settings.get("pytorch_version", result["torch"])
    result["transformers"] = settings.get("transformers_version", result["transformers"])
    result["cuda"] = settings.get("cuda_version", "not_recorded")
    result["gpu"] = settings.get("gpu_name", "not_recorded")
    result["python"] = runtime.get("python", result["python"])
    result["transformers"] = runtime.get("transformers", result["transformers"])
    return result


def main() -> None:
    if OUT.exists() and ("--refresh-generated" not in sys.argv or not (OUT / "experiment_provenance.json").is_file()):
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=SOURCE_ROOT).decode("utf-8").strip("\0").split("\0"))
    records: list[dict] = []
    checkpoints: list[dict] = []

    def checkpoint(experiment: str, path: Path | None, role: str = "checkpoint", recorded: str = "not_recorded") -> str:
        if path is None:
            return "not_applicable"
        name = rel(path)
        file = actual(path)
        digest = sha256(file) if file.is_file() else "missing"
        if recorded not in ("not_recorded", digest):
            raise ValueError(f"Recorded checkpoint hash differs: {name}")
        checkpoints.append({"experiment_id": experiment, "role": role, "path": name,
                            "sha256": digest, "recorded_sha256": recorded,
                            "exists": file.is_file(), "git_tracked": name in tracked,
                            "delivery_exists": (ROOT / path).is_file(),
                            "size_bytes": file.stat().st_size if file.is_file() else "missing"})
        return digest

    def add(experiment_id: str, thesis_result: str, split: str, entry: str,
            configs: list[Path], predictions: list[Path], metrics: list[Path], logs: list[Path],
            *, seed: str = "not_recorded", checkpoint_path: Path | None = None,
            checkpoint_recorded_sha: str = "not_recorded", git_commit: str = "not_recorded",
            preparation_git_commit: str = "not_recorded", worktree: str = "not_recorded",
            env_config: Path | None = None, python: str = "not_recorded",
            verification: str = "stored_metrics_only", notes: str = "",
            inputs: list[str] | None = None) -> None:
        artifact_paths = [Path(item) for item in (inputs or INPUTS)] + [*configs, *predictions, *metrics, *logs]
        if checkpoint_path is not None:
            artifact_paths.append(checkpoint_path)
        names = [rel(path) for path in artifact_paths]
        presence = {name: actual(name).is_file() for name in names}
        tracking = {name: name in tracked for name in names}
        digest = checkpoint(experiment_id, checkpoint_path, recorded=checkpoint_recorded_sha)
        missing = [name for name, exists in presence.items() if not exists]
        record = {
            "experiment_id": experiment_id, "thesis_result": thesis_result,
            "split": split, "input_files": inputs or INPUTS,
            "entry_script": entry, "historical_entry_command_recorded": False,
            "git_commit": git_commit, "preparation_git_commit": preparation_git_commit,
            "run_worktree_state": worktree, "random_seed": seed,
            "environment": environment(env_config, python=python),
            "configuration_paths": [rel(path) for path in configs],
            "checkpoint_path": rel(checkpoint_path) if checkpoint_path else "not_applicable",
            "checkpoint_sha256": digest,
            "prediction_paths": [rel(path) for path in predictions],
            "metrics_paths": [rel(path) for path in metrics],
            "run_log_paths": [rel(path) for path in logs],
            "file_exists": presence, "git_tracked": tracking,
            "verification_status": verification,
            "reproducibility_status": "Missing" if experiment_id in (
                "STRAT-HEAD", "STRAT-PARTIAL-1E5", "STRAT-PARTIAL-2E5", "STRAT-LORA-1E4"
            ) else "Partial",
            "missing_artifacts": missing, "notes": notes,
        }
        records.append(record)

    add("DATA-SPLIT", "§3 dataset split; §4.1 dataset composition", "train/validation/test", "src/split_dataset.py",
        [Path("config/canonical_split.json")], [Path(x) for x in INPUTS[:2]] + [Path("data/processed/splits/test.csv")],
        [Path("data/processed/splits/split_summary.csv")], [], verification="counts_and_hashes_verified",
        notes="Data files are local and excluded from Git; historical split execution SHA and rationale are not recorded.")

    pool_runs = {"CLS": "run_20260905_123026", "CONCAT": "run_20260905_151150",
                 "MEAN": "run_20260905_132403", "ATTENTION": "run_20260905_141749"}
    for label, run in pool_runs.items():
        directory = POOL / label.lower() / run
        config = directory / "config.json"
        add(f"POOL-{label}", "§4.2 Pooling Strategy Comparison", "validation", "scripts/run_pooling_comparison.py; src/train_classifier_head.py",
            [config], [directory / "validation_predictions.csv"], [directory / "metrics.json", directory / "run_summary.json"],
            [directory / "run.log"], seed="42", checkpoint_path=directory / "best_pr_auc_classifier_head.pt",
            env_config=config, verification="recomputed_from_22397_predictions",
            notes="Public copies in results/reproducibility/pooling_comparison; exact historical Git SHA not recorded.")

    strategy_manifest = read_json(Path("outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/comparison_manifest.json"))
    strategy_names = ["HEAD", "PARTIAL-1E5", "PARTIAL-2E5", "LORA-5E5", "LORA-1E4"]
    result_names = ["classifier_head_only", "partial_unfreezing_1e-5", "partial_unfreezing_2e-5", "lora_5e-5", "lora_1e-4"]
    for exp, label, run in zip(strategy_names, result_names, strategy_manifest["runs"]):
        directory = STRAT / label
        config = directory / "config.json"
        old = run["checkpoint_path"]
        if "/outputs/" not in old:
            raise ValueError("Unexpected historical checkpoint path format")
        ckpt = Path("outputs") / old.split("/outputs/", 1)[1]
        pred = ckpt.parent / "validation_predictions.csv"
        add(f"STRAT-{exp}", "§4.3 Training Strategy Comparison", "validation", "scripts/run_training_strategy_comparison.py; src/train_classifier_head.py",
            [config], [pred], [directory / "metrics.json", directory / "run_summary.json", directory / "training_history.csv"],
            [ckpt.parent / "run.log"], seed="42", checkpoint_path=ckpt, env_config=config,
            checkpoint_recorded_sha="f635ed339a6902d4f2c695b27baf3ed977813c1fa361c9b61f9cf293fe44a6d1" if exp == "LORA-5E5" else "not_recorded",
            verification="recomputed_from_22397_predictions" if exp == "LORA-5E5" else "stored_metrics_only",
            notes="Four nonselected runs lack local original predictions and checkpoints; copied config/metrics/history/summary remain." if exp != "LORA-5E5" else "Winner prediction and checkpoint exist; original run log is missing.")

    for suffix, variant in (("A", "baseline_a_title_abstract"), ("B", "baseline_b_criteria_title_abstract")):
        directory = TFIDF / variant
        add(f"TFIDF-{suffix}", "§4.4 Supplementary Validation TF-IDF", "validation", "scripts/run_tfidf_logreg_supplementary.py",
            [TFIDF / "config.json", directory / "config.json", directory / "run_manifest.json"],
            [directory / "validation_predictions.csv"], [directory / "validation_metrics.json", directory / "threshold_calibration.json"],
            [TFIDF / "run.log"], seed="42", verification="recomputed_from_22397_predictions",
            notes="Local logistic-regression/vectorizer joblib files exist but are untracked; model artifact hashes appear in checkpoint_sha256.csv.")
        for role, filename in (("logistic_regression", "logistic_regression.joblib"), ("tfidf_vectorizer", "tfidf_vectorizer.joblib")):
            checkpoint(f"TFIDF-{suffix}", directory / "model" / filename, role=role)

    for seed in (123, 2026):
        directory = run_dir(CLS, seed, "cls")
        config = directory / "config.json"
        add(f"CLS-LORA-SEED-{seed}", "§4.4 Supplementary Validation multi-seed", "validation", "src/train_classifier_head.py",
            [config, CLS / "manifest.json"], [directory / "validation_predictions.csv"],
            [directory / "metrics.json", directory / "run_summary.json"], [directory / "run.log"], seed=str(seed),
            checkpoint_path=directory / "best_pr_auc_classifier_head.pt", env_config=config,
            verification="recomputed_from_22397_predictions")
    for seed in (42, 123, 2026):
        directory = run_dir(MEAN, seed, "mean")
        config = directory / "config.json"
        add(f"MEAN-LORA-SEED-{seed}", "§4.4 Supplementary Validation mean pooling", "validation", "scripts/run_mean_lora_supplementary.py; src/train_classifier_head.py",
            [config, MEAN / "mean_lora_manifest.json"], [directory / "validation_predictions.csv"],
            [directory / "metrics.json", directory / "run_summary.json"], [directory / "run.log"], seed=str(seed),
            checkpoint_path=directory / "best_pr_auc_classifier_head.pt", env_config=config,
            verification="recomputed_from_22397_predictions")

    unweighted_dir = run_dir(UNWEIGHTED, 42, "cls")
    add("BCE-UNWEIGHTED-ORIGINAL", "§4.4 Supplementary Validation; default threshold 0.5", "validation", "src/train_classifier_head.py",
        [unweighted_dir / "config.json", UNWEIGHTED / "manifest.json"], [unweighted_dir / "validation_predictions.csv"],
        [unweighted_dir / "metrics.json", unweighted_dir / "run_summary.json"], [unweighted_dir / "run.log"],
        seed="42", checkpoint_path=unweighted_dir / "best_pr_auc_classifier_head.pt", env_config=unweighted_dir / "config.json",
        verification="recomputed_from_22397_predictions")
    add("BCE-VALIDATION-CALIBRATED", "Teacher comment 1; supplementary fair comparison not yet in thesis", "validation", "scripts/evaluate_weighted_bce_ablation.py",
        [Path("results/weighted_bce_ablation/validation_provenance.json"), Path("results/weighted_bce_ablation/unweighted_validation_threshold.json")],
        [WINNER / "validation_predictions.csv", unweighted_dir / "validation_predictions.csv"],
        [Path("results/weighted_bce_ablation/validation_comparison.csv"), Path("results/weighted_bce_ablation/validation_preflight.json")], [],
        seed="42", checkpoint_path=WINNER / "best_pr_auc_classifier_head.pt", env_config=STRAT / "lora_5e-5/config.json",
        verification="paired_identity_and_metrics_verified", notes="Each model uses its own Validation threshold: maximum threshold satisfying Recall >= 0.95. Original predictions are unchanged.")

    for target, filename in (("090", "validation_threshold_recall_090.json"), ("095", "selected_validation_threshold.json"), ("098", "validation_threshold_recall_098.json")):
        add(f"THRESHOLD-{target}", "§4.5 Validation threshold calibration", "validation", "scripts/calibrate_validation_threshold.py",
            [STRAT / "lora_5e-5/config.json"], [WINNER / "validation_predictions.csv"], [STRAT / "thresholds" / filename], [],
            seed="42", checkpoint_path=WINNER / "best_pr_auc_classifier_head.pt", env_config=STRAT / "lora_5e-5/config.json",
            verification="recomputed_from_saved_validation_predictions")
    add("PUBMEDBERT-FINAL-TEST", "§4.6 Final Test and four per-review rows", "test", "scripts/evaluate_final_test.py",
        [Path("outputs/final_test/final_test_config.json")], [Path("outputs/final_test/final_test_predictions.csv")],
        [Path("outputs/final_test/final_test_metrics.json"), Path("outputs/final_test/per_review_test_metrics.csv")], [],
        seed="42", checkpoint_path=WINNER / "best_pr_auc_classifier_head.pt", env_config=STRAT / "lora_5e-5/config.json",
        checkpoint_recorded_sha="f635ed339a6902d4f2c695b27baf3ed977813c1fa361c9b61f9cf293fe44a6d1",
        verification="recomputed_from_23592_predictions", inputs=[*INPUTS, "data/processed/splits/test.csv"],
        notes="Fixed Validation-selected threshold 0.016082723; original execution log and historical Git SHA not recorded.")

    add("QWEN-FINAL-TEST", "§4.7 Complete Test Qwen comparison", "test", "scripts/run_qwen_final_test.py; src/llm_baseline/",
        [QWEN / "run_manifest.json", Path("results/qwen_final_test/run_configuration.json")],
        [QWEN / "predictions.jsonl"], [QWEN / "qwen_final_test_metrics.json", Path("results/qwen_final_test/qwen_final_test_metrics.json")],
        [QWEN / "final_test_execution_summary.json", QWEN / "progress.json"],
        git_commit="b45968fdd3eefd644083f672c6534216e3776c56", worktree="dirty",
        env_config=QWEN / "run_manifest.json",
        verification="recomputed_from_23592_decisions", inputs=["data/processed/splits/test.csv", INPUTS[-1]],
        notes="Prompt v2, model qwen3.8-flash; no immutable provider revision, raw API response bodies, or per-attempt retry log.")
    for mode in ("full", "truncated"):
        add(f"QWEN-VALIDATION-{mode.upper()}", "Teacher comment 3; paired input sensitivity not yet in thesis", "validation_5000", "scripts/run_qwen_full_input_validation_comparison.py",
            [QLEN / "experiment_config.json", QLEN / f"predictions_{mode}/run_manifest.json"],
            [QLEN / f"predictions_{mode}/predictions.jsonl"],
            [Path("results/qwen_input_length_sensitivity/metrics_summary.json"), Path("results/qwen_input_length_sensitivity/truncation_summary.json")], [],
            seed="42 (sampling)", preparation_git_commit="150c1334e5f1e35a127b246bcdbb24d971c0c7b1",
            worktree="not_recorded_for_execution; preparation_dirty",
            verification="paired_5000_predictions_and_prompt_hashes_verified",
            notes="Both use max_tokens=128 and same Prompt v2; original run script absent at preparation commit. New comparison files are local and untracked.")
    add("QWEN-FULL-TEST-SUBSET", "§4.8 Post-hoc precision-matched Qwen column", "test_5000", "scripts/run_qwen_full_test_subset.py",
        [FULL / "experiment_config.json", FULL / "predictions_qwen_full/run_manifest.json", FULL / "sample_manifest.csv"],
        [FULL / "predictions_qwen_full/predictions.jsonl"], [FULL / "comparison_evaluation.json"], [],
        seed="42 (sampling)", preparation_git_commit="150c1334e5f1e35a127b246bcdbb24d971c0c7b1",
        worktree="not_recorded_for_execution; preparation_dirty", verification="recomputed_from_5000_decisions",
        inputs=["data/processed/splits/test.csv", INPUTS[-1]])
    add("PRECISION-MATCHED", "§4.8 Post-hoc Test-subset PubMedBERT frozen/matched rows", "test_5000", "scripts/analyze_precision_matched_pubmedbert_vs_qwen_full.py",
        [FULL / "experiment_config.json", FULL / "sample_manifest.csv"],
        [FULL / "pubmedbert_predictions_subset.csv", FULL / "predictions_qwen_full/predictions.jsonl"],
        [FULL / "precision_matched_comparison/precision_matched_summary.json", FULL / "precision_matched_comparison/threshold_analysis.csv"],
        [FULL / "precision_matched_comparison/precision_matched_report.md"], seed="42 (sampling)",
        checkpoint_path=WINNER / "best_pr_auc_classifier_head.pt", verification="recomputed_from_saved_subset_probabilities",
        inputs=["data/processed/splits/test.csv", INPUTS[-1]],
        notes="Threshold 0.85556966 was chosen using Test-subset labels; exploratory in-sample result only.")
    add("ARTICLE-OVERLAP", "§4.9 Post-hoc article-disjoint sensitivity", "test/strict_subset", "scripts/analyze_article_overlap_sensitivity.py",
        [Path("config/canonical_split.json")], [Path("outputs/final_test/final_test_predictions.csv"), Path("results/article_overlap_sensitivity/test_overlap_audit.csv")],
        [Path("results/article_overlap_sensitivity/article_disjoint_metrics.json"), Path("results/article_overlap_sensitivity/overlap_summary.json")],
        [Path("results/article_overlap_sensitivity/README.md")], seed="42 (source model)",
        checkpoint_path=WINNER / "best_pr_auc_classifier_head.pt", verification="recomputed_from_frozen_predictions",
        inputs=[*INPUTS, "data/processed/splits/test.csv"])
    add("FINAL-FN-CASES", "§3 and §5 two false-negative case analysis", "test", "scripts/prepare_final_error_analysis.py",
        [Path("outputs/final_test/final_test_config.json")], [Path("outputs/final_test/final_test_predictions.csv")],
        [Path("results/final_error_analysis/false_negative_cases.csv"), Path("results/final_error_analysis/missing_abstract_context.json")],
        [Path("results/final_error_analysis/README.md")], seed="42 (source model)", verification="case_records_traced_to_final_test",
        inputs=["data/processed/splits/test.csv", INPUTS[-1]])

    # The upstream PubMedBERT weights and the four traditional-model objects are
    # model artifacts, never copied into this delivery package.
    checkpoint("PUBMEDBERT-BASE", Path("model/pubmedbert/pytorch_model.bin"), role="base_model",
               recorded="d513412d396cecec5f67936ac871b4aaf1178dfb5193776e790fd2b28bba240c")

    # Validate pooling identity and package privacy-minimized public copies.
    with actual("data/processed/splits/validation.csv").open(encoding="utf-8-sig", newline="") as stream:
        validation = list(csv.DictReader(stream))
    if len(validation) != 22397:
        raise ValueError("Unexpected Validation row count")
    for label, run in pool_runs.items():
        directory = POOL / label.lower() / run
        with actual(directory / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(validation):
            raise ValueError(f"{label}: Validation prediction count differs")
        for source, prediction in zip(validation, rows):
            if any(source[a] != prediction[b] for a, b in (
                ("review_key", "review_name"), ("label_included", "true_label"),
                ("title", "title"), ("pmid", "pmid"), ("doi", "doi"), ("openalex_id", "openalex_id"))):
                raise ValueError(f"{label}: Validation row identity or label differs")

    OUT.mkdir(parents=True, exist_ok=True)
    for label, run in pool_runs.items():
        directory = POOL / label.lower() / run
        public = OUT / "pooling_comparison" / label.lower()
        public.mkdir(parents=True, exist_ok=True)
        for filename in ("metrics.json", "training_history.csv", "per_review_validation_metrics.csv"):
            shutil.copyfile(actual(directory / filename), public / filename)
        for source_name, public_name in (("config.json", "config.public.json"),
                                         ("run_summary.json", "run_summary.public.json")):
            value = scrub(read_json(directory / source_name))
            (public / public_name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        original_log = actual(directory / "run.log").read_text(encoding="utf-8", errors="replace")
        (public / "run.public.log").write_text(scrub_string(original_log), encoding="utf-8")
        with actual(directory / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (public / "validation_predictions.public.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_row_index", "review_name", "true_label", "predicted_probability", "predicted_label"])
            for i, row in enumerate(rows):
                writer.writerow([i, row["review_name"], row["true_label"], row["predicted_probability"], row["predicted_label"]])
        source_paths = {name: rel(directory / name) for name in (
            "config.json", "validation_predictions.csv", "metrics.json", "run_summary.json",
            "run.log", "training_history.csv", "per_review_validation_metrics.csv", "best_pr_auc_classifier_head.pt")}
        public_manifest = {
            "experiment_id": f"POOL-{label}", "source_files_sha256": {name: sha256(actual(path)) for name, path in source_paths.items()},
            "source_files": source_paths,
            "public_copy_note": "Config, summary and log have paths/tokens redacted; prediction copy omits title and external identifiers but retains frozen Validation row index. Metrics, history and per-review CSVs are exact copies.",
            "validation_source_path": "data/processed/splits/validation.csv",
            "validation_source_sha256": sha256(actual("data/processed/splits/validation.csv")),
            "checkpoint_not_copied": True,
        }
        (public / "source_manifest.json").write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Preserve the already-computed Qwen pairing audit without publishing its
    # private endpoint or the analyst's local absolute paths.
    qwen_audit_source = Path("results/qwen_input_length_sensitivity/source_verification.json")
    if actual(qwen_audit_source).is_file():
        public_audit = read_json(qwen_audit_source)
        public_audit["api_configuration"]["base_url"] = "<PRIVATE_API_ENDPOINT>"
        qwen_source_paths = {
            "experiment_config": QLEN / "experiment_config.json",
            "sample_manifest": QLEN / "sample_manifest.csv",
            "sampling_summary": QLEN / "sampling_summary.json",
            "truncation_audit": QLEN / "truncation_audit.csv",
            "truncation_summary": QLEN / "truncation_summary.json",
            "full_manifest": QLEN / "predictions_full/run_manifest.json",
            "full_predictions": QLEN / "predictions_full/predictions.jsonl",
            "truncated_manifest": QLEN / "predictions_truncated/run_manifest.json",
            "truncated_predictions": QLEN / "predictions_truncated/predictions.jsonl",
            "validation_csv": Path("data/processed/splits/validation.csv"),
            "criteria_xlsx": Path(CRITERIA),
        }
        if set(public_audit["source_files"]) != set(qwen_source_paths):
            raise ValueError("Qwen audit source list has changed")
        for name, path in qwen_source_paths.items():
            if public_audit["source_files"][name]["sha256"] != sha256(actual(path)):
                raise ValueError(f"Qwen audit source hash differs: {name}")
            public_audit["source_files"][name]["path"] = rel(path)
        public_audit["original_source_path"] = rel(qwen_audit_source)
        public_audit["original_source_sha256"] = sha256(actual(qwen_audit_source))
        (OUT / "qwen_input_length_source_verification.public.json").write_text(
            json.dumps(public_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    redaction_manifest = OUT / "qwen_historical_sha256.csv"
    if not redaction_manifest.is_file():
        raise FileNotFoundError("Generate the nine Qwen public copies before refreshing the provenance index")
    with redaction_manifest.open(encoding="utf-8", newline="") as stream:
        redaction_rows = list(csv.DictReader(stream))
    if len(redaction_rows) != 9:
        raise ValueError("Expected nine Qwen historical public copies")
    historical_to_public = {}
    for item in redaction_rows:
        public = ROOT / item["public_path"]
        source = actual(item["original_path"])
        if not public.is_file() or sha256(public) != item["public_sha256"] or sha256(source) != item["original_sha256"]:
            raise ValueError(f"Qwen source/public hash mismatch: {item['original_path']}")
        historical_to_public[item["original_path"]] = item["public_path"]

    for record in records:
        replaced = {}
        for field in ("configuration_paths", "prediction_paths", "metrics_paths", "run_log_paths"):
            updated = []
            for path in record[field]:
                if path in historical_to_public:
                    public = historical_to_public[path]
                    replaced[path] = public
                    record["file_exists"].pop(path, None)
                    record["git_tracked"].pop(path, None)
                    record["file_exists"][public] = (ROOT / public).is_file()
                    record["git_tracked"][public] = False
                    updated.append(public)
                else:
                    updated.append(path)
            record[field] = updated
        record["historical_original_paths"] = replaced
        branch_paths = [*record["input_files"], *record["configuration_paths"],
                        *record["prediction_paths"], *record["metrics_paths"], *record["run_log_paths"]]
        if record["checkpoint_path"] != "not_applicable":
            branch_paths.append(record["checkpoint_path"])
        record["delivery_file_exists"] = {path: (ROOT / path).is_file() for path in dict.fromkeys(branch_paths)}
        record["entry_script_exists_in_delivery"] = {
            path.strip(): (ROOT / path.strip()).exists()
            for path in record["entry_script"].split(";")
        }
        if record["experiment_id"].startswith("POOL-"):
            pool = record["experiment_id"].split("-", 1)[1].lower()
            public_dir = Path("results/reproducibility/pooling_comparison") / pool
            record["delivery_public_alternatives"] = [
                rel(public_dir / name) for name in (
                    "config.public.json", "validation_predictions.public.csv", "metrics.json",
                    "run_summary.public.json", "run.public.log", "source_manifest.json"
                )
            ]
        else:
            record["delivery_public_alternatives"] = []

    with (OUT / "experiment_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": 2, "root": "repository-relative", "source_evidence": "local historical workspace",
                   "delivery_branch": "delivery/reproducibility-safe",
                   "historical_values_policy": "file_exists and git_tracked refer to the local historical source workspace; delivery_file_exists refers to this clean worktree; not_recorded means no historical record; missing means expected source artifact absent",
                   "experiments": records}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    columns = ["experiment_id", "thesis_result", "split", "input_files", "entry_script", "historical_entry_command_recorded",
               "git_commit", "preparation_git_commit", "run_worktree_state", "random_seed", *environment().__iter__(),
               "configuration_paths", "checkpoint_path", "checkpoint_sha256", "prediction_paths", "metrics_paths", "run_log_paths",
               "file_exists", "git_tracked", "delivery_file_exists", "entry_script_exists_in_delivery",
               "historical_original_paths", "delivery_public_alternatives",
               "verification_status", "reproducibility_status", "missing_artifacts", "notes"]
    with (OUT / "experiment_provenance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            flat = dict(record)
            flat.update(record["environment"])
            for key in ("input_files", "configuration_paths", "prediction_paths", "metrics_paths", "run_log_paths",
                        "file_exists", "git_tracked", "delivery_file_exists", "entry_script_exists_in_delivery",
                        "historical_original_paths", "delivery_public_alternatives", "missing_artifacts"):
                flat[key] = json.dumps(flat[key], ensure_ascii=False, sort_keys=True)
            flat.pop("environment")
            writer.writerow(flat)
    with (OUT / "checkpoint_sha256.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_id", "role", "path", "sha256", "recorded_sha256", "exists", "git_tracked", "delivery_exists", "size_bytes"])
        writer.writeheader()
        writer.writerows(checkpoints)
    input_paths = sorted({path for record in records for path in record["input_files"]})
    with (OUT / "input_sha256.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256", "exists", "git_tracked", "delivery_exists", "size_bytes"])
        writer.writeheader()
        for name in input_paths:
            source = actual(name)
            writer.writerow({"path": name, "sha256": sha256(source) if source.is_file() else "missing",
                             "exists": source.is_file(), "git_tracked": name in tracked,
                             "delivery_exists": (ROOT / name).is_file(),
                             "size_bytes": source.stat().st_size if source.is_file() else "missing"})
    print(json.dumps({"experiments": len(records), "model_artifact_rows": len(checkpoints), "pooling_public_copies": len(pool_runs), "output": rel(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
