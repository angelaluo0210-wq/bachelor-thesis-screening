"""Run the future Qwen Validation Pilot; this command intentionally has no Test option.

The default is a dry plan. ``--execute`` is deliberately required before any
injected API transport may be used, and must only be used in the later,
authorized AutoDL Validation Pilot after provider compatibility checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import screening_data as data
from src.llm_baseline.client import QwenClient, QwenConfig, UrllibTransport
from src.llm_baseline.prompt import PROMPT_VERSION, prompt_fingerprint
from src.llm_baseline.records import configuration_fingerprint, create_or_resume_run, sha256_file, stable_record_id
from src.llm_baseline.reconstruction import reconstruct_input
from src.llm_baseline.runner import execute_validation_records, reconstruction_audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=25,
                        help="Deterministic, label-blind Validation Pilot size.")
    parser.add_argument("--sample-seed", type=int, default=202610,
                        help="Frozen seed for deterministic record-ID sampling.")
    parser.add_argument("--execute", action="store_true", help="Reserved for the authorized AutoDL Validation Pilot.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 16 or args.max_records < 1:
        parser.error("max-length must be at least 16 and max-records must be positive")
    return args


def _git_metadata() -> dict[str, object]:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.strip())
    return {"git_commit": commit, "git_worktree_dirty": dirty}


def _tokenizer_identity(path: str) -> dict[str, object]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"A local tokenizer directory is required: {root}")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json",
             "merges.txt", "added_tokens.json")
    files = {name: sha256_file(root / name) for name in names if (root / name).is_file()}
    if not files:
        raise FileNotFoundError(f"No recognized tokenizer files found in {root}")
    return {"path": str(root), "files_sha256": files,
            "identity_sha256": configuration_fingerprint(files)}


def _select_pilot(frame: pd.DataFrame, max_records: int, sample_seed: int) -> pd.DataFrame:
    """Select a reproducible label-blind pilot from Validation rows only."""
    indexed = frame.copy().reset_index(drop=True)
    indexed["_qwen_source_row_index"] = range(len(indexed))
    ranked = []
    for index, row in indexed.iterrows():
        record_id = stable_record_id(row.to_dict(), int(row["_qwen_source_row_index"]))
        digest = hashlib.sha256(f"{sample_seed}\n{record_id}".encode("utf-8")).hexdigest()
        ranked.append((digest, index))
    selected = [index for _, index in sorted(ranked)[:max_records]]
    return indexed.iloc[selected].sort_values("_qwen_source_row_index").reset_index(drop=True)


def _prepare_validation_pilot(args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, object]:
    """Load only Train/Validation-era inputs, select pilot rows, and audit all rows."""
    splits = data.load_splits(args.split_dir, evaluate_test=False)
    data.validate_splits(splits, evaluate_test=False)
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    validation = data.join_criteria(splits["validation"], criteria, schema, "validation")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    pilot = _select_pilot(validation, args.max_records, args.sample_seed)
    audit = reconstruction_audit(pilot, tokenizer, args.max_length)
    config = {
        "experiment": "qwen3.8_flash_zero_shot", "phase": "validation_pilot", "test_access": "forbidden",
        "test_evaluated": False, "model_name_or_path": args.model_name_or_path, "max_length": args.max_length,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(), "api": QwenConfig().public_dict(),
        "input_sha256": {"train.csv": sha256_file(args.split_dir / "train.csv"),
                         "validation.csv": sha256_file(args.split_dir / "validation.csv"),
                         "criteria.xlsx": sha256_file(args.criteria_path)},
        "tokenizer": _tokenizer_identity(args.model_name_or_path),
        "runtime": {"python": platform.python_version(), "transformers": transformers.__version__},
        "source": _git_metadata(),
        "pilot_selection": {"method": "sha256(seed + stable_record_id), label_blind", "sample_seed": args.sample_seed,
                            "max_records": args.max_records, "validation_population_rows": len(validation),
                            "selected_rows": len(pilot)},
    }
    plan = {"configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
            "selected_record_ids": audit["records"], "reconstruction_audit": audit,
            "planned_utc": datetime.now(timezone.utc).isoformat()}
    return plan, pilot, tokenizer


def plan_validation_run(args: argparse.Namespace) -> dict[str, object]:
    """Public no-network planning entry point used by tests and documentation."""
    plan, _, _ = _prepare_validation_pilot(args)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, pilot, tokenizer = _prepare_validation_pilot(args)
    if not args.execute:
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing Qwen plan: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "validation_pilot_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    # Do not invoke this branch in Phase 1. It is deliberately Validation-only
    # and is retained for the subsequently authorized AutoDL pilot.
    manifest = create_or_resume_run(args.output_dir, plan["configuration"], args.resume)
    started = time.monotonic()
    (args.output_dir / "reconstruction_audit.json").write_text(
        json.dumps(plan["reconstruction_audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = execute_validation_records(pilot, tokenizer, args.max_length,
                                        QwenClient(QwenConfig(), UrllibTransport()), args.output_dir)
    (args.output_dir / "execution_summary.json").write_text(
        json.dumps({"manifest": manifest, "elapsed_seconds": time.monotonic() - started, **result},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
