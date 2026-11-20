"""Create clearly labeled public copies of historical Qwen JSON records.

The source directory is a local historical workspace. No source file is changed.
This script does not issue API requests or recalculate experiment metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "results/reproducibility/qwen_historical_public"
ORIGINALS = (
    "outputs/pubmedbert_vs_qwen_full_test_subset/experiment_config.json",
    "outputs/pubmedbert_vs_qwen_full_test_subset/predictions_qwen_full/run_manifest.json",
    "outputs/qwen_full_input_validation_comparison/experiment_config.json",
    "outputs/qwen_full_input_validation_comparison/predictions_full/run_manifest.json",
    "outputs/qwen_full_input_validation_comparison/predictions_truncated/run_manifest.json",
    "outputs/qwen_full_input_validation_comparison/technical_validation_full/run_manifest.json",
    "outputs/qwen_full_input_validation_comparison/technical_validation_truncated/run_manifest.json",
    "supplementary_202610/llm_baseline/qwen_final_test_20261029/final_test_execution_summary.json",
    "supplementary_202610/llm_baseline/qwen_final_test_20261029/run_manifest.json",
)
ENDPOINT_RE = re.compile(r"https://[^\s\"']+\.maas\.aliyuncs\.com/compatible-mode/v1", re.I)
WINDOWS_PROJECT_RE = re.compile(r"(?i)[A-Z]:\\Users\\[^\\]+\\Desktop\\[^\\]+")
WINDOWS_HOME_RE = re.compile(r"(?i)[A-Z]:\\Users\\[^\\]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(original: str) -> Path:
    path = Path(original)
    # Avoid an `outputs/` directory inside the public package: the repository
    # intentionally ignores every directory with that name.
    top = "archived_outputs" if path.parts[0] == "outputs" else "archived_supplementary"
    return PUBLIC_ROOT / top / Path(*path.parts[1:]).with_name(path.stem + ".public.json")


def redact(value, counts: dict[str, int]):
    if isinstance(value, str):
        value, n = ENDPOINT_RE.subn("<HISTORICAL_WORKSPACE_ENDPOINT_REDACTED>", value)
        counts["endpoint"] += n
        value, n = WINDOWS_PROJECT_RE.subn("<LOCAL_PROJECT_ROOT>", value)
        counts["windows_project_path"] += n
        value, n = WINDOWS_HOME_RE.subn("<LOCAL_USER_HOME>", value)
        counts["windows_home_path"] += n
        return value
    if isinstance(value, list):
        return [redact(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, counts) for key, item in value.items()}
    return value


def refresh_public_fingerprints(value, original_fingerprints: list[str]) -> None:
    if isinstance(value, list):
        for item in value:
            refresh_public_fingerprints(item, original_fingerprints)
    elif isinstance(value, dict):
        if "configuration" in value and "configuration_fingerprint" in value:
            original_fingerprints.append(value["configuration_fingerprint"])
            canonical = json.dumps(value["configuration"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            value["configuration_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        for item in value.values():
            refresh_public_fingerprints(item, original_fingerprints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if source_root == PROJECT_ROOT:
        raise ValueError("Public copies must be built from a separate local historical workspace")
    rows = []
    endpoint_values: set[str] = set()
    for original in ORIGINALS:
        source = source_root / original
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = json.loads(source.read_text(encoding="utf-8"))
        original_text = source.read_text(encoding="utf-8")
        endpoints = ENDPOINT_RE.findall(original_text)
        if len(endpoints) != 1:
            raise ValueError(f"Expected exactly one endpoint in {original}")
        endpoint_values.update(endpoints)
        counts = {"endpoint": 0, "windows_project_path": 0, "windows_home_path": 0}
        public = redact(raw, counts)
        fingerprints: list[str] = []
        refresh_public_fingerprints(public, fingerprints)
        public["_public_copy"] = {
            "is_original_run_record": False,
            "original_repository_path": original,
            "original_sha256": sha256(source),
            "historical_endpoint_sha256": hashlib.sha256(endpoints[0].encode("utf-8")).hexdigest(),
            "original_configuration_fingerprints": fingerprints,
            "redactions": counts,
        }
        target = public_path(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if ENDPOINT_RE.search(target.read_text(encoding="utf-8")) or WINDOWS_HOME_RE.search(target.read_text(encoding="utf-8")):
            raise ValueError(f"Sensitive value remains in {target}")
        rows.append({"original_path": original, "original_sha256": sha256(source),
                     "public_path": target.relative_to(PROJECT_ROOT).as_posix(),
                     "public_sha256": sha256(target), "endpoint_redactions": counts["endpoint"],
                     "historical_endpoint_sha256": hashlib.sha256(endpoints[0].encode("utf-8")).hexdigest(),
                     "windows_path_redactions": counts["windows_project_path"] + counts["windows_home_path"],
                     "original_is_local_only": True})
    if len(endpoint_values) != 1:
        raise ValueError("Historical JSON files did not use the same endpoint")
    manifest = PROJECT_ROOT / "results/reproducibility/qwen_historical_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"public_copies": len(rows), "one_historical_endpoint": True,
                      "sha256_manifest": manifest.relative_to(PROJECT_ROOT).as_posix()}))


if __name__ == "__main__":
    main()
