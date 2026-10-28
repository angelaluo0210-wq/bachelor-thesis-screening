#!/usr/bin/env bash
# Dynamic gate: tests and disposable real forward/backward smoke runs only.
set -Eeuo pipefail
IFS=$'\n\t'
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/pubmedbert}"
CRITERIA_PATH="${CRITERIA_PATH:-$ROOT/data/eligibility_criteria(排纳标准).xlsx}"
OUT="$ROOT/supplementary_202610/autodl_validation"
SMOKE_ROOT="$OUT/smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT" "$SMOKE_ROOT"
LOG="$OUT/validate_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "VALIDATION FAILED at line $LINENO; smoke artifacts retained: $SMOKE_ROOT" >&2' ERR
cd "$ROOT"
"$PYTHON_BIN" -m pytest -q tests/test_training_strategies.py tests/test_phase3_supplementary_static.py
"$PYTHON_BIN" scripts/run_mean_lora_supplementary.py --execute --smoke --seeds 42 --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root "$SMOKE_ROOT/mean"
"$PYTHON_BIN" scripts/run_cls_lora_supplementary.py multiseed --execute --smoke --seeds 123 --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root "$SMOKE_ROOT/cls"
"$PYTHON_BIN" scripts/run_cls_lora_supplementary.py unweighted-ablation --execute --smoke --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root "$SMOKE_ROOT/ablation"
"$PYTHON_BIN" - <<PY
import json, pathlib
root=pathlib.Path(r"$SMOKE_ROOT")
summaries=list(root.rglob("run_summary.json"))
assert len(summaries)==3, f"expected 3 smoke summaries, got {len(summaries)}"
for p in summaries:
 s=json.loads(p.read_text()); assert not s["test_evaluated"]; assert s["checkpoint_roundtrip"]["passed"]; assert s["parameter_update_check"]["frozen_parameters_unchanged"]
print("Smoke artifact validation passed", [str(p) for p in summaries])
PY
printf '{"status":"passed","smoke_root":"%s","log":"%s"}\n' "$SMOKE_ROOT" "$LOG" > "$OUT/validation_manifest.json"
echo "VALIDATION PASSED"
