#!/usr/bin/env bash
# Default is plan-only. --execute requires a passed validation gate.
set -Eeuo pipefail
IFS=$'\n\t'
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/pubmedbert}"; CRITERIA_PATH="${CRITERIA_PATH:-$ROOT/data/eligibility_criteria(排纳标准).xlsx}"
EXECUTE=0; RESUME=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --resume) RESUME=1;; *) echo "Unknown argument: $arg"; exit 2;; esac; done
echo "Plan: Mean+LoRA weighted seeds 42,123,2026; CLS+LoRA weighted seeds 123,2026; CLS+LoRA unweighted seed 42. Test is disabled."
[[ "$EXECUTE" == 1 ]] || exit 0
VALIDATION_MANIFEST="$ROOT/supplementary_202610/autodl_validation/validation_manifest.json"
[[ -f "$VALIDATION_MANIFEST" ]] || { echo "Run validate_autodl.sh successfully first"; exit 2; }
"$PYTHON_BIN" - "$VALIDATION_MANIFEST" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") != "passed":
    raise SystemExit("Validation manifest does not record status=passed")
PY
LOG_DIR="$ROOT/supplementary_202610/autodl_orchestration"; mkdir -p "$LOG_DIR"; LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO; later experiments were not launched" >&2' ERR
run() { echo "START $(date -u +%FT%TZ): $*"; "$@"; echo "END $(date -u +%FT%TZ): $*"; }
group_state() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json, pathlib, sys
root, manifest_name, expected = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not root.exists() or not any(root.iterdir()):
    print("empty"); raise SystemExit(0)
manifests = list(root.rglob(manifest_name))
if len(manifests) != 1:
    print("ambiguous"); raise SystemExit(0)
try:
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    valid = manifest.get("status") == "completed" and len(runs) == expected
    for row in runs:
        summary = pathlib.Path(row["summary_path"])
        payload = json.loads(summary.read_text(encoding="utf-8"))
        valid &= summary.is_file() and payload.get("test_evaluated") is False
except (KeyError, OSError, ValueError, TypeError):
    valid = False
print("complete" if valid else "ambiguous")
PY
}
run_or_resume() {
  local label="$1" root="$2" manifest="$3" expected="$4"; shift 4
  local state
  state="$(group_state "$root" "$manifest" "$expected")"
  if [[ "$state" == "complete" ]]; then
    if [[ "$RESUME" == 1 ]]; then echo "SKIP verified completed group: $label"; return 0; fi
    echo "Existing completed group found for $label; refuse to create another. Use --resume to skip it."; return 2
  fi
  if [[ "$state" != "empty" ]]; then
    echo "Ambiguous or partial artifacts for $label at $root; inspect manually. Refusing overwrite."; return 2
  fi
  run "$@"
}
if [[ "$RESUME" == 1 ]]; then echo "Resume skips only one fully verified group; exact optimizer/RNG checkpoint resume is unsupported."; fi
run_or_resume "Mean+LoRA weighted" "$ROOT/supplementary_202610/mean_lora" mean_lora_manifest.json 3 "$PYTHON_BIN" scripts/run_mean_lora_supplementary.py --execute --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root supplementary_202610/mean_lora
run_or_resume "CLS+LoRA weighted multi-seed" "$ROOT/supplementary_202610/cls_lora_multiseed" manifest.json 2 "$PYTHON_BIN" scripts/run_cls_lora_supplementary.py multiseed --execute --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root supplementary_202610
run_or_resume "CLS+LoRA unweighted BCE" "$ROOT/supplementary_202610/cls_lora_unweighted_bce" manifest.json 1 "$PYTHON_BIN" scripts/run_cls_lora_supplementary.py unweighted-ablation --execute --model-name-or-path "$MODEL_DIR" --criteria-path "$CRITERIA_PATH" --output-root supplementary_202610
echo "ALL SIX RUNS COMPLETED; retain complete supplementary_202610 tree for backup."
