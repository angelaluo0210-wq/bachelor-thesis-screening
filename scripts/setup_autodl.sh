#!/usr/bin/env bash
# Idempotent preflight and non-torch dependency setup for supplementary runs.
set -Eeuo pipefail
IFS=$'\n\t'

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/pubmedbert}"
DATA_DIR="${DATA_DIR:-$ROOT/data/processed/splits}"
CRITERIA_PATH="${CRITERIA_PATH:-$ROOT/data/eligibility_criteria(排纳标准).xlsx}"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/supplementary_202610/autodl_setup}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/setup_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "ERROR: setup failed at line $LINENO" >&2' ERR

echo "[setup] root=$ROOT"; "$PYTHON_BIN" --version; "$PYTHON_BIN" -m pip --version
command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: nvidia-smi unavailable"
df -h "$ROOT"
for p in "$ROOT/src/train_classifier_head.py" "$ROOT/config/canonical_split.json" "$DATA_DIR/train.csv" "$DATA_DIR/validation.csv" "$CRITERIA_PATH"; do [[ -f "$p" ]] || { echo "Missing required input: $p"; exit 2; }; done
for p in config.json tokenizer_config.json vocab.txt; do [[ -f "$MODEL_DIR/$p" ]] || { echo "Missing model file: $MODEL_DIR/$p"; exit 2; }; done
[[ -f "$MODEL_DIR/pytorch_model.bin" || -f "$MODEL_DIR/model.safetensors" ]] || { echo "Missing model weights"; exit 2; }

"$PYTHON_BIN" - <<'PY'
import importlib.util
for name in ("torch",):
    if importlib.util.find_spec(name) is None: raise SystemExit(f"Required preinstalled GPU dependency missing: {name}")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA is required for formal supplementary runs")
print("gpu", torch.cuda.get_device_name(0), "GB", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
PY
"$PYTHON_BIN" -m pip install --upgrade-strategy only-if-needed -r "$ROOT/requirements-autodl-supplementary.txt"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - <<'PY'
import torch, transformers, peft, accelerate, tokenizers, numpy, pandas, sklearn, openpyxl
print({"torch":torch.__version__, "cuda":torch.version.cuda, "transformers":transformers.__version__, "peft":peft.__version__, "accelerate":accelerate.__version__, "tokenizers":tokenizers.__version__, "numpy":numpy.__version__, "pandas":pandas.__version__, "sklearn":sklearn.__version__, "openpyxl":openpyxl.__version__})
PY
hash_inputs=("$ROOT/config/canonical_split.json" "$DATA_DIR/train.csv" "$DATA_DIR/validation.csv" "$CRITERIA_PATH" "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer_config.json" "$MODEL_DIR/vocab.txt")
if [[ -f "$MODEL_DIR/pytorch_model.bin" ]]; then hash_inputs+=("$MODEL_DIR/pytorch_model.bin"); fi
if [[ -f "$MODEL_DIR/model.safetensors" ]]; then hash_inputs+=("$MODEL_DIR/model.safetensors"); fi
sha256sum "${hash_inputs[@]}" | tee "$LOG_DIR/input_sha256_$(date -u +%Y%m%dT%H%M%SZ).txt"
echo "SETUP PASSED: $LOG"
