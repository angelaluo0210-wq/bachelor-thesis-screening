"""Run one real CUDA batch through the frozen-encoder training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_classifier_head import (  # noqa: E402
    DEFAULT_CRITERIA_PATH,
    DEFAULT_SPLIT_DIR,
    MODEL_NAME,
    build_model,
    compute_pos_weight,
    configure_logging,
    dry_run_forward,
    join_criteria,
    load_eligibility_criteria,
    load_splits,
    parameter_report,
    set_seed,
    validate_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default=MODEL_NAME)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--criteria-path", type=Path, default=DEFAULT_CRITERIA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean", "attention", "concat"),
        default="cls",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length < 16:
        parser.error("batch-size must be positive and max-length must be at least 16")
    return args


def main() -> int:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger("train_classifier_head")
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight failed: torch.cuda.is_available() is False")

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    logger.info("GPU preflight | name=%s", torch.cuda.get_device_name(0))
    logger.info("GPU preflight | total_vram_gib=%.2f", properties.total_memory / 1024**3)
    logger.info("GPU preflight | pytorch=%s cuda=%s", torch.__version__, torch.version.cuda)
    logger.info(
        "GPU preflight settings | pooling=%s batch_size=%d max_length=%d dropout=%.3f seed=%d",
        args.pooling,
        args.batch_size,
        args.max_length,
        args.dropout,
        args.seed,
    )

    set_seed(args.seed)
    # evaluate_test=False guarantees test.csv is not opened.
    splits = load_splits(args.split_dir, evaluate_test=False)
    validate_splits(splits, evaluate_test=False)
    criteria, schema = load_eligibility_criteria(args.criteria_path)
    train = join_criteria(splits["train"], criteria, schema, "preflight train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = build_model(args.model_name_or_path, args.dropout, args.pooling)
    model.to(device)
    report = parameter_report(model)
    if not all(not parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Preflight failed: PubMedBERT backbone is not fully frozen")
    logger.info(
        "GPU preflight parameters | pooling=%s trainable=%d frozen=%d names=%s",
        args.pooling,
        report["trainable_parameters"],
        report["frozen_parameters"],
        report["trainable_parameter_names"],
    )

    pos_weight, _, _ = compute_pos_weight(train)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    try:
        dry_run_forward(model, train, tokenizer, args, criterion, device)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Preflight failed: CUDA OOM for pooling={args.pooling} "
            f"at batch_size={args.batch_size}, "
            f"max_length={args.max_length}. Re-run with --batch-size 16."
        ) from error

    logger.info(
        "GPU PREFLIGHT PASS | one forward/backward batch only | "
        "pooling=%s backbone_frozen=true test_evaluated=false",
        args.pooling,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
