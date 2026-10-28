"""Train a PubMedBERT screening classifier with head-only, upper-layer or LoRA adaptation.

The existing review-level splits are treated as immutable. During ordinary
training only train.csv and validation.csv are opened. Pass ``--evaluate-test``
only for a final, explicitly requested test evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import transformers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = ROOT / "data" / "processed" / "splits"
DEFAULT_CRITERIA_PATH = ROOT / "data" / "eligibility_criteria(排纳标准).xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "classifier_head"
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

REVIEW_COLUMN = "review_key"
TITLE_COLUMN = "title"
ABSTRACT_COLUMN = "abstract"
LABEL_COLUMN = "label_included"
IDENTIFIER_COLUMNS = ("pmid", "doi", "openalex_id")
VALID_LABELS = {0, 1}

LOGGER = logging.getLogger("train_classifier_head")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    add_strategy_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--criteria-path", type=Path, default=DEFAULT_CRITERIA_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--class-weighting", choices=("weighted", "unweighted"), default="weighted",
        help="weighted preserves the historical train-negative/train-positive BCE weighting.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=MODEL_NAME,
        help="Hugging Face checkpoint name or a local model/tokenizer directory.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of silently training on CPU/MPS (recommended for formal runs).",
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean", "attention", "concat"),
        default="cls",
        help="Pooling strategy applied to the frozen encoder's final hidden states.",
    )
    parser.add_argument("--evaluate-test", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Deterministically stratify-sample train rows for a smoke test.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Deterministically stratify-sample validation rows for a smoke test.",
    )
    args = parser.parse_args(argv)
    validate_strategy_args(args, parser)

    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        parser.error("learning-rate, batch-size, and epochs must be positive")
    if args.max_length < 16:
        parser.error("max-length must be at least 16")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    for option in ("max_train_samples", "max_validation_samples"):
        value = getattr(args, option)
        if value is not None and value < 2:
            parser.error(f"{option.replace('_', '-')} must be at least 2 to retain both classes")
    smoke_test = args.max_train_samples is not None or args.max_validation_samples is not None
    if smoke_test and args.evaluate_test:
        parser.error("--evaluate-test cannot be used in smoke test mode")
    return args


STRATEGIES = ("classifier_head_only", "partial_unfreezing", "lora")


def add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-strategy", choices=STRATEGIES, default=STRATEGIES[0])
    parser.add_argument("--head-learning-rate", type=float, default=None,
                        help="Defaults to --learning-rate, preserving legacy head-only runs.")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--adapter-learning-rate", type=float, default=5e-5)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--comparison-mode", action="store_true",
                        help="Enforce CLS/threshold=0.5 and prohibit test evaluation.")


def validate_strategy_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("head_learning_rate", "backbone_learning_rate", "adapter_learning_rate"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.unfreeze_last_n < 1 or args.lora_r < 1 or args.lora_alpha < 1:
        parser.error("unfreeze-last-n, lora-r and lora-alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("lora-dropout must be in [0, 1)")
    if sorted(args.lora_target_modules) != ["query", "value"]:
        parser.error("This experiment supports exactly query value LoRA targets")
    # The historical partial-unfreezing comparison remains CLS-only.  LoRA may
    # use the already-established frozen-encoder mean pooling outside the
    # locked comparison protocol (e.g. supplementary pooling ablations).
    if args.training_strategy == "partial_unfreezing" and args.pooling != "cls":
        parser.error("Partial unfreezing requires CLS pooling")
    if args.comparison_mode:
        if args.evaluate_test or args.pooling != "cls" or args.threshold != 0.5:
            parser.error("Comparison requires CLS, threshold=0.5 and test disabled")
        if args.class_weighting != "weighted":
            parser.error("Comparison requires the historical weighted BCE")
        if (args.unfreeze_last_n, args.lora_r, args.lora_alpha, args.lora_dropout) != (2, 8, 16, 0.05):
            parser.error("Comparison fixes last_n=2, r=8, alpha=16, LoRA dropout=0.05")


def strategy_options(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in (
        "training_strategy", "unfreeze_last_n", "lora_r", "lora_alpha",
        "lora_dropout", "lora_target_modules",
    )}


def configure_logging(run_dir: Path | None = None) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if run_dir is not None:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_text(value: Any) -> str:
    """Convert a possibly missing dataframe value to tokenizer-safe text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonicalize_column(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def normalize_review_key(value: Any) -> str:
    text = safe_text(value).lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def read_split(path: Path, review_keys_only: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    usecols = [REVIEW_COLUMN] if review_keys_only else None
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    if REVIEW_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing required column {REVIEW_COLUMN!r}")
    if review_keys_only:
        return frame

    required = {REVIEW_COLUMN, TITLE_COLUMN, ABSTRACT_COLUMN, LABEL_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def load_splits(split_dir: Path, evaluate_test: bool) -> dict[str, pd.DataFrame]:
    splits = {
        "train": read_split(split_dir / "train.csv"),
        "validation": read_split(split_dir / "validation.csv"),
    }
    # Do not even open test.csv during tuning/model selection.
    if evaluate_test:
        splits["test"] = read_split(split_dir / "test.csv")
    return splits


@dataclass(frozen=True)
class CriteriaSchema:
    sheet_name: str
    review_key: str
    inclusion: str | None
    exclusion: str | None
    eligibility: str | None
    review_text_columns: tuple[str, ...]


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def load_eligibility_criteria(path: Path) -> tuple[pd.DataFrame, CriteriaSchema]:
    """Inspect the workbook and load the sheet containing its review key."""
    if not path.exists():
        raise FileNotFoundError(f"Eligibility criteria workbook not found: {path}")

    workbook = pd.ExcelFile(path)
    candidates: list[tuple[str, pd.DataFrame]] = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.rename(columns={column: canonicalize_column(column) for column in frame.columns})
        if _first_present(frame.columns, ("review_key", "review_id")):
            candidates.append((sheet_name, frame))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(
            "Expected exactly one eligibility sheet with review_key/review_id; "
            f"found {names} among sheets {workbook.sheet_names}"
        )

    sheet_name, criteria = candidates[0]
    key_column = _first_present(criteria.columns, ("review_key", "review_id"))
    assert key_column is not None
    inclusion = _first_present(
        criteria.columns,
        ("inclusion_criteria", "inclusion_criterion", "inclusion"),
    )
    exclusion = _first_present(
        criteria.columns,
        ("exclusion_criteria", "exclusion_criterion", "exclusion"),
    )
    eligibility = _first_present(
        criteria.columns,
        ("eligibility_criteria", "eligibility_criterion", "eligibility", "criteria"),
    )
    if not any((inclusion, exclusion, eligibility)):
        raise ValueError(f"No criteria text columns found in workbook sheet {sheet_name!r}")

    review_text_aliases = (
        "review_title",
        "review_question",
        "review_description",
        "systematic_review_title",
    )
    review_text_columns = tuple(column for column in review_text_aliases if column in criteria.columns)
    criteria = criteria.copy()
    criteria["_normalized_review_key"] = criteria[key_column].map(normalize_review_key)
    if criteria["_normalized_review_key"].eq("").any():
        raise ValueError("Eligibility workbook contains an empty review key")
    duplicated = criteria.loc[
        criteria["_normalized_review_key"].duplicated(keep=False), key_column
    ].astype(str).tolist()
    if duplicated:
        raise ValueError(f"Eligibility workbook has duplicate normalized review keys: {duplicated}")

    schema = CriteriaSchema(
        sheet_name=sheet_name,
        review_key=key_column,
        inclusion=inclusion,
        exclusion=exclusion,
        eligibility=eligibility,
        review_text_columns=review_text_columns,
    )
    return criteria, schema


def join_criteria(
    frame: pd.DataFrame,
    criteria: pd.DataFrame,
    schema: CriteriaSchema,
    split_name: str,
) -> pd.DataFrame:
    """Join existing criteria without generating or paraphrasing their content."""
    result = frame.copy()
    result["_normalized_review_key"] = result[REVIEW_COLUMN].map(normalize_review_key)
    lookup = criteria.set_index("_normalized_review_key", drop=False)
    matched = result["_normalized_review_key"].isin(lookup.index)
    unmatched_ids = sorted(result.loc[~matched, REVIEW_COLUMN].dropna().astype(str).unique())
    LOGGER.info(
        "%s criteria QC | examples=%d matched=%d unmatched=%d",
        split_name,
        len(result),
        int(matched.sum()),
        int((~matched).sum()),
    )
    if unmatched_ids:
        LOGGER.error("%s unmatched review IDs: %s", split_name, unmatched_ids)
        raise ValueError(f"Unmatched eligibility criteria in {split_name}: {unmatched_ids}")

    def mapped_text(column: str | None) -> pd.Series:
        if column is None:
            return pd.Series("", index=result.index, dtype="string")
        mapping = lookup[column].map(safe_text)
        return result["_normalized_review_key"].map(mapping).fillna("").map(safe_text)

    result["_inclusion_text"] = mapped_text(schema.inclusion)
    result["_exclusion_text"] = mapped_text(schema.exclusion)
    unified = mapped_text(schema.eligibility)
    # A workbook may contain only unified eligibility prose. Preserve it verbatim
    # in the inclusion section rather than inventing an inclusion/exclusion split.
    neither_split = result["_inclusion_text"].eq("") & result["_exclusion_text"].eq("")
    result.loc[neither_split, "_inclusion_text"] = unified.loc[neither_split]

    result["_review_text"] = ""
    for column in schema.review_text_columns:
        values = mapped_text(column)
        result["_review_text"] = [
            "\n\n".join(part for part in (left, right) if part)
            for left, right in zip(result["_review_text"], values)
        ]

    no_criteria = result["_inclusion_text"].eq("") & result["_exclusion_text"].eq("")
    if no_criteria.any():
        ids = sorted(result.loc[no_criteria, REVIEW_COLUMN].astype(str).unique())
        raise ValueError(f"Matched reviews have no usable criteria text in {split_name}: {ids}")
    return result


def validate_labels(frame: pd.DataFrame, split_name: str) -> None:
    numeric = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    invalid = frame.loc[numeric.isna() | ~numeric.isin(VALID_LABELS), LABEL_COLUMN].unique().tolist()
    if invalid:
        raise ValueError(f"{split_name} contains labels outside {{0, 1}}: {invalid}")
    frame[LABEL_COLUMN] = numeric.astype(np.int64)
    present = set(frame[LABEL_COLUMN].unique().tolist())
    if present != VALID_LABELS:
        raise ValueError(f"{split_name} must contain both classes; found {sorted(present)}")


def summarize_split(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    labels = frame[LABEL_COLUMN]
    missing_title = frame[TITLE_COLUMN].isna() | frame[TITLE_COLUMN].astype(str).str.strip().eq("")
    missing_abstract = frame[ABSTRACT_COLUMN].isna() | frame[ABSTRACT_COLUMN].astype(str).str.strip().eq("")
    summary = {
        "split": split_name,
        "reviews": int(frame[REVIEW_COLUMN].nunique()),
        "rows": int(len(frame)),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "positive_rate": float((labels == 1).mean()),
        "missing_title": int(missing_title.sum()),
        "missing_abstract": int(missing_abstract.sum()),
        "missing_abstract_positive": int((missing_abstract & labels.eq(1)).sum()),
        "missing_abstract_negative": int((missing_abstract & labels.eq(0)).sum()),
    }
    LOGGER.info("%s data QC | %s", split_name, json.dumps(summary, ensure_ascii=False))
    return summary


def validate_splits(splits: dict[str, pd.DataFrame], evaluate_test: bool) -> None:
    review_sets = {
        name: set(frame[REVIEW_COLUMN].dropna().astype(str))
        for name, frame in splits.items()
    }
    pairs = [("train", "validation")]
    if evaluate_test:
        pairs.extend((("train", "test"), ("validation", "test")))
    for left, right in pairs:
        overlap = sorted(review_sets[left] & review_sets[right])
        if overlap:
            raise ValueError(f"Review leakage between {left} and {right}: {overlap}")
    checked = "train/validation/test" if evaluate_test else "train/validation"
    LOGGER.info("Review-level leakage QC: PASS (%s are disjoint)", checked)
    validate_labels(splits["train"], "train")
    validate_labels(splits["validation"], "validation")
    if evaluate_test:
        validate_labels(splits["test"], "test")


def stratified_subsample(
    frame: pd.DataFrame,
    max_samples: int | None,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    """Return an exact-size deterministic binary stratified subset.

    Sampling occurs only within the supplied existing split. At least one row
    from each class is retained, and the requested size is never silently
    increased. If the request covers the whole split, a copy is returned.
    """
    if max_samples is None or max_samples >= len(frame):
        return frame.copy()
    if max_samples < 2:
        raise ValueError(
            f"{split_name} smoke subset needs at least 2 rows to contain labels 0 and 1"
        )

    positive = frame[frame[LABEL_COLUMN] == 1]
    negative = frame[frame[LABEL_COLUMN] == 0]
    if positive.empty or negative.empty:
        raise ValueError(f"{split_name} cannot be stratified because one class is absent")

    positive_target = round(max_samples * len(positive) / len(frame))
    positive_target = min(max(positive_target, 1), len(positive), max_samples - 1)
    negative_target = max_samples - positive_target
    if negative_target > len(negative):
        negative_target = len(negative)
        positive_target = max_samples - negative_target
    if positive_target > len(positive) or positive_target < 1 or negative_target < 1:
        raise ValueError(
            f"Cannot draw {max_samples} stratified rows from {split_name}: "
            f"available positive={len(positive)}, negative={len(negative)}"
        )

    sampled_positive = positive.sample(n=positive_target, random_state=seed)
    sampled_negative = negative.sample(n=negative_target, random_state=seed + 1)
    subset = pd.concat([sampled_positive, sampled_negative], axis=0)
    # Deterministically mix class blocks without changing split membership.
    return subset.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def build_input_text(row: pd.Series | dict[str, Any]) -> str:
    """Build readable input; the bracketed labels remain ordinary tokenizer text."""
    sections: list[str] = []
    review_text = safe_text(row.get("_review_text", ""))
    if review_text:
        sections.append(f"[REVIEW]\n{review_text}")
    sections.extend(
        (
            f"[INCLUSION]\n{safe_text(row.get('_inclusion_text', ''))}",
            f"[EXCLUSION]\n{safe_text(row.get('_exclusion_text', ''))}",
            f"[TITLE]\n{safe_text(row.get(TITLE_COLUMN, ''))}",
            f"[ABSTRACT]\n{safe_text(row.get(ABSTRACT_COLUMN, ''))}",
        )
    )
    return "\n\n".join(sections)


def _input_segments(row: dict[str, Any]) -> tuple[str, str, str]:
    review = safe_text(row.get("_review_text", ""))
    criteria_parts = []
    if review:
        criteria_parts.append(f"[REVIEW]\n{review}")
    criteria_parts.extend(
        (
            f"[INCLUSION]\n{safe_text(row.get('_inclusion_text', ''))}",
            f"[EXCLUSION]\n{safe_text(row.get('_exclusion_text', ''))}",
        )
    )
    return (
        "\n\n".join(criteria_parts),
        f"[TITLE]\n{safe_text(row.get(TITLE_COLUMN, ''))}",
        f"[ABSTRACT]\n{safe_text(row.get(ABSTRACT_COLUMN, ''))}",
    )


def tokenize_with_section_budget(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize with separate, deterministic budgets for all four sections.

    Budgets include each ordinary-text section label. Title receives at most
    64 tokens, inclusion and exclusion first receive up to 128 tokens each,
    and abstract receives the remaining capacity. If abstract is shorter than
    that capacity, unused tokens are redistributed fairly between truncated
    inclusion and exclusion sections. Every section uses prefix truncation.
    """
    review_text = safe_text(row.get("_review_text", ""))
    inclusion_parts = []
    if review_text:
        # Preserve the existing optional review-text compatibility and order;
        # current project data leaves this field empty.
        inclusion_parts.append(f"[REVIEW]\n{review_text}")
    inclusion_parts.append(f"[INCLUSION]\n{safe_text(row.get('_inclusion_text', ''))}")
    inclusion_text = "\n\n".join(inclusion_parts)
    exclusion_text = f"[EXCLUSION]\n{safe_text(row.get('_exclusion_text', ''))}"
    title_text = f"[TITLE]\n{safe_text(row.get(TITLE_COLUMN, ''))}"
    abstract_text = f"[ABSTRACT]\n{safe_text(row.get(ABSTRACT_COLUMN, ''))}"

    inclusion_ids = tokenizer.encode(inclusion_text, add_special_tokens=False)
    exclusion_ids = tokenizer.encode(exclusion_text, add_special_tokens=False)
    title_ids = tokenizer.encode(title_text, add_special_tokens=False)
    abstract_ids = tokenizer.encode(abstract_text, add_special_tokens=False)

    # Transformers 5.x no longer exposes ``prepare_for_model`` on every slow
    # tokenizer class.  This is a single-sequence BERT input, so construct the
    # canonical [CLS] content [SEP] sequence explicitly.  max_length includes
    # both special tokens and any right padding.
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    if cls_token_id is None or sep_token_id is None or pad_token_id is None:
        raise ValueError(
            "BERT tokenizer must define cls_token_id, sep_token_id, and pad_token_id"
        )

    special_token_count = 2
    usable = max_length - special_token_count
    if usable <= 0:
        raise ValueError(f"max_length={max_length} leaves no content-token budget")

    # Guaranteed/base allocation. For the normal max_length=512 case these
    # caps leave 190 tokens for abstract when all three base budgets are full.
    title_budget = min(len(title_ids), min(64, usable))
    capacity_after_title = usable - title_budget
    inclusion_budget = min(len(inclusion_ids), min(128, capacity_after_title))
    capacity_after_inclusion = capacity_after_title - inclusion_budget
    exclusion_budget = min(len(exclusion_ids), min(128, capacity_after_inclusion))

    remaining = usable - title_budget - inclusion_budget - exclusion_budget
    abstract_budget = min(len(abstract_ids), remaining)
    remaining -= abstract_budget

    # If abstract is short, expand truncated inclusion/exclusion fairly. This
    # water-filling allocation gives equal extras while both need them; once
    # one is complete, the other receives the rest. An unavoidable odd token
    # goes to the section with the greater remaining need.
    inclusion_need = len(inclusion_ids) - inclusion_budget
    exclusion_need = len(exclusion_ids) - exclusion_budget
    if remaining > 0 and inclusion_need > 0 and exclusion_need > 0:
        equal_extra = min(inclusion_need, exclusion_need, remaining // 2)
        inclusion_budget += equal_extra
        exclusion_budget += equal_extra
        inclusion_need -= equal_extra
        exclusion_need -= equal_extra
        remaining -= 2 * equal_extra

    if remaining > 0 and inclusion_need > 0 and exclusion_need > 0:
        if inclusion_need >= exclusion_need:
            inclusion_budget += 1
            inclusion_need -= 1
        else:
            exclusion_budget += 1
            exclusion_need -= 1
        remaining -= 1

    if remaining > 0 and inclusion_need > 0:
        extra = min(inclusion_need, remaining)
        inclusion_budget += extra
        inclusion_need -= extra
        remaining -= extra
    if remaining > 0 and exclusion_need > 0:
        extra = min(exclusion_need, remaining)
        exclusion_budget += extra
        exclusion_need -= extra
        remaining -= extra

    kept_inclusion = inclusion_ids[:inclusion_budget]
    kept_exclusion = exclusion_ids[:exclusion_budget]
    kept_title = title_ids[:title_budget]
    kept_abstract = abstract_ids[:abstract_budget]
    content_ids = kept_inclusion + kept_exclusion + kept_title + kept_abstract

    # Build section masks from the known post-truncation segment lengths. This
    # is deterministic and avoids trying to rediscover multi-token text markers
    # such as [INCLUSION] in the final input IDs.
    criteria_length = len(kept_inclusion) + len(kept_exclusion)
    article_length = len(kept_title) + len(kept_abstract)
    criteria_mask = [0] + [1] * criteria_length + [0] * article_length + [0]
    article_mask = [0] + [0] * criteria_length + [1] * article_length + [0]

    # The segment budgets above already guarantee len(content_ids) <= usable.
    # Add each BERT special token exactly once, then right-pad to max_length.
    input_ids = [cls_token_id, *content_ids, sep_token_id]
    if len(input_ids) > max_length:
        raise RuntimeError(
            f"Section-budget tokenization produced {len(input_ids)} tokens "
            f"for max_length={max_length}"
        )
    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)
    input_ids.extend([pad_token_id] * padding_length)
    attention_mask.extend([0] * padding_length)
    criteria_mask.extend([0] * padding_length)
    article_mask.extend([0] * padding_length)

    if not all(
        len(values) == max_length
        for values in (input_ids, attention_mask, criteria_mask, article_mask)
    ):
        raise RuntimeError("Token IDs and all masks must be padded to max_length")
    if input_ids.count(cls_token_id) != 1 or input_ids.count(sep_token_id) != 1:
        raise RuntimeError("Expected exactly one [CLS] token and one [SEP] token")
    last_non_padding = sum(attention_mask) - 1
    if input_ids[last_non_padding] != sep_token_id:
        raise RuntimeError("The final non-padding token must be [SEP]")
    if not all(attention_mask[: last_non_padding + 1]) or any(
        attention_mask[last_non_padding + 1 :]
    ):
        raise RuntimeError("Attention mask does not match right padding")
    if any(left and right for left, right in zip(criteria_mask, article_mask)):
        raise RuntimeError("Criteria and article masks overlap")
    if any(section and not valid for section, valid in zip(criteria_mask, attention_mask)):
        raise RuntimeError("Criteria mask includes padding or special-token positions")
    if any(section and not valid for section, valid in zip(article_mask, attention_mask)):
        raise RuntimeError("Article mask includes padding or special-token positions")
    if criteria_mask[0] or article_mask[0] or criteria_mask[last_non_padding] or article_mask[last_non_padding]:
        raise RuntimeError("[CLS] and [SEP] must be excluded from section masks")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "criteria_mask": criteria_mask,
        "article_mask": article_mask,
    }


class ScreeningDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "row_index": index,
            REVIEW_COLUMN: safe_text(row[REVIEW_COLUMN]),
            TITLE_COLUMN: safe_text(row[TITLE_COLUMN]),
            ABSTRACT_COLUMN: safe_text(row[ABSTRACT_COLUMN]),
            LABEL_COLUMN: int(row[LABEL_COLUMN]),
            "_inclusion_text": safe_text(row["_inclusion_text"]),
            "_exclusion_text": safe_text(row["_exclusion_text"]),
            "_review_text": safe_text(row["_review_text"]),
        }


class BudgetedCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encodings = [tokenize_with_section_budget(self.tokenizer, row, self.max_length) for row in rows]
        batch = {
            "input_ids": torch.tensor([item["input_ids"] for item in encodings], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in encodings], dtype=torch.long),
            "criteria_mask": torch.tensor([item["criteria_mask"] for item in encodings], dtype=torch.long),
            "article_mask": torch.tensor([item["article_mask"] for item in encodings], dtype=torch.long),
            "labels": torch.tensor([row[LABEL_COLUMN] for row in rows], dtype=torch.float32),
            "row_indices": torch.tensor([row["row_index"] for row in rows], dtype=torch.long),
        }
        empty_criteria = batch["criteria_mask"].sum(dim=1).eq(0)
        empty_article = batch["article_mask"].sum(dim=1).eq(0)
        if empty_criteria.any() or empty_article.any():
            raise ValueError(
                "Empty section mask(s) in batch: "
                f"criteria={int(empty_criteria.sum())}, article={int(empty_article.sum())}"
            )
        return batch


def report_truncation(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    split_name: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    exceeded = 0
    lengths: list[int] = []
    for start in tqdm(range(0, len(frame), batch_size), desc=f"Token length QC ({split_name})"):
        texts = [build_input_text(row) for _, row in frame.iloc[start : start + batch_size].iterrows()]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        lengths.extend(batch_lengths)
        exceeded += sum(length > max_length for length in batch_lengths)
    result = {
        "examples": len(frame),
        "exceeding_max_length": int(exceeded),
        "percentage_exceeding": float(100.0 * exceeded / len(frame)) if len(frame) else 0.0,
        "max_observed_tokens": max(lengths, default=0),
    }
    LOGGER.info("%s truncation QC | %s", split_name, json.dumps(result))
    return result


class FrozenPubMedBertClassifier(nn.Module):
    """Legacy-compatible name; strategy-aware encoder plus unchanged pooling/head."""

    VALID_POOLING = {"cls", "mean", "attention", "concat"}

    def __init__(self, model_name: str, dropout: float, pooling_type: str = "cls",
                 training_strategy: str = "classifier_head_only", unfreeze_last_n: int = 2,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_target_modules: Sequence[str] = ("query", "value"),
                 model_revision: str | None = None) -> None:
        super().__init__()
        if pooling_type not in self.VALID_POOLING:
            raise ValueError(f"Unsupported pooling type: {pooling_type!r}")
        self.model_name = model_name
        self.pooling_type = pooling_type
        if training_strategy not in STRATEGIES:
            raise ValueError(f"Unsupported training strategy: {training_strategy}")
        if training_strategy == "partial_unfreezing" and pooling_type != "cls":
            raise ValueError("Partial unfreezing requires CLS pooling")
        self.training_strategy = training_strategy
        self.strategy_config = dict(training_strategy=training_strategy,
                                    unfreeze_last_n=unfreeze_last_n, lora_r=lora_r,
                                    lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                    lora_target_modules=list(lora_target_modules))
        self.adapted_layer_indices: list[int] = []
        self.lora_module_names: list[str] = []
        self.encoder = AutoModel.from_pretrained(model_name, **({"revision": model_revision} if model_revision else {}))
        self.model_revision = getattr(self.encoder.config, "_commit_hash", None)
        self.freeze_encoder()
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.pooling_output_dim = self.hidden_size * 3 if pooling_type == "concat" else self.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Keep classifier construction in the same position for the CLS path.
        self.classifier = nn.Linear(self.pooling_output_dim, 1)
        self.attention_scorer = (
            # A scalar-score bias would cancel inside token-wise softmax and is
            # therefore intentionally omitted as an unidentifiable parameter.
            nn.Linear(self.hidden_size, 1, bias=False)
            if pooling_type == "attention"
            else None
        )
        # Construct the head BEFORE injecting adapters: PEFT initialization must
        # not change the head's random initialization relative to the legacy A.
        if training_strategy == "partial_unfreezing":
            layers = self.encoder.encoder.layer
            if not 1 <= unfreeze_last_n <= len(layers):
                raise ValueError("unfreeze_last_n exceeds the encoder layer count")
            self.adapted_layer_indices = list(range(len(layers) - unfreeze_last_n, len(layers)))
            for index in self.adapted_layer_indices:
                layers[index].requires_grad_(True)
        elif training_strategy == "lora":
            if sorted(lora_target_modules) != ["query", "value"]:
                raise ValueError("LoRA must target exactly query and value")
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as error:
                raise ImportError("LoRA requires the PEFT version in requirements.txt") from error
            targets = [f"encoder.layer.{i}.attention.self.{name}"
                       for i in range(len(self.encoder.encoder.layer))
                       for name in ("query", "value")]
            modules = dict(self.encoder.named_modules())
            if not all(isinstance(modules.get(name), nn.Linear) for name in targets):
                raise ValueError("Expected BERT attention query/value Linear modules")
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=targets, bias="none",
            ))
            self.lora_module_names = [name for name, module in self.encoder.named_modules()
                                      if hasattr(module, "lora_A")]
            if len(self.lora_module_names) != len(targets):
                raise RuntimeError("Unexpected LoRA target count")
        self.train(self.training)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenPubMedBertClassifier":
        super().train(mode)
        # Start with every frozen native dropout disabled. Enable only the
        # trainable blocks (B) or adapter dropout (C), never the frozen base.
        self.encoder.eval()
        if mode and self.training_strategy == "partial_unfreezing":
            for index in self.adapted_layer_indices:
                self.encoder.encoder.layer[index].train(True)
        elif mode and self.training_strategy == "lora":
            for module in self.encoder.modules():
                if hasattr(module, "lora_dropout"):
                    module.lora_dropout.train(True)
        return self

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask.to(hidden_states.dtype).unsqueeze(-1)
        numerator = (hidden_states * expanded_mask).sum(dim=1)
        denominator = expanded_mask.sum(dim=1).clamp(min=1.0)
        return numerator / denominator

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        criteria_mask: torch.Tensor | None,
        article_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cls_representation = hidden_states[:, 0, :]
        attention_weights: torch.Tensor | None = None

        if self.pooling_type == "cls":
            pooled = cls_representation
        elif self.pooling_type == "mean":
            # Include [CLS] and [SEP]; exclude padding only.
            pooled = self._masked_mean(hidden_states, attention_mask)
        elif self.pooling_type == "attention":
            assert self.attention_scorer is not None
            scores = self.attention_scorer(hidden_states).squeeze(-1)
            valid = attention_mask.bool()
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention_weights = torch.softmax(scores, dim=1)
            # Force exact zero on padding and renormalize valid positions.
            attention_weights = attention_weights * valid.to(attention_weights.dtype)
            attention_weights = attention_weights / attention_weights.sum(
                dim=1, keepdim=True
            ).clamp(min=torch.finfo(attention_weights.dtype).eps)
            pooled = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        else:
            if criteria_mask is None or article_mask is None:
                raise ValueError("Concat pooling requires criteria_mask and article_mask")
            if criteria_mask.shape != attention_mask.shape or article_mask.shape != attention_mask.shape:
                raise ValueError("Concat section masks must match attention_mask shape")
            if criteria_mask.sum(dim=1).eq(0).any() or article_mask.sum(dim=1).eq(0).any():
                raise ValueError("Concat pooling received an empty criteria or article mask")
            criteria_representation = self._masked_mean(hidden_states, criteria_mask)
            article_representation = self._masked_mean(hidden_states, article_mask)
            pooled = torch.cat(
                [cls_representation, criteria_representation, article_representation], dim=-1
            )
        return pooled, attention_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        criteria_mask: torch.Tensor | None = None,
        article_mask: torch.Tensor | None = None,
        return_pooling_details: bool = False,
    ) -> Any:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled, attention_weights = self._pool(
            outputs.last_hidden_state, attention_mask, criteria_mask, article_mask
        )
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        if return_pooling_details:
            return logits, {
                "pooled": pooled,
                "attention_weights": attention_weights,
            }
        return logits


def build_model(
    model_name: str,
    dropout: float,
    pooling_type: str = "cls",
    **strategy_kwargs: Any,
) -> FrozenPubMedBertClassifier:
    return FrozenPubMedBertClassifier(
        model_name=model_name,
        dropout=dropout,
        pooling_type=pooling_type,
        **strategy_kwargs,
    )


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected_names = {"classifier.weight", "classifier.bias"}
    if isinstance(model, FrozenPubMedBertClassifier) and model.pooling_type == "attention":
        expected_names.add("attention_scorer.weight")
    strategy = getattr(model, "training_strategy", "classifier_head_only")
    if strategy == "partial_unfreezing":
        for index in model.adapted_layer_indices:
            expected_names.update(f"encoder.encoder.layer.{index}.{name}"
                                  for name, _ in model.encoder.encoder.layer[index].named_parameters())
    elif strategy == "lora":
        expected_names.update(f"encoder.{module}.{matrix}.default.weight"
                              for module in model.lora_module_names
                              for matrix in ("lora_A", "lora_B"))
    if set(names) != expected_names:
        raise RuntimeError(f"Unexpected trainable parameters: {names}")
    report = {
        "training_strategy": strategy,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_percentage": 100.0 * trainable / total,
        "trainable_parameter_names": names,
        "pooling": getattr(model, "pooling_type", None),
        "pooling_output_dim": getattr(model, "pooling_output_dim", None),
        "classifier_parameters": sum(p.numel() for p in model.classifier.parameters()),
        "encoder_trainable_parameters": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        "adapter_parameters": sum(p.numel() for name, p in model.named_parameters()
                                  if ".lora_A." in name or ".lora_B." in name),
        "adapted_layer_indices": getattr(model, "adapted_layer_indices", []),
        "lora_target_modules": getattr(model, "lora_module_names", []),
    }
    LOGGER.info("Parameter report | %s", json.dumps(report))
    return report


def build_optimizer(model: FrozenPubMedBertClassifier, args: argparse.Namespace) -> AdamW:
    parameter_report(model)
    head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    if model.training_strategy == "classifier_head_only":
        # Same parameter order, defaults and decay policy as the historical A.
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=head_lr, weight_decay=0.01)
    else:
        encoder_lr = (args.backbone_learning_rate if model.training_strategy == "partial_unfreezing"
                      else args.adapter_learning_rate)
        optimizer = AdamW([
            {"params": list(model.classifier.parameters()), "lr": head_lr, "name": "classifier"},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad],
             "lr": encoder_lr, "name": "backbone" if model.training_strategy == "partial_unfreezing" else "adapter"},
        ], lr=head_lr, weight_decay=0.01)
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Optimizer must contain each trainable parameter exactly once")
    return optimizer


def optimizer_report(optimizer: AdamW) -> dict[str, Any]:
    return {"name": "AdamW", "weight_decay": 0.01, "learning_rate": optimizer.param_groups[0]["lr"],
            "parameter_groups": [{"name": group.get("name", "classifier"), "lr": group["lr"],
                                  "weight_decay": group["weight_decay"], "betas": list(group["betas"]),
                                  "eps": group["eps"], "parameters": sum(p.numel() for p in group["params"])}
                                 for group in optimizer.param_groups]}


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def is_better_checkpoint(current_ap: float, best_ap: float) -> bool:
    return not np.isnan(current_ap) and current_ap > best_ap


def compute_pos_weight(train_frame: pd.DataFrame) -> tuple[float, int, int]:
    positive = int((train_frame[LABEL_COLUMN] == 1).sum())
    negative = int((train_frame[LABEL_COLUMN] == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training split must contain positive and negative examples")
    weight = negative / positive
    LOGGER.info(
        "Training-only class weight | positive=%d negative=%d pos_weight=%.10f",
        positive,
        negative,
        weight,
    )
    return weight, positive, negative


def build_criterion(
    train_frame: pd.DataFrame, class_weighting: str, device: torch.device,
) -> tuple[nn.BCEWithLogitsLoss, float | None, int, int]:
    """Construct the training loss using Train labels only.

    The weighted branch is byte-for-byte equivalent in semantics to the
    historical BCE construction; unweighted is an explicit supplementary
    ablation with no positive-class multiplier.
    """
    positive = int((train_frame[LABEL_COLUMN] == 1).sum())
    negative = int((train_frame[LABEL_COLUMN] == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training split must contain positive and negative examples")
    if class_weighting == "weighted":
        pos_weight_value, _, _ = compute_pos_weight(train_frame)
        tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=tensor), pos_weight_value, positive, negative
    if class_weighting == "unweighted":
        LOGGER.info("Training loss | unweighted BCE | positive=%d negative=%d", positive, negative)
        return nn.BCEWithLogitsLoss(), None, positive, negative
    raise ValueError(f"Unsupported class weighting: {class_weighting}")


def make_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ScreeningDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=BudgetedCollator(tokenizer, max_length),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in tqdm(loader, desc=f"Train epoch {epoch}"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        criteria_mask = batch["criteria_mask"].to(device, non_blocking=True)
        article_mask = batch["article_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            criteria_mask=criteria_mask,
            article_mask=article_mask,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_rows = labels.shape[0]
        total_loss += float(loss.item()) * batch_rows
        total_rows += batch_rows
    return total_loss / total_rows


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Keep thresholding isolated for future validation-only tuning."""
    return (probabilities >= threshold).astype(np.int64)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    context: str = "evaluation",
) -> dict[str, Any]:
    predictions = apply_threshold(probabilities, threshold)
    labels = labels.astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    def divide(numerator: int, denominator: int, metric_name: str) -> float:
        if denominator:
            return float(numerator / denominator)
        LOGGER.warning("%s %s is undefined; recording NaN", context, metric_name)
        return float("nan")

    recall = divide(tp, tp + fn, "Recall")
    precision = divide(tp, tp + fp, "Precision")
    f1_denominator = 2 * tp + fp + fn
    predicted_include_rate = divide(tp + fp, len(labels), "Predicted include rate")
    metrics: dict[str, Any] = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": divide(2 * tp, f1_denominator, "F1"),
        "fnr": divide(fn, fn + tp, "FNR"),
        "specificity": divide(tn, tn + fp, "Specificity"),
        "npv": divide(tn, tn + fn, "NPV"),
        "predicted_include_rate": predicted_include_rate,
        "workload_reduction": 1.0 - predicted_include_rate,
        "accuracy": divide(tp + tn, len(labels), "Accuracy"),
        "threshold": float(threshold),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
    }
    if len(np.unique(labels)) == 2:
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        LOGGER.warning(
            "%s PR-AUC and ROC-AUC are undefined because only one class is present; "
            "recording NaN",
            context,
        )
    return metrics


def calculate_per_review_metrics(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    context_name: str = "Validation",
) -> list[dict[str, Any]]:
    """Calculate threshold diagnostics separately for each review."""
    aligned = pd.DataFrame(
        {
            "review_name": source.iloc[indices][REVIEW_COLUMN].astype(str).to_numpy(),
            "true_label": labels,
            "predicted_probability": probabilities,
        }
    )
    results: list[dict[str, Any]] = []
    for review_name, group in aligned.groupby("review_name", sort=True):
        review_labels = group["true_label"].to_numpy(dtype=np.int64)
        review_probabilities = group["predicted_probability"].to_numpy(dtype=float)
        metrics = calculate_metrics(
            review_labels,
            review_probabilities,
            threshold,
            context=f"{context_name} review {review_name}",
        )
        results.append(
            {
                "review_name": review_name,
                "n_samples": int(len(group)),
                "n_positive": int((review_labels == 1).sum()),
                "n_negative": int((review_labels == 0).sum()),
                **metrics,
            }
        )
    return results


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    description: str,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            criteria_mask = batch["criteria_mask"].to(device, non_blocking=True)
            article_mask = batch["article_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                criteria_mask=criteria_mask,
                article_mask=article_mask,
            )
            loss = criterion(logits, labels)
            batch_rows = labels.shape[0]
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(batch["row_indices"].numpy())
    logits_array = np.concatenate(all_logits)
    labels_array = np.concatenate(all_labels).astype(np.int64)
    indices_array = np.concatenate(all_indices).astype(np.int64)
    # torch.sigmoid is numerically stable for large-magnitude logits.
    probabilities = torch.sigmoid(torch.from_numpy(logits_array)).numpy()
    metrics = calculate_metrics(labels_array, probabilities, threshold)
    return total_loss / total_rows, metrics, probabilities, labels_array, indices_array


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: FrozenPubMedBertClassifier,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, Any],
    pos_weight: float | None,
) -> None:
    checkpoint = {
        "checkpoint_version": 2,
        "training_strategy": model.training_strategy,
        "strategy_config": model.strategy_config,
        "parameter_report": parameter_report(model),
        "classifier_head_state_dict": {
            key: value.detach().cpu() for key, value in model.classifier.state_dict().items()
        },
        "attention_pooling_state_dict": (
            {
                key: value.detach().cpu()
                for key, value in model.attention_scorer.state_dict().items()
            }
            if model.attention_scorer is not None
            else None
        ),
        "model_config": {
            "hidden_size": int(model.encoder.config.hidden_size),
            "dropout": float(args.dropout),
            "num_labels": 1,
            "pooling": model.pooling_type,
            "pooling_output_dim": model.pooling_output_dim,
            "attention_pooling_trainable": model.attention_scorer is not None,
        },
        "training_arguments": serializable_args(args),
        "model_name": args.model_name_or_path,
        "model_revision": model.model_revision,
        "optimizer": optimizer_report(build_optimizer(model, args)),
        "epoch": int(epoch),
        "validation_metrics": metrics,
        "threshold": float(args.threshold),
        "pos_weight": float(pos_weight) if pos_weight is not None else None,
        "seed": int(args.seed),
    }
    if model.training_strategy == "partial_unfreezing":
        checkpoint["upper_encoder_state_dicts"] = {
            str(index): {name: value.detach().cpu()
                         for name, value in model.encoder.encoder.layer[index].state_dict().items()}
            for index in model.adapted_layer_indices
        }
    elif model.training_strategy == "lora":
        from peft import get_peft_model_state_dict
        checkpoint["adapter_config"] = json.loads(json.dumps(
            model.encoder.peft_config["default"].to_dict(), default=lambda value: sorted(value)))
        checkpoint["adapter_state_dict"] = {
            name: value.detach().cpu() for name, value in get_peft_model_state_dict(model.encoder).items()
        }
    torch.save(checkpoint, path)


def load_classifier_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FrozenPubMedBertClassifier, dict[str, Any]]:
    """Recreate the frozen encoder and restore the compact classifier head."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    pooling_type = str(checkpoint["model_config"].get("pooling", "cls")).lower()
    model = build_model(
        checkpoint["model_name"],
        checkpoint["model_config"]["dropout"],
        pooling_type,
        model_revision=checkpoint.get("model_revision"),
        **checkpoint.get("strategy_config", {}),
    )
    if checkpoint.get("training_strategy", "classifier_head_only") != model.training_strategy:
        raise ValueError("Checkpoint strategy metadata mismatch")
    if model.training_strategy == "partial_unfreezing":
        upper = checkpoint["upper_encoder_state_dicts"]
        if set(upper) != {str(i) for i in model.adapted_layer_indices}:
            raise ValueError("Checkpoint upper encoder layers are incomplete")
        for index in model.adapted_layer_indices:
            model.encoder.encoder.layer[index].load_state_dict(upper[str(index)], strict=True)
    elif model.training_strategy == "lora":
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        saved = checkpoint["adapter_state_dict"]
        expected = get_peft_model_state_dict(model.encoder)
        if set(saved) != set(expected) or any(saved[k].shape != expected[k].shape for k in expected):
            raise ValueError("Checkpoint adapter keys/shapes are incomplete or incompatible")
        expected_config = model.encoder.peft_config["default"].to_dict()
        for key in ("r", "lora_alpha", "lora_dropout", "bias", "target_modules"):
            left, right = checkpoint["adapter_config"][key], expected_config[key]
            if key == "target_modules":
                left, right = sorted(left), sorted(right)
            if left != right:
                raise ValueError(f"Checkpoint adapter config mismatch: {key}")
        set_peft_model_state_dict(model.encoder, saved)
    model.classifier.load_state_dict(checkpoint["classifier_head_state_dict"])
    attention_state = checkpoint.get("attention_pooling_state_dict")
    if model.attention_scorer is not None:
        if attention_state is None:
            raise ValueError("Attention checkpoint is missing attention-pooling parameters")
        model.attention_scorer.load_state_dict(attention_state)
    parameter_report(model)
    model.to(device)
    model.eval()
    return model, checkpoint


def prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    columns = [REVIEW_COLUMN, TITLE_COLUMN]
    columns.extend(column for column in IDENTIFIER_COLUMNS if column in source.columns)
    output = source.iloc[indices][columns].reset_index(drop=True).copy()
    output = output.rename(columns={REVIEW_COLUMN: "review_name"})
    output["true_label"] = labels
    output["predicted_probability"] = probabilities
    output["predicted_label"] = apply_threshold(probabilities, threshold)
    return output


def create_run_dir(base: Path, prefix: str = "run") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def dry_run_forward(
    model: nn.Module,
    frame: pd.DataFrame,
    tokenizer: Any,
    args: argparse.Namespace,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    # Preflight/dry-run uses the requested batch size so it can expose an OOM
    # before a formal run. It still executes exactly one forward/backward step.
    sample = frame.iloc[: min(args.batch_size, len(frame))].copy()
    loader = make_loader(sample, tokenizer, args.max_length, len(sample), False, 0, args.seed)
    batch = next(iter(loader))
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    criteria_mask = batch["criteria_mask"].to(device)
    article_mask = batch["article_mask"].to(device)
    logits, details = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        criteria_mask=criteria_mask,
        article_mask=article_mask,
        return_pooling_details=True,
    )
    loss = criterion(logits, batch["labels"].to(device))
    if not torch.isfinite(loss):
        raise RuntimeError("Dry-run loss is NaN or Inf")
    loss.backward()

    trainable_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    encoder_gradients = [
        name for name, parameter in model.encoder.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if missing_gradients or encoder_gradients:
        raise RuntimeError(
            f"Gradient sanity check failed: missing={missing_gradients}, "
            f"encoder_gradients={encoder_gradients}"
        )

    attention_weights = details["attention_weights"]
    if model.pooling_type == "attention":
        assert attention_weights is not None
        valid_sums = attention_weights.sum(dim=1)
        padding_weights = attention_weights.masked_select(attention_mask.eq(0))
        if not torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-6):
            raise RuntimeError("Attention pooling weights do not sum to one")
        if padding_weights.numel() and not torch.allclose(
            padding_weights, torch.zeros_like(padding_weights), atol=1e-7
        ):
            raise RuntimeError("Attention pooling assigned weight to padding")
        LOGGER.info(
            "Attention QC | weights_shape=%s valid_sums=%s max_padding_weight=%.8g",
            tuple(attention_weights.shape),
            [float(value) for value in valid_sums.detach().cpu()],
            float(padding_weights.abs().max().item()) if padding_weights.numel() else 0.0,
        )

    criteria_counts = criteria_mask.sum(dim=1)
    article_counts = article_mask.sum(dim=1)
    valid_counts = attention_mask.sum(dim=1)
    padding_counts = attention_mask.shape[1] - valid_counts
    LOGGER.info(
        "Section-mask QC | valid=%s criteria=%s article=%s padding=%s "
        "empty_criteria=%d empty_article=%d overlap=%d",
        valid_counts.cpu().tolist(),
        criteria_counts.cpu().tolist(),
        article_counts.cpu().tolist(),
        padding_counts.cpu().tolist(),
        int(criteria_counts.eq(0).sum()),
        int(article_counts.eq(0).sum()),
        int((criteria_mask.bool() & article_mask.bool()).sum()),
    )
    LOGGER.info(
        "Dry-run forward PASS | pooling=%s input_shape=%s pooled_shape=%s "
        "logits_shape=%s loss=%.8f device=%s trainable_gradients=%s frozen_encoder_gradients=NONE",
        model.pooling_type,
        tuple(batch["input_ids"].shape),
        tuple(details["pooled"].shape),
        tuple(logits.shape),
        float(loss.item()),
        device,
        trainable_gradient_names,
    )
    model.zero_grad(set_to_none=True)


def one_step_preflight(model: FrozenPubMedBertClassifier, frame: pd.DataFrame,
                       tokenizer: Any, args: argparse.Namespace, criterion: nn.Module,
                       device: torch.device) -> dict[str, Any]:
    """One real optimizer update; compare values, including frozen parameters."""
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    loader = make_loader(frame.iloc[:args.batch_size], tokenizer, args.max_length,
                         args.batch_size, False, 0, args.seed)
    optimizer = build_optimizer(model, args)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss = train_one_epoch(model, loader, optimizer, criterion, device, 1)
    changed = [name for name, p in model.named_parameters()
               if not torch.equal(before[name], p.detach().cpu())]
    frozen_changed = [name for name, p in model.named_parameters()
                      if not p.requires_grad and name in changed]
    if frozen_changed or not any(name.startswith("classifier.") for name in changed):
        raise RuntimeError(f"Parameter update invariant failed: frozen={frozen_changed}")
    if model.training_strategy != "classifier_head_only" and not any(name.startswith("encoder.") for name in changed):
        raise RuntimeError("No encoder/adapter parameter changed")
    if not np.isfinite(loss):
        raise RuntimeError("Non-finite preflight loss")
    result = {"strategy": model.training_strategy, "loss": loss,
              "changed_parameter_names": changed, "frozen_parameters_unchanged": True,
              "microbatch_size": min(args.batch_size, len(frame)),
              "effective_batch_size": min(args.batch_size, len(frame)),
              "gradient_accumulation_steps": 1, "max_length": args.max_length,
              "device": str(device)}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update(gpu_name=torch.cuda.get_device_name(device),
                      total_vram_bytes=torch.cuda.get_device_properties(device).total_memory,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    LOGGER.info("Optimizer-step preflight | %s", json.dumps(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    smoke_test = args.max_train_samples is not None or args.max_validation_samples is not None
    if smoke_test:
        LOGGER.warning("=" * 72)
        LOGGER.warning("SMOKE TEST MODE")
        LOGGER.warning(
            "Requested limits | max_train_samples=%s max_validation_samples=%s",
            args.max_train_samples,
            args.max_validation_samples,
        )
        LOGGER.warning("=" * 72)
    set_seed(args.seed)
    device = select_device()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this run, but torch.cuda.is_available() is False"
        )
    LOGGER.info(
        "Environment | pooling=%s seed=%d device=%s torch=%s transformers=%s model=%s",
        args.pooling,
        args.seed,
        device,
        torch.__version__,
        transformers.__version__,
        args.model_name_or_path,
    )

    splits = load_splits(args.split_dir, args.evaluate_test)
    validate_splits(splits, args.evaluate_test)
    criteria, criteria_schema = load_eligibility_criteria(args.criteria_path)
    LOGGER.info(
        "Eligibility workbook | sheets_used=%s columns=%s review_text_columns=%s",
        criteria_schema.sheet_name,
        list(criteria.columns),
        list(criteria_schema.review_text_columns),
    )
    splits["train"] = join_criteria(splits["train"], criteria, criteria_schema, "train")
    splits["validation"] = join_criteria(
        splits["validation"], criteria, criteria_schema, "validation"
    )
    if args.evaluate_test:
        splits["test"] = join_criteria(splits["test"], criteria, criteria_schema, "test")

    original_summaries = {
        "train": summarize_split(splits["train"], "original train" if smoke_test else "train"),
        "validation": summarize_split(
            splits["validation"], "original validation" if smoke_test else "validation"
        ),
    }
    if args.evaluate_test:
        original_summaries["test"] = summarize_split(splits["test"], "test")

    if smoke_test:
        splits["train"] = stratified_subsample(
            splits["train"], args.max_train_samples, args.seed, "train"
        )
        splits["validation"] = stratified_subsample(
            splits["validation"], args.max_validation_samples, args.seed + 10_000, "validation"
        )
        summaries = {
            "train": summarize_split(splits["train"], "smoke train"),
            "validation": summarize_split(splits["validation"], "smoke validation"),
        }
        LOGGER.warning(
            "SMOKE TEST ROWS | Original train rows=%d Smoke train rows=%d | "
            "Original validation rows=%d Smoke validation rows=%d",
            original_summaries["train"]["rows"],
            summaries["train"]["rows"],
            original_summaries["validation"]["rows"],
            summaries["validation"]["rows"],
        )
        LOGGER.warning(
            "SMOKE TEST CLASSES | train positive=%d negative=%d | "
            "validation positive=%d negative=%d",
            summaries["train"]["positive"],
            summaries["train"]["negative"],
            summaries["validation"]["positive"],
            summaries["validation"]["negative"],
        )
    else:
        summaries = original_summaries

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # These labels are ordinary text; tokenizer vocabulary and embeddings are unchanged.
    model = build_model(args.model_name_or_path, args.dropout, args.pooling, **strategy_options(args))
    if args.comparison_mode and not smoke_test and not args.dry_run:
        if model.hidden_size != 768 or model.encoder.config.num_hidden_layers != 12:
            raise ValueError("Formal comparison requires a 12-layer, hidden-size-768 BERT base")
    model_limit = min(
        int(getattr(tokenizer, "model_max_length", args.max_length)),
        int(getattr(model.encoder.config, "max_position_embeddings", args.max_length)),
    )
    if args.max_length > model_limit:
        raise ValueError(
            f"Requested max_length={args.max_length} exceeds model/tokenizer limit {model_limit}"
        )
    model.to(device)
    parameter_info = parameter_report(model)
    backbone_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    backbone_frozen = backbone_trainable_parameters == 0
    if args.training_strategy == "classifier_head_only" and not backbone_frozen:
        raise RuntimeError(
            f"Frozen-backbone invariant failed: {backbone_trainable_parameters} "
            "encoder parameters are trainable"
        )
    criterion, pos_weight_value, train_positive, train_negative = build_criterion(
        splits["train"], args.class_weighting, device
    )
    if smoke_test:
        LOGGER.warning("Smoke train positive: %d", train_positive)
        LOGGER.warning("Smoke train negative: %d", train_negative)
        LOGGER.warning("Smoke train class_weighting=%s pos_weight=%s", args.class_weighting, pos_weight_value)

    truncation = {
        "train": report_truncation(splits["train"], tokenizer, args.max_length, "train"),
        "validation": report_truncation(
            splits["validation"], tokenizer, args.max_length, "validation"
        ),
    }
    if args.evaluate_test:
        truncation["test"] = report_truncation(splits["test"], tokenizer, args.max_length, "test")

    if args.dry_run:
        dry_run_forward(model, splits["train"], tokenizer, args, criterion, device)
        one_step_preflight(model, splits["train"], tokenizer, args, criterion, device)
        LOGGER.info("Dry-run complete: one disposable optimizer step; no formal checkpoint written")
        return 0

    pooling_output_base = args.output_dir / args.pooling
    if args.comparison_mode or args.training_strategy != "classifier_head_only":
        pooling_output_base = args.output_dir / args.training_strategy / f"seed_{args.seed}" / args.pooling
    run_dir = create_run_dir(
        pooling_output_base,
        prefix="smoke" if smoke_test else "run",
    )
    configure_logging(run_dir)
    if smoke_test:
        LOGGER.warning("=" * 72)
        LOGGER.warning("SMOKE TEST MODE")
        LOGGER.warning(
            "Original train rows: %d | Smoke train rows: %d",
            original_summaries["train"]["rows"],
            summaries["train"]["rows"],
        )
        LOGGER.warning(
            "Original validation rows: %d | Smoke validation rows: %d",
            original_summaries["validation"]["rows"],
            summaries["validation"]["rows"],
        )
        LOGGER.warning(
            "Smoke train positive: %d | Smoke train negative: %d | "
            "class_weighting=%s | pos_weight=%s",
            summaries["train"]["positive"],
            summaries["train"]["negative"],
            args.class_weighting,
            pos_weight_value,
        )
        LOGGER.warning(
            "Smoke validation positive: %d | Smoke validation negative: %d",
            summaries["validation"]["positive"],
            summaries["validation"]["negative"],
        )
        LOGGER.warning("=" * 72)
    # Repeat preflight results after the file logger is attached so run.log is
    # self-contained (the same information was already printed to the console).
    for split_name, split_summary in summaries.items():
        LOGGER.info("%s data QC | %s", split_name, json.dumps(split_summary, ensure_ascii=False))
        LOGGER.info(
            "%s criteria QC | examples=%d matched=%d unmatched=0",
            split_name,
            len(splits[split_name]),
            len(splits[split_name]),
        )
    for split_name, truncation_summary in truncation.items():
        LOGGER.info("%s truncation QC | %s", split_name, json.dumps(truncation_summary))
    LOGGER.info("Parameter report | %s", json.dumps(parameter_info))
    LOGGER.info(
        "Training loss | class_weighting=%s | positive=%d negative=%d pos_weight=%s",
        args.class_weighting,
        summaries["train"]["positive"],
        summaries["train"]["negative"],
        pos_weight_value,
    )
    optimizer = build_optimizer(model, args)
    config = {
        **serializable_args(args),
        "smoke_test": smoke_test,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.model_name_or_path,
        "training_strategy": args.training_strategy,
        "pooling": args.pooling,
        "pooling_output_dim": model.pooling_output_dim,
        "attention_pooling_trainable": model.attention_scorer is not None,
        "trainable_parameters": parameter_info["trainable_parameters"],
        "frozen_parameters": parameter_info["frozen_parameters"],
        "backbone_frozen": backbone_frozen,
        "backbone_trainable_parameters": backbone_trainable_parameters,
        "optimizer": optimizer_report(optimizer),
        "warmup": None,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "scheduler": None,
        "gradient_clipping": None,
        "loss": "weighted_BCEWithLogitsLoss" if args.class_weighting == "weighted" else "BCEWithLogitsLoss",
        "class_weighting": args.class_weighting,
        "pos_weight": pos_weight_value,
        "checkpoint_selection_metric": "validation_pr_auc",
        "criteria_sheet": criteria_schema.sheet_name,
        "criteria_review_key_column": criteria_schema.review_key,
        "criteria_inclusion_column": criteria_schema.inclusion,
        "criteria_exclusion_column": criteria_schema.exclusion,
        "criteria_eligibility_column": criteria_schema.eligibility,
        "review_text_columns": list(criteria_schema.review_text_columns),
        "parameter_report": parameter_info,
        "data_qc": summaries,
        "original_data_qc": original_summaries if smoke_test else None,
        "truncation_qc": truncation,
        "test_evaluated": bool(args.evaluate_test),
        "environment": {
            "packages": package_versions(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "model_revision": model.model_revision,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "parameter_report.json").write_text(
        json.dumps(parameter_info, indent=2), encoding="utf-8")

    train_loader = make_loader(
        splits["train"], tokenizer, args.max_length, args.batch_size, True,
        args.num_workers, args.seed,
    )
    validation_loader = make_loader(
        splits["validation"], tokenizer, args.max_length, args.batch_size, False,
        args.num_workers, args.seed,
    )
    initial_trainable_parameters = (
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        if smoke_test
        else None
    )

    history: list[dict[str, Any]] = []
    best_pr_auc = -float("inf")
    best_pr_auc_epoch = 0
    best_checkpoint_metrics: dict[str, Any] | None = None
    best_per_review_metrics: list[dict[str, Any]] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        validation_loss, metrics, probabilities, labels, indices = evaluate(
            model, validation_loader, criterion, device, args.threshold, f"Validation epoch {epoch}"
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **metrics,
        }
        history.append(row)
        LOGGER.info("Epoch %d | %s", epoch, json.dumps(row))

        current_pr_auc = float(metrics["pr_auc"])
        if is_better_checkpoint(current_pr_auc, best_pr_auc):
            best_pr_auc = current_pr_auc
            best_pr_auc_epoch = epoch
            checkpoint_metrics = {**metrics, "validation_loss": validation_loss}
            best_checkpoint_metrics = checkpoint_metrics
            save_checkpoint(
                run_dir / "best_pr_auc_classifier_head.pt",
                model,
                args,
                epoch,
                checkpoint_metrics,
                pos_weight_value,
            )
            best_predictions = prediction_frame(
                splits["validation"], probabilities, labels, indices, args.threshold
            )
            best_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
            best_per_review_metrics = calculate_per_review_metrics(
                splits["validation"], probabilities, labels, indices, args.threshold
            )
            pd.DataFrame(best_per_review_metrics).to_csv(
                run_dir / "per_review_validation_metrics.csv", index=False
            )
            best_artifacts = {
                "training_strategy": args.training_strategy,
                "pooling": args.pooling,
                "best_epoch": epoch,
                "selection_metric": "validation_pr_auc",
                "overall_validation": checkpoint_metrics,
                "per_review_validation": best_per_review_metrics,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(best_artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    parameter_update_check: dict[str, bool] | None = None
    if smoke_test:
        assert initial_trainable_parameters is not None
        parameter_update_check = {
            f"{name}_changed": not torch.equal(
                initial_trainable_parameters[name], parameter.detach().cpu()
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        parameter_update_check["frozen_parameters_unchanged"] = all(
            torch.equal(initial_trainable_parameters[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters() if not parameter.requires_grad
        )
        parameter_update_check["encoder_frozen"] = all(
            not p.requires_grad for p in model.encoder.parameters())
        if not parameter_update_check["frozen_parameters_unchanged"]:
            raise RuntimeError("Frozen parameter values changed during smoke test")
        if not parameter_update_check["classifier.weight_changed"]:
            raise RuntimeError("Classifier did not update during smoke test")
        if args.training_strategy != "classifier_head_only" and not any(
            value for name, value in parameter_update_check.items() if name.startswith("encoder.")
        ):
            raise RuntimeError("Encoder/adapter did not update during smoke test")
        LOGGER.info("Smoke parameter update | %s", json.dumps(parameter_update_check))
        del initial_trainable_parameters

    if best_checkpoint_metrics is None or best_per_review_metrics is None:
        raise RuntimeError("No best checkpoint was selected because validation PR-AUC was NaN")
    checkpoint_roundtrip = None
    if smoke_test:
        restored, _ = load_classifier_checkpoint(run_dir / "best_pr_auc_classifier_head.pt", device)
        _, _, restored_probabilities, _, _ = evaluate(
            restored, validation_loader, criterion, device, args.threshold, "Smoke checkpoint round-trip")
        saved_probabilities = pd.read_csv(run_dir / "validation_predictions.csv")["predicted_probability"].to_numpy()
        if not np.allclose(restored_probabilities, saved_probabilities, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Best checkpoint predictions changed after reload")
        checkpoint_roundtrip = {"passed": True, "rtol": 1e-5, "atol": 1e-6}
        del restored
    LOGGER.info(
        "Best checkpoint summary | pooling=%s epoch=%d selection=validation_pr_auc metrics=%s",
        args.pooling,
        best_pr_auc_epoch,
        json.dumps(best_checkpoint_metrics),
    )
    for review_metrics in best_per_review_metrics:
        LOGGER.info(
            "Per-review validation | %s",
            json.dumps(review_metrics, ensure_ascii=False),
        )

    summary: dict[str, Any] = {
        "model": args.model_name_or_path,
        "smoke_test": smoke_test,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "training_strategy": args.training_strategy,
        "strategy_config": model.strategy_config,
        "optimizer": optimizer_report(optimizer),
        "head_learning_rate": optimizer.param_groups[0]["lr"],
        "backbone_learning_rate": args.backbone_learning_rate if args.training_strategy == "partial_unfreezing" else None,
        "adapter_learning_rate": args.adapter_learning_rate if args.training_strategy == "lora" else None,
        "parameter_report": parameter_info,
        "pooling": args.pooling,
        "pooling_output_dim": model.pooling_output_dim,
        "attention_pooling_trainable": model.attention_scorer is not None,
        "trainable_parameters": parameter_info["trainable_parameters"],
        "frozen_parameters": parameter_info["frozen_parameters"],
        "backbone_frozen": backbone_frozen,
        "backbone_trainable_parameters": backbone_trainable_parameters,
        "loss": "weighted_BCEWithLogitsLoss" if args.class_weighting == "weighted" else "BCEWithLogitsLoss",
        "class_weighting": args.class_weighting,
        "pos_weight": pos_weight_value,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "seed": args.seed,
        "threshold": args.threshold,
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "checkpoint_selection_metric": "validation_pr_auc",
        "best_pr_auc_epoch": best_pr_auc_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "best_checkpoint_path": str(run_dir / "best_pr_auc_classifier_head.pt"),
        "best_checkpoint_metrics": best_checkpoint_metrics,
        "per_review_validation": best_per_review_metrics,
        "test_evaluated": bool(args.evaluate_test),
        "parameter_update_check": parameter_update_check,
        "checkpoint_roundtrip": checkpoint_roundtrip,
    }

    if args.evaluate_test:
        # This occurs after selection is complete and always uses the best-PR-AUC checkpoint.
        model, _ = load_classifier_checkpoint(
            run_dir / "best_pr_auc_classifier_head.pt", device
        )
        test_loader = make_loader(
            splits["test"], tokenizer, args.max_length, args.batch_size, False,
            args.num_workers, args.seed,
        )
        test_loss, test_metrics, probabilities, labels, indices = evaluate(
            model, test_loader, criterion, device, args.threshold, "Final test evaluation"
        )
        summary["test_loss"] = test_loss
        summary["test_metrics"] = test_metrics
        prediction_frame(
            splits["test"], probabilities, labels, indices, args.threshold
        ).to_csv(run_dir / "test_predictions.csv", index=False)

    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Training complete | output_dir=%s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
