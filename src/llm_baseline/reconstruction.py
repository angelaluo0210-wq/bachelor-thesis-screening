"""Reconstruct readable, sectioned content retained by PubMedBERT.

The source of truth for the final BERT sequence is
``train_classifier_head.tokenize_with_section_budget``.  This module invokes
that function and independently derives field boundaries only to make its
otherwise opaque concatenated content sequence readable for a generative
model.  It refuses to return a reconstruction if that derivation does not
reassemble to the source-of-truth token sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src import train_classifier_head as bert


FIELD_NAMES = ("inclusion", "exclusion", "title", "abstract")


@dataclass(frozen=True)
class ReconstructedInput:
    """Readable forms of the exact content-token prefixes given to PubMedBERT."""

    inclusion: str
    exclusion: str
    title: str
    abstract: str
    token_ids: dict[str, tuple[int, ...]]
    input_ids: tuple[int, ...]
    reconstruction_roundtrip: dict[str, bool]


def _field_texts(row: dict[str, Any]) -> dict[str, str]:
    review = bert.safe_text(row.get("_review_text", ""))
    inclusion_parts = []
    if review:
        inclusion_parts.append(f"[REVIEW]\n{review}")
    inclusion_parts.append(f"[INCLUSION]\n{bert.safe_text(row.get('_inclusion_text', ''))}")
    return {
        "inclusion": "\n\n".join(inclusion_parts),
        "exclusion": f"[EXCLUSION]\n{bert.safe_text(row.get('_exclusion_text', ''))}",
        "title": f"[TITLE]\n{bert.safe_text(row.get(bert.TITLE_COLUMN, ''))}",
        "abstract": f"[ABSTRACT]\n{bert.safe_text(row.get(bert.ABSTRACT_COLUMN, ''))}",
    }


def _budgets(token_ids: dict[str, list[int]], max_length: int) -> dict[str, int]:
    """Mirror only the public field allocation contract for boundary recovery.

    The final sequence remains verified against the production function below.
    Keeping this tiny calculation local avoids changing the frozen training
    collator's public return schema merely to expose diagnostic boundaries.
    """
    usable = max_length - 2
    if usable <= 0:
        raise ValueError(f"max_length={max_length} leaves no content-token budget")
    title = min(len(token_ids["title"]), min(64, usable))
    inclusion = min(len(token_ids["inclusion"]), min(128, usable - title))
    exclusion = min(len(token_ids["exclusion"]), min(128, usable - title - inclusion))
    abstract = min(len(token_ids["abstract"]), usable - title - inclusion - exclusion)
    remaining = usable - title - inclusion - exclusion - abstract
    inclusion_need = len(token_ids["inclusion"]) - inclusion
    exclusion_need = len(token_ids["exclusion"]) - exclusion
    if remaining and inclusion_need and exclusion_need:
        extra = min(inclusion_need, exclusion_need, remaining // 2)
        inclusion += extra; exclusion += extra; inclusion_need -= extra; exclusion_need -= extra; remaining -= 2 * extra
    if remaining and inclusion_need and exclusion_need:
        if inclusion_need >= exclusion_need:
            inclusion += 1; inclusion_need -= 1
        else:
            exclusion += 1; exclusion_need -= 1
        remaining -= 1
    if remaining and inclusion_need:
        extra = min(inclusion_need, remaining)
        inclusion += extra; remaining -= extra
    if remaining and exclusion_need:
        exclusion += min(exclusion_need, remaining)
    return {"inclusion": inclusion, "exclusion": exclusion, "title": title, "abstract": abstract}


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def reconstruct_input(tokenizer: Any, row: dict[str, Any], max_length: int = 512) -> ReconstructedInput:
    """Return each retained BERT field as readable decoded text.

    BERT decoding is lossy at the character level for some WordPiece models:
    it may normalize case/whitespace and render unknown Unicode as ``[UNK]``.
    ``reconstruction_roundtrip`` reports whether each decoded section
    re-encodes to the retained token IDs.  It is not a claim of character-level
    equivalence to the raw CSV text.
    """
    texts = _field_texts(row)
    original_ids = {name: tokenizer.encode(text, add_special_tokens=False) for name, text in texts.items()}
    budgets = _budgets(original_ids, max_length)
    retained = {name: ids[:budgets[name]] for name, ids in original_ids.items()}
    encoded = bert.tokenize_with_section_budget(tokenizer, row, max_length)
    active = int(sum(encoded["attention_mask"]))
    expected = [tokenizer.cls_token_id] + [token for name in FIELD_NAMES for token in retained[name]] + [tokenizer.sep_token_id]
    if encoded["input_ids"][:active] != expected:
        raise RuntimeError("Reconstruction boundary derivation disagrees with PubMedBERT tokenization")
    decoded = {name: _decode(tokenizer, retained[name]) for name in FIELD_NAMES}
    roundtrip = {
        name: tokenizer.encode(decoded[name], add_special_tokens=False) == retained[name]
        for name in FIELD_NAMES
    }
    return ReconstructedInput(
        inclusion=decoded["inclusion"], exclusion=decoded["exclusion"],
        title=decoded["title"], abstract=decoded["abstract"],
        token_ids={name: tuple(retained[name]) for name in FIELD_NAMES},
        input_ids=tuple(encoded["input_ids"]), reconstruction_roundtrip=roundtrip,
    )
