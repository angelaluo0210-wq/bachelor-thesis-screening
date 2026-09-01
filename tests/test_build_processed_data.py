import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_processed_data import (
    POST_QC_CONFIRMED_REPAIRS,
    _read_rows,
    _validate_source_rows,
    _write_rows,
    apply_confirmed_repairs,
)
from src.mojibake_label1_qc import article_identifier


def _repair(pattern, replacement):
    return {
        "suspicious_pattern": pattern,
        "replacement": replacement,
        "source_file": "review.csv",
        "row_index": "7",
        "field": "abstract",
    }


def test_confirmed_repairs_are_applied_in_logged_order():
    original = "thresholds �?1 and �?2"
    repairs = [_repair("�?", "≥"), _repair("�?", "≤")]

    assert apply_confirmed_repairs(original, repairs) == "thresholds ≥1 and ≤2"


def test_missing_confirmed_pattern_fails_closed():
    with pytest.raises(ValueError, match="not found"):
        apply_confirmed_repairs("already clean", [_repair("�?", "≥")])


def test_article_identifier_prefers_doi():
    row = {"doi": "https://doi.org/10.1/example", "pmid": "123", "title": "Example"}

    assert article_identifier(row) == ("doi", "https://doi.org/10.1/example")


def test_post_qc_sleep_wake_patch_is_narrow_and_confirmed():
    repair = POST_QC_CONFIRMED_REPAIRS[0]

    assert repair["suspicious_pattern"] == "�C"
    assert repair["replacement"] == "–"
    assert repair["action"] == "auto_repaired"


@pytest.mark.parametrize(
    "csv_text, message",
    [
        ("title,,review_key\nExample,,Review\n", "Empty CSV header"),
        ("title,title,review_key\nA,B,Review\n", "Duplicate CSV header"),
    ],
)
def test_read_rows_rejects_empty_or_duplicate_headers(tmp_path, csv_text, message):
    path = tmp_path / "Review_work_qc.csv"
    path.write_text(csv_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _read_rows(path)


def test_read_rows_rejects_unexpected_field_count(tmp_path):
    path = tmp_path / "Review_work_qc.csv"
    path.write_text("title,label_included,review_key\nExample,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected field count"):
        _read_rows(path)


@pytest.mark.parametrize(
    "row, message",
    [
        ({"label_included": "2", "review_key": "Review"}, "Unexpected label"),
        ({"label_included": "0", "review_key": "Wrong"}, "Unexpected review_key"),
    ],
)
def test_source_validation_rejects_unknown_label_or_review(tmp_path, row, message):
    source_path = tmp_path / "Review_work_qc.csv"

    with pytest.raises(ValueError, match=message):
        _validate_source_rows(source_path, [row])


def test_source_validation_rejects_embedded_secondary_record(tmp_path):
    source_path = tmp_path / "Review_work_qc.csv"
    row = {
        "label_included": "0",
        "review_key": "Review",
        "abstract": "first\ufffd?,PubMed,OpenAlex,0,Review\nhttps://doi.org/10.1/second",
    }

    with pytest.raises(ValueError, match="Embedded secondary-record"):
        _validate_source_rows(source_path, [row])


def test_write_rows_rejects_extras(tmp_path):
    path = tmp_path / "output.csv"

    with pytest.raises(ValueError, match="extras"):
        _write_rows(path, ["title"], [{"title": "Example", "hidden": "must fail"}])

    assert not path.exists()
