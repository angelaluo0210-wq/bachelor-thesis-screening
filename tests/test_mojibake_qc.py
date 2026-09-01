import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mojibake_qc import repair_text


def test_known_chinese_encoding_artifact_is_repaired():
    repaired, issues = repair_text("sleep鈥搘ake cycle")

    assert repaired == "sleep–wake cycle"
    assert issues[0]["action"] == "auto_repaired"
    assert issues[0]["confidence"] == "high"


def test_normal_en_dash_is_preserved():
    repaired, issues = repair_text("sleep–wake")

    assert repaired == "sleep–wake"
    assert issues == []


def test_normal_scientific_unicode_is_preserved():
    for text in ("γ-aminobutyric acid", "α-synuclein", "mean ± SD", "value ≤ 5"):
        repaired, issues = repair_text(text)
        assert repaired == text
        assert issues == []


def test_known_windows_1252_artifacts_are_repaired():
    repaired, issues = repair_text("quoted â€œtextâ€ with â€™ and â€“ dash")

    assert repaired == "quoted “text” with ’ and – dash"
    assert all(issue["action"] == "auto_repaired" for issue in issues)


def test_irrecoverable_replacement_character_requires_manual_review():
    repaired, issues = repair_text("unknown � artifact")

    assert repaired == "unknown � artifact"
    assert issues[0]["action"] == "manual_review_required"
    assert issues[0]["replacement"] == ""


def test_standalone_markers_are_not_guessed():
    for text in ("standalone 鈥 marker", "standalone 锟 marker", "standalone Ã marker", "standalone Â marker", "standalone â marker"):
        repaired, issues = repair_text(text)
        assert repaired == text
        assert issues[0]["action"] == "manual_review_required"


def test_adjacent_fragments_are_not_partially_repaired():
    text = "paperââ‚¬â„¢s title"
    repaired, issues = repair_text(text)

    assert repaired == text
    assert issues
    assert all(issue["action"] == "manual_review_required" for issue in issues)


def test_numeric_plus_minus_artifact_is_contextual():
    repaired, issues = repair_text("12 卤 3 mg")
    assert repaired == "12 ± 3 mg"
    assert issues[0]["action"] == "auto_repaired"

    unchanged, ambiguous = repair_text("the character 卤 is unexplained")
    assert unchanged == "the character 卤 is unexplained"
    assert ambiguous[0]["action"] == "manual_review_required"
