import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mojibake_label1_qc import is_valid_accented_latin, recover_from_reference
from src.mojibake_qc import repair_text


def test_reference_alignment_recovers_only_damaged_span():
    prefix = "This randomized controlled trial used strict eligibility criteria. "
    suffix = " All participants completed the planned follow-up assessment."
    local = prefix + "Patients aged �?8 years were included." + suffix
    reference = prefix + "Patients aged ≥18 years were included." + suffix

    repaired, repairs = recover_from_reference(local, reference)

    assert repaired == reference
    assert repairs[0]["suspicious_pattern"] == "�?"
    assert repairs[0]["replacement"] == "≥1"


def test_reference_alignment_rejects_low_similarity_text():
    repaired, repairs = recover_from_reference("value �?5", "unrelated abstract text")

    assert repaired == "value �?5"
    assert repairs == []


def test_reference_alignment_requires_all_replacement_characters_to_align():
    repaired, repairs = recover_from_reference("A � and B �", "A ≥ and something wholly different")

    assert repaired == "A � and B �"
    assert repairs == []


def test_valid_portuguese_accent_is_preserved():
    text = "radiological accident in Goiânia, Brazil"
    _, issues = repair_text(text)

    assert is_valid_accented_latin(text, issues)
