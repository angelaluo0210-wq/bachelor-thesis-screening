import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prepare_dataset import load_eligibility_criteria, match_criteria_to_reviews

CRITERIA_PATH = ROOT / "data" / "eligibility_criteria(排纳标准).xlsx"


def test_eligibility_excel_has_expected_structure():
    criteria_df = load_eligibility_criteria(CRITERIA_PATH)

    required_fields = {
        "review_key",
        "eligibility_criteria",
        "inclusion_criteria",
        "exclusion_criteria",
    }
    assert required_fields.issubset(set(criteria_df.columns))
    assert len(criteria_df) == 26
    assert criteria_df["review_key"].nunique() == 26


def test_criteria_match_coverage():
    # Training-stage coverage only. Never load the sealed test split or all raw
    # review articles just to test review-key matching.
    article_df = pd.concat([
        pd.read_csv(ROOT / "data" / "processed" / "splits" / f"{name}.csv", usecols=["review_key"])
        for name in ("train", "validation")
    ], ignore_index=True).rename(columns={"review_key": "review_id"})
    criteria_df = load_eligibility_criteria(CRITERIA_PATH)
    match_df = match_criteria_to_reviews(article_df, criteria_df)

    assert set(match_df["match_status"].unique()) <= {"exact", "normalized", "unmatched"}
    assert int((match_df["match_status"] == "unmatched").sum()) == 0
    assert match_df["review_id_article_data"].nunique() == len(article_df["review_id"].dropna().unique())
    assert match_df["review_id_article_data"].nunique() == 22
