import csv
import tempfile
import unittest
from pathlib import Path

from src.split_dataset import (
    DatasetSplitError,
    SPLIT_REVIEWS,
    identify_column,
    read_csv,
    run,
)


FIELDS = ["title", "abstract", "label_included", "review_key", "audit_note"]


def write_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class SplitDatasetTests(unittest.TestCase):
    def valid_rows(self) -> list[dict[str, str]]:
        rows = []
        for split_name, reviews in SPLIT_REVIEWS.items():
            for review in sorted(reviews):
                rows.append(
                    {
                        "title": f"{review} title",
                        "abstract": "" if review == "Bos_2018" else "abstract",
                        "label_included": "1" if split_name == "test" else "0",
                        "review_key": review,
                        "audit_note": "preserve me",
                    }
                )
        rows.append(dict(rows[0]))  # duplicate multiplicity must be preserved
        return rows

    def test_fixed_split_preserves_rows_columns_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "all_reviews_cleaned.csv"
            output = root / "splits"
            expected = self.valid_rows()
            write_fixture(source, expected)

            created = run(source, output)

            self.assertEqual(len(created), 4)
            combined = []
            for name in ("train", "validation", "test"):
                fields, rows = read_csv(output / f"{name}.csv")
                self.assertEqual(fields, FIELDS)
                self.assertEqual({row["review_key"] for row in rows}, SPLIT_REVIEWS[name])
                combined.extend(rows)
            self.assertCountEqual(combined, expected)
            _, summary = read_csv(output / "split_summary.csv")
            self.assertEqual(len(summary), 26)

    def test_unassigned_review_fails_before_creating_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "all_reviews_cleaned.csv"
            output = root / "splits"
            rows = self.valid_rows()
            rows[0]["review_key"] = "unexpected_review"
            write_fixture(source, rows)

            with self.assertRaisesRegex(DatasetSplitError, "Unassigned input review"):
                run(source, output)
            self.assertFalse(output.exists())

    def test_ambiguous_review_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetSplitError, "Cannot uniquely identify review"):
            identify_column(
                ["review_id", "review_key", "label_included"],
                ("review_id", "review", "dataset", "review_key"),
                "review",
            )


if __name__ == "__main__":
    unittest.main()
