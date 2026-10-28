import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_article_overlap_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("article_overlap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ArticleOverlapSensitivityTests(unittest.TestCase):
    def test_conservative_identifier_normalization(self):
        self.assertEqual(MODULE.normalize_doi(" DOI:10.1000/ABC "), ("10.1000/abc", True))
        self.assertEqual(MODULE.normalize_doi("https://doi.org/10.1000/ABC"), ("10.1000/abc", True))
        self.assertEqual(MODULE.normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/12345/"), ("12345", True))
        self.assertEqual(MODULE.normalize_openalex("https://openalex.org/W123"), ("w123", True))
        self.assertEqual(MODULE.normalize_pmid("not-a-pmid"), ("", False))

    def test_metrics_use_frozen_threshold_and_handle_undefined_auc(self):
        rows = [
            {"true_label": 1, "predicted_probability": 0.9},
            {"true_label": 1, "predicted_probability": 0.4},
            {"true_label": 0, "predicted_probability": 0.6},
            {"true_label": 0, "predicted_probability": 0.1},
        ]
        result = MODULE.calculate_metrics(rows, 0.5)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        one_class = MODULE.calculate_metrics(rows[:2], 0.5)
        self.assertIsNone(one_class["roc_auc"])


if __name__ == "__main__":
    unittest.main()
