import tempfile
import unittest
from pathlib import Path

from core import classify_embedding, iter_image_files, rank_categories


class RankingTests(unittest.TestCase):
    def test_best_category_is_selected_when_clear(self):
        vectors = {"Hydrodrill": [[1.0, 0.0]], "Hammer": [[0.0, 1.0]]}
        result = classify_embedding([0.98, 0.02], vectors, threshold=0.5, min_margin=0.1)
        self.assertEqual(result[0], "Hydrodrill")

    def test_close_categories_go_to_review(self):
        vectors = {"A": [[1.0, 0.0]], "B": [[0.99, 0.01]]}
        result = classify_embedding([1.0, 0.0], vectors, threshold=0.5, min_margin=0.05)
        self.assertEqual(result[0], "На проверку")

    def test_low_score_goes_to_review(self):
        vectors = {"A": [[1.0, 0.0]], "B": [[0.0, 1.0]]}
        result = classify_embedding([-1.0, 0.0], vectors, threshold=0.1)
        self.assertEqual(result[0], "На проверку")


    def test_two_best_examples_are_averaged(self):
        vectors = {"A": [[1.0, 0.0], [0.0, 1.0]], "B": [[0.7, 0.714142842]]}
        ranked = rank_categories([1.0, 0.0], vectors)
        scores = dict(ranked)
        self.assertAlmostEqual(scores["A"], 0.5, places=6)
        self.assertGreater(scores["B"], scores["A"])

    def test_ranking_is_descending(self):
        ranked = rank_categories([1.0, 0.0], {"B": [[0.0, 1.0]], "A": [[1.0, 0.0]]})
        self.assertEqual(ranked[0][0], "A")


class FileScanTests(unittest.TestCase):
    def test_excludes_output_and_service_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "good").mkdir()
            (root / "output").mkdir()
            (root / ".git").mkdir()
            (root / "good" / "one.jpg").write_bytes(b"x")
            (root / "output" / "two.jpg").write_bytes(b"x")
            (root / ".git" / "three.jpg").write_bytes(b"x")
            found = list(iter_image_files(root, {".jpg"}, [root / "output"]))
            self.assertEqual([p.name for p in found], ["one.jpg"])


if __name__ == "__main__":
    unittest.main()
