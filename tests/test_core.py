import tempfile
import unittest
from pathlib import Path

from core import EmbeddingCache, classify_embedding, iter_image_files, rank_categories


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


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_miss_then_hit_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "sample.jpg"
            image.write_bytes(b"image-bytes")
            cache = EmbeddingCache(root / "cache.sqlite3")
            self.assertIsNone(cache.get(image, "model-v1"))
            vector = [0.125, -0.5, 0.875]
            self.assertTrue(cache.put(image, "model-v1", vector))
            self.assertEqual(cache.get(image, "model-v1"), vector)
            self.assertEqual(cache.hits, 1)
            cache.close()

    def test_file_change_invalidates_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "sample.jpg"
            image.write_bytes(b"old")
            cache = EmbeddingCache(root / "cache.sqlite3")
            cache.put(image, "model-v1", [1.0, 0.0])
            image.write_bytes(b"new content")
            self.assertIsNone(cache.get(image, "model-v1"))
            cache.close()

    def test_model_id_is_part_of_cache_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "sample.jpg"
            image.write_bytes(b"image")
            cache = EmbeddingCache(root / "cache.sqlite3")
            cache.put(image, "model-v1", [1.0, 0.0])
            self.assertIsNone(cache.get(image, "model-v2"))
            cache.close()

    def test_corrupt_vector_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "sample.jpg"
            image.write_bytes(b"image")
            cache = EmbeddingCache(root / "cache.sqlite3")
            cache.put(image, "model-v1", [1.0, 0.0])
            cache.connection.execute(
                "UPDATE embeddings SET vector = ?",
                (b"broken",)
            )
            cache.connection.commit()
            self.assertIsNone(cache.get(image, "model-v1"))
            cache.close()


if __name__ == "__main__":
    unittest.main()
