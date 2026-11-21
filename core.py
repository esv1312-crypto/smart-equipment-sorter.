"""Вспомогательная логика визуального поиска по эмбеддингам."""
from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".idea", ".vscode",
}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity для уже нормализованных векторов."""
    if len(a) != len(b) or not a:
        raise ValueError("Векторы должны иметь одинаковую ненулевую длину")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def rank_categories(
    image_vector: Sequence[float],
    category_vectors: Mapping[str, Sequence[Sequence[float]]],
) -> list[tuple[str, float]]:
    """Ранжирует категории; оценка категории — лучший score среди её образцов."""
    ranked = []
    for category, references in category_vectors.items():
        scores = sorted(
            (cosine(image_vector, ref) for ref in references if len(ref) == len(image_vector)),
            reverse=True,
        )
        if scores:
            # With multiple examples, average the two closest matches. This reduces
            # accidental high scores caused by a single unusual reference photo.
            top_scores = scores[:2]
            ranked.append((category, sum(top_scores) / len(top_scores)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def classify_embedding(
    image_vector: Sequence[float],
    category_vectors: Mapping[str, Sequence[Sequence[float]]],
    threshold: float,
    min_margin: float = 0.035,
) -> tuple[str, float, float | None, list[tuple[str, float]]]:
    """Возвращает категорию или «На проверку», лучший score, margin и рейтинг."""
    ranked = rank_categories(image_vector, category_vectors)
    if not ranked:
        return "На проверку", -1.0, None, []
    best_category, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else None
    if best_score < threshold:
        return "На проверку", best_score, margin, ranked
    if margin is not None and margin < min_margin:
        return "На проверку", best_score, margin, ranked
    return best_category, best_score, margin, ranked


def iter_image_files(
    source: Path,
    extensions: Iterable[str],
    excluded_dirs: Iterable[Path] = (),
):
    """Рекурсивно перечисляет изображения, не заходя в служебные и исключённые папки."""
    source = Path(source).resolve()
    extensions = {ext.lower() for ext in extensions}
    excluded = []
    for item in excluded_dirs:
        try:
            excluded.append(Path(item).expanduser().resolve())
        except OSError:
            continue

    for root, dirs, files in os.walk(source):
        root_path = Path(root).resolve()
        kept = []
        for dirname in dirs:
            candidate = (root_path / dirname).resolve()
            if dirname.lower() in SKIP_DIR_NAMES:
                continue
            if any(candidate == ex or ex in candidate.parents for ex in excluded):
                continue
            kept.append(dirname)
        dirs[:] = kept
        for filename in files:
            path = root_path / filename
            if path.suffix.lower() in extensions and path.is_file():
                yield path


class EmbeddingCache:
    """SQLite-кеш эмбеддингов, привязанный к пути, версии модели и метаданным файла."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path), timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                path TEXT NOT NULL,
                model_id TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (path, model_id)
            )
        """)
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def _identity(path: Path):
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        return os.path.normcase(str(resolved)).casefold(), stat.st_size, stat.st_mtime_ns

    def get(self, path: Path, model_id: str):
        try:
            key, size, mtime_ns = self._identity(path)
            row = self.connection.execute(
                "SELECT file_size, mtime_ns, dimension, vector FROM embeddings "
                "WHERE path = ? AND model_id = ?", (key, model_id)
            ).fetchone()
            if row is None or row[0] != size or row[1] != mtime_ns:
                self.misses += 1
                return None
            dimension, blob = row[2], row[3]
            if dimension <= 0 or len(blob) != dimension * 4:
                self.connection.execute(
                    "DELETE FROM embeddings WHERE path = ? AND model_id = ?", (key, model_id)
                )
                self.connection.commit()
                self.misses += 1
                return None
            vector = list(struct.unpack(f"<{dimension}f", blob))
            self.hits += 1
            return vector
        except (OSError, sqlite3.Error, ValueError, struct.error):
            self.misses += 1
            return None

    def put(self, path: Path, model_id: str, vector):
        try:
            key, size, mtime_ns = self._identity(path)
            values = [float(value) for value in vector]
            if not values:
                return False
            blob = struct.pack(f"<{len(values)}f", *values)
            self.connection.execute(
                "INSERT INTO embeddings(path, model_id, file_size, mtime_ns, dimension, vector) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(path, model_id) DO UPDATE SET "
                "file_size=excluded.file_size, mtime_ns=excluded.mtime_ns, "
                "dimension=excluded.dimension, vector=excluded.vector",
                (key, model_id, size, mtime_ns, len(values), sqlite3.Binary(blob)),
            )
            self.connection.commit()
            self.writes += 1
            return True
        except (OSError, sqlite3.Error, ValueError, struct.error):
            return False

    def close(self):
        self.connection.close()
