"""Вспомогательная логика визуального поиска по эмбеддингам."""
from __future__ import annotations

import os
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
