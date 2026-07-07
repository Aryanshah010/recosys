from __future__ import annotations

import logging
import os

from typing import cast

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
CBF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "cbf_matrix.pkl")
CBF_METADATA_PATH = os.path.join(PROCESSED_DIR, "cbf_metadata.pkl")

REQUIRED_METADATA_KEYS = [
    "movie_ids",
    "movie_index",
    "titles",
    "clean_genres",
    "language",
    "n_movies",
]

QUALITY_BLEND_ALPHA: float = 0.15


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. Run build_cbf_matrix.py first."
        )


def load_cbf_artifacts(
    matrix_path: str = CBF_MATRIX_PATH,
    meta_path: str = CBF_METADATA_PATH,
) -> tuple[csr_matrix, dict[str, object]]:

    check_file_exists(matrix_path)
    check_file_exists(meta_path)

    matrix = joblib.load(matrix_path)
    metadata = joblib.load(meta_path)

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(f"cbf_metadata.pkl is missing required keys: {missing}")

    if matrix.shape[0] != metadata["n_movies"]:
        raise ValueError(
            f"TF-IDF matrix rows ({matrix.shape[0]}) do not match "
            f"metadata n_movies ({metadata['n_movies']})."
        )

    return matrix, metadata


class ContentBasedFilter:
    def __init__(
        self,
        matrix_path: str = CBF_MATRIX_PATH,
        meta_path: str = CBF_METADATA_PATH,
    ) -> None:
        self.matrix, self.metadata = load_cbf_artifacts(matrix_path, meta_path)
        self.movie_index = cast(dict[int, int], self.metadata["movie_index"])
        self.movie_ids = cast(np.ndarray, self.metadata["movie_ids"])
        self.titles = cast(list[str], self.metadata["titles"])
        self.clean_genres = cast(list[str], self.metadata["clean_genres"])
        self.language = cast(list[str], self.metadata["language"])
        self.n_movies = cast(int, self.metadata["n_movies"])

        if "quality_scores" in self.metadata:
            self.quality_scores = cast(
                np.ndarray, self.metadata["quality_scores"]
            ).astype(np.float32)
            logger.info(
                "Loaded quality_scores from metadata (non-zero: %d / %d).",
                int((self.quality_scores > 0).sum()),
                self.n_movies,
            )
        else:
            logger.warning(
                "quality_scores not found in cbf_metadata.pkl — "
                "falling back to zeros. Re-run build_cbf_matrix.py to enable "
                "quality-based re-ranking."
            )
            self.quality_scores = np.zeros(self.n_movies, dtype=np.float32)

        logger.info("Loaded CBF matrix: %d movies.", self.n_movies)

    def _resolve_indices(self, movie_ids: list[int]) -> list[int]:
        idxs, missing = [], []
        for m in movie_ids:
            idx = self.movie_index.get(m)
            if idx is None:
                missing.append(m)
            else:
                idxs.append(idx)
        if missing:
            logger.warning(
                "%d/%d movieId(s) not in CBF catalog, skipped: %s",
                len(missing),
                len(movie_ids),
                missing[:10],
            )
        return idxs

    def get_content_scores(self, liked_movie_ids: list[int]) -> np.ndarray:

        idxs = self._resolve_indices(liked_movie_ids)
        if not idxs:
            return np.zeros(self.n_movies, dtype=np.float32)

        raw_sims = linear_kernel(self.matrix[idxs], self.matrix).mean(axis=0)

        lo, hi = raw_sims.min(), raw_sims.max()
        if hi - lo > 1e-8:
            norm_sims = (raw_sims - lo) / (hi - lo)
        else:
            norm_sims = np.zeros_like(raw_sims)

        blended = (
            1.0 - QUALITY_BLEND_ALPHA
        ) * norm_sims + QUALITY_BLEND_ALPHA * self.quality_scores
        return blended.astype(np.float32)

    def recommend(self, liked_movie_ids: list[int], k: int = 10) -> list[dict]:
        scores = self.get_content_scores(liked_movie_ids)
        exclude = set(liked_movie_ids)

        ranked_idx = np.argsort(-scores)
        results: list[dict] = []
        for idx in ranked_idx:
            movie_id = int(self.movie_ids[idx])
            if movie_id in exclude:
                continue
            results.append(
                {
                    "movieId": movie_id,
                    "title": self.titles[idx],
                    "clean_genres": self.clean_genres[idx],
                    "language": self.language[idx],
                    "score": float(scores[idx]),
                }
            )
            if len(results) >= k:
                break
        return results


def main() -> None:
    engine = ContentBasedFilter()

    demo_liked = [
        1,
        2571,
    ]  # Toy Story
    recs = engine.recommend(demo_liked, k=10)

    print(f"\nContent-based recommendations for liked={demo_liked}:")
    for r in recs:
        print(
            f"  {r['movieId']:>7}  {r['title'][:40]:<40}  {r['clean_genres']:<25}  {r['language']:<10}  {r['score']:.4f}"
        )


if __name__ == "__main__":
    main()
