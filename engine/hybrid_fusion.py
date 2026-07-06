from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .collaborative_filtering import load_model, MODEL_PATH
from .content_filter import ContentBasedFilter, CBF_MATRIX_PATH, CBF_METADATA_PATH
from src.localization_config import (
    LOCALIZATION_GENRE_WEIGHT,
    LOCALIZATION_LANGUAGE_WEIGHT,
    STANDARD_HYBRID_WEIGHTS,
    LOCALIZED_HYBRID_WEIGHTS,
    build_genre_weight_vector,
    build_genre_onehot_from_list,
    compute_language_preference_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UserProfile:
    genres: tuple[str, ...]
    languages: tuple[str, ...]
    archetype: str | None = None


@dataclass(slots=True)
class Recommendation:
    movie_id: int
    title: str
    genres: str
    language: str
    score: float


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def get_liked_movie_ids_from_ratings(
    ratings_df: pd.DataFrame,
    user_id: int,
    split: str = "train",
    min_rating: float = 3.5,
) -> list[int]:

    mask = (
        (ratings_df["userId"] == user_id)
        & (ratings_df["split"] == split)
        & (ratings_df["rating"] >= min_rating)
    )
    return ratings_df.loc[mask, "movieId"].astype(int).tolist()


class HybridRecommender:
    def __init__(
        self,
        cf_model_path: str = MODEL_PATH,
        cbf_matrix_path: str = CBF_MATRIX_PATH,
        cbf_meta_path: str = CBF_METADATA_PATH,
    ) -> None:
        self.cf_algo = load_model(cf_model_path)
        self.cbf = ContentBasedFilter(cbf_matrix_path, cbf_meta_path)

        self.movie_ids = self.cbf.movie_ids
        self.titles = self.cbf.titles
        self.clean_genres = self.cbf.clean_genres
        self.language = self.cbf.language
        self.n_movies = self.cbf.n_movies

        self._genre_onehot = build_genre_onehot_from_list(self.clean_genres)
        self._movie_language = np.asarray(self.language)
        self._inner_iid_map = self._build_inner_iid_map()

        logger.info("HybridRecommender ready: %d movies.", self.n_movies)

    def _build_inner_iid_map(self) -> np.ndarray:
        trainset = self.cf_algo.trainset
        inner = np.full(self.n_movies, -1, dtype=np.int64)
        for idx, mid in enumerate(self.movie_ids):
            try:
                inner[idx] = trainset.to_inner_iid(int(mid))
            except ValueError:
                continue
        return inner

    def get_cf_scores(self, user_id: int) -> np.ndarray:

        trainset = self.cf_algo.trainset
        global_mean = trainset.global_mean
        scores = np.full(self.n_movies, global_mean, dtype=np.float32)

        try:
            inner_uid = trainset.to_inner_uid(int(user_id))
        except ValueError:
            return scores

        bu = self.cf_algo.bu[inner_uid]
        pu = self.cf_algo.pu[inner_uid]

        valid_mask = self._inner_iid_map >= 0
        valid_inner_iids = self._inner_iid_map[valid_mask]
        bi = self.cf_algo.bi[valid_inner_iids]
        qi = self.cf_algo.qi[valid_inner_iids]

        scores[valid_mask] = global_mean + bu + bi + qi @ pu
        return scores.astype(np.float32)

    def get_cbf_scores(self, liked_movie_ids: list[int]) -> np.ndarray:
        return self.cbf.get_content_scores(liked_movie_ids)

    def get_localization_scores(self, profile: UserProfile) -> np.ndarray:

        genre_vec = build_genre_weight_vector(list(profile.genres))
        genre_score = min_max_normalize(self._genre_onehot @ genre_vec)

        lang_score = min_max_normalize(
            compute_language_preference_scores(
                self._movie_language, set(profile.languages)
            )
        )

        return (
            LOCALIZATION_GENRE_WEIGHT * genre_score
            + LOCALIZATION_LANGUAGE_WEIGHT * lang_score
        ).astype(np.float32)

    def get_standard_hybrid_scores(
        self, user_id: int, liked_movie_ids: list[int]
    ) -> np.ndarray:

        w = STANDARD_HYBRID_WEIGHTS
        cf = min_max_normalize(self.get_cf_scores(user_id))
        cbf = min_max_normalize(self.get_cbf_scores(liked_movie_ids))
        return w.cf * cf + w.cbf * cbf

    def get_localized_hybrid_scores(
        self,
        user_id: int,
        liked_movie_ids: list[int],
        profile: UserProfile,
    ) -> np.ndarray:

        w = LOCALIZED_HYBRID_WEIGHTS
        cf = min_max_normalize(self.get_cf_scores(user_id))
        cbf = min_max_normalize(self.get_cbf_scores(liked_movie_ids))
        loc = self.get_localization_scores(profile)
        return w.cf * cf + w.cbf * cbf + w.localization * loc

    def recommend_from_scores(
        self, scores: np.ndarray, exclude_movie_ids: list[int], k: int = 10
    ) -> list[Recommendation]:
        exclude = set(exclude_movie_ids)
        ranked_idx = np.argsort(-scores)
        results: list[Recommendation] = []
        for idx in ranked_idx:
            movie_id = int(self.movie_ids[idx])
            if movie_id in exclude:
                continue
            results.append(
                Recommendation(
                    movie_id=movie_id,
                    title=self.titles[idx],
                    genres=self.clean_genres[idx],
                    language=self.language[idx],
                    score=float(scores[idx]),
                )
            )
            if len(results) >= k:
                break
        return results


def main() -> None:
    engine = HybridRecommender()

    demo_user_id = 1
    demo_liked = [1, 2571]
    demo_profile = UserProfile(
        genres=("Sci-Fi", "Action"),
        languages=("English", "Hindi"),
        archetype="hollywood",
    )

    for label, scores in [
        ("Model 1: CF", engine.get_cf_scores(demo_user_id)),
        ("Model 2: CBF", engine.get_cbf_scores(demo_liked)),
        (
            "Model 3: Standard Hybrid",
            engine.get_standard_hybrid_scores(demo_user_id, demo_liked),
        ),
        (
            "Model 4: Localized Hybrid",
            engine.get_localized_hybrid_scores(demo_user_id, demo_liked, demo_profile),
        ),
    ]:
        print(f"\n{label}")
        for r in engine.recommend_from_scores(
            scores, exclude_movie_ids=demo_liked, k=10
        ):
            print(
                f"  {r.movie_id:>7}  {r.title[:40]:<40}  {r.genres:<25}  {r.language:<10}  {r.score:.4f}"
            )


if __name__ == "__main__":
    main()
