from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.hybrid_fusion import (
    HybridRecommender,
    UserProfile,
    get_liked_movie_ids_from_ratings,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

MODEL_NAMES = {"cf", "cbf", "hybrid", "localized"}
MODEL_LABELS = {
    "cf": "CF (Cold-start)",
    "cbf": "Content-Based Filtering",
    "hybrid": "Standard Hybrid",
    "localized": "Localized Hybrid",
}

# The first value is the current evaluation artifact name.  The legacy value
# keeps the UI usable with results produced before the terminology change.
PROTO_TO_EVAL_MODELS: dict[str, tuple[str, ...]] = {
    "cf": ("CF_ColdStart", "MF_ColdStart"),
    "cbf": ("CBF",),
    "hybrid": ("NonLocal_Hybrid",),
    "localized": ("Localized_Hybrid",),
}

# evaluation_metrics.py may name these columns differently; add variants seen.
METRIC_COLUMN_ALIASES = {
    "precision_at_10": ["precision_at_10", "Precision@10", "precision@10"],
    "recall_at_10": ["recall_at_10", "Recall@10", "recall@10"],
    "ndcg_at_10": ["ndcg_at_10", "NDCG@10", "ndcg@10"],
    "novelty_at_10": ["novelty_at_10", "Novelty@10", "novelty@10"],
    "filter_bubble": ["Filter_Bubble_Score", "filter_bubble_score", "filter_bubble"],
}
USER_COL_ALIASES = ["UserId", "userId", "user_id"]
MODEL_COL_ALIASES = ["model", "model_name", "Model"]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class RecommenderService:
    def __init__(self) -> None:
        logger.info("Loading trained engines and synthetic cohort...")
        self.engine = HybridRecommender()
        self.ratings_df = pd.read_csv(PROCESSED_DIR / "synthetic_ratings.csv")
        self.users_df = pd.read_csv(PROCESSED_DIR / "synthetic_users.csv")
        self.user_level_metrics = self._load_csv_safe(
            RESULTS_DIR / "evaluation_user_level.csv"
        )
        logger.info("RecommenderService ready.")

    @staticmethod
    def _load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    def get_user_profile(self, user_id: int) -> Optional[dict]:
        row = self.users_df.loc[self.users_df["userId"] == user_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "userId": int(r["userId"]),
            "age": int(r["age"]),
            "gender": r["gender"],
            "education": r["education"],
            "archetype": r["archetype"],
            "preferred_genres": str(r["preferred_genres"]).split("|"),
            "preferred_language": str(r["preferred_language"]).split("|"),
        }

    def _profile_obj(self, user_id: int) -> UserProfile:
        p = self.get_user_profile(user_id)
        assert p is not None
        return UserProfile(
            genres=tuple(p["preferred_genres"]),
            languages=tuple(p["preferred_language"]),
            archetype=p["archetype"],
        )

    def _liked_movie_ids(self, user_id: int) -> list[int]:
        return get_liked_movie_ids_from_ratings(self.ratings_df, user_id, split="train")

    def recommend(self, user_id: int, model: str, k: int = 10) -> list[dict]:
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{model}'. Choose from {MODEL_NAMES}.")

        liked = self._liked_movie_ids(user_id)

        if model == "cf":
            scores = self.engine.get_cf_scores(user_id, liked)
        elif model == "cbf":
            scores = self.engine.get_cbf_scores(liked)
        elif model == "hybrid":
            scores = self.engine.get_standard_hybrid_scores(user_id, liked)
        else:  # localized
            profile = self._profile_obj(user_id)
            scores = self.engine.get_localized_hybrid_scores(user_id, liked, profile)

        recs = self.engine.recommend_from_scores(scores, exclude_movie_ids=liked, k=k)
        return [
            {
                "rank": i + 1,
                "movieId": r.movie_id,
                "title": r.title,
                "genres": r.genres,
                "language": r.language,
                "score": round(r.score, 4),
            }
            for i, r in enumerate(recs)
        ]

    def get_user_metrics(self, user_id: int, model: str) -> Optional[dict]:
        df = self.user_level_metrics
        if df is None:
            return None

        user_col = _first_present(df, USER_COL_ALIASES)
        model_col = _first_present(df, MODEL_COL_ALIASES)
        if user_col is None or model_col is None:
            return None

        eval_model_names = PROTO_TO_EVAL_MODELS.get(model, (model,))
        row = df[
            (df[user_col] == user_id) & df[model_col].isin(eval_model_names)
        ]
        if row.empty:
            return None
        r = row.iloc[0]

        out = {}
        for key, aliases in METRIC_COLUMN_ALIASES.items():
            col = _first_present(df, aliases)
            out[key] = round(float(r[col]), 4) if col is not None else None
        return out


_service: Optional[RecommenderService] = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service
