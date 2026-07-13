"""Presentation-friendly access to the existing recommender engines.

The trained SVD and TF-IDF artifacts remain the scoring source.  This layer only
adds persisted viva ratings, explanations and small, transparent live metrics.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.hybrid_fusion import (
    HybridRecommender,
    UserProfile,
    get_liked_movie_ids_from_ratings,
    min_max_normalize,
)
from src.localization_config import (
    LOCALIZED_HYBRID_WEIGHTS,
    STANDARD_HYBRID_WEIGHTS,
    build_genre_weight_vector,
    compute_language_preference_scores,
)

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODEL_NAMES = {"cf", "cbf", "hybrid", "localized"}
MODEL_LABELS = {
    "cf": "Collaborative Filtering",
    "cbf": "Content-Based Filtering",
    "hybrid": "Standard Hybrid",
    "localized": "Localized Hybrid",
}
PROTO_TO_EVAL_MODELS = {
    "cf": ("CF_ColdStart", "MF_ColdStart"),
    "cbf": ("CBF",),
    "hybrid": ("NonLocal_Hybrid",),
    "localized": ("Localized_Hybrid",),
}
METRIC_COLUMN_ALIASES = {
    "precision_at_10": ["precision_at_10", "Precision@10"],
    "recall_at_10": ["recall_at_10", "Recall@10"],
    "ndcg_at_10": ["ndcg_at_10", "NDCG@10"],
    "novelty_at_10": ["novelty_at_10", "Novelty@10"],
    "filter_bubble": ["Filter_Bubble_Score", "filter_bubble_score"],
}


def _first_present(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    return next((name for name in names if name in df.columns), None)


class RecommenderService:
    def __init__(self) -> None:
        logger.info("Loading trained engines and synthetic cohort...")
        self.engine = HybridRecommender()
        self.ratings_df = pd.read_csv(PROCESSED_DIR / "synthetic_ratings.csv")
        self.users_df = pd.read_csv(PROCESSED_DIR / "synthetic_users.csv")
        self.movies_df = pd.read_csv(PROCESSED_DIR / "movies_final.csv").fillna("")
        self.movies_by_id = self.movies_df.set_index("movieId", drop=False)
        self._poster_cache: dict[int, str] = {}
        path = RESULTS_DIR / "evaluation_user_level.csv"
        self.user_level_metrics = pd.read_csv(path) if path.exists() else None
        logger.info("RecommenderService ready.")

    def get_user_profile(self, user_id: int) -> Optional[dict]:
        row = self.users_df.loc[self.users_df["userId"] == user_id]
        if row.empty:
            return None
        value = row.iloc[0]
        base_count = int((self.ratings_df["userId"] == user_id).sum())
        return {
            "userId": int(value["userId"]),
            "age": int(value["age"]),
            "gender": str(value["gender"]),
            "education": str(value["education"]),
            "archetype": str(value["archetype"]),
            "occupation": "IT student",
            "preferred_genres": [
                x for x in str(value["preferred_genres"]).split("|") if x
            ],
            "preferred_language": [
                x for x in str(value["preferred_language"]).split("|") if x
            ],
            "base_rating_count": base_count,
        }

    def _profile_obj(self, user_id: int) -> UserProfile:
        profile = self.get_user_profile(user_id)
        if profile is None:
            raise ValueError("Synthetic user not found")
        return UserProfile(
            tuple(profile["preferred_genres"]),
            tuple(profile["preferred_language"]),
            profile["archetype"],
        )

    def movie(self, movie_id: int) -> Optional[dict]:
        """Return one catalogue movie as plain JSON-friendly values."""
        if movie_id not in self.movies_by_id.index:
            return None
        selected = self.movies_by_id.loc[movie_id]
        row = selected.iloc[0] if isinstance(selected, pd.DataFrame) else selected
        if not isinstance(row, pd.Series):
            return None
        catalogue_id = self._optional_int(row.get("movieId"))
        if catalogue_id is None:
            return None
        tmdb_id = self._optional_int(row.get("tmdbId"))
        # Poster paths are intentionally optional: a TMDB API enrichment can set
        # them later; the visual fallback keeps the demonstration usable offline.
        poster_path = self._text(row.get("poster_path"))
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path
            else self._tmdb_poster_url(tmdb_id)
            if tmdb_id is not None
            else ""
        )
        return {
            "movieId": catalogue_id,
            "tmdbId": tmdb_id,
            "title": self._text(row.get("title")),
            "genres": self._text(row.get("clean_genres")),
            "language": self._text(row.get("language")),
            "overview": self._text(row.get("overview")),
            "year": self._optional_int(row.get("release_year")),
            "average_rating": self._optional_float(row.get("vote_average")),
            "popularity": self._optional_float(row.get("popularity")) or 0.0,
            "runtime": None,
            "poster_url": poster_url,
        }

    @staticmethod
    def _text(value: object) -> str:
        """Convert one scalar dataframe cell to display text."""
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    @classmethod
    def _optional_int(cls, value: object) -> int | None:
        text = cls._text(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    @classmethod
    def _optional_float(cls, value: object) -> float | None:
        text = cls._text(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _tmdb_poster_url(self, tmdb_id: int) -> str:
        """Resolve a real TMDB poster when the optional API key is configured.

        TMDB stores a poster *path*, not an image at the numeric movie id.  This
        intentionally uses the metadata endpoint first, rather than inventing a
        URL that could show an unrelated image.  Missing credentials leave the
        attractive local fallback card in place for fully offline viva demos.
        """
        if tmdb_id in self._poster_cache:
            return self._poster_cache[tmdb_id]
        api_key = os.getenv("TMDB_API_KEY")
        if not api_key:
            self._poster_cache[tmdb_id] = ""
            return ""
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                import json

                poster_path = json.loads(response.read().decode("utf-8")).get(
                    "poster_path"
                )
            value = (
                f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
            )
        except OSError, urllib.error.URLError, ValueError:
            value = ""
        self._poster_cache[tmdb_id] = value
        return value

    def top_movies(self, k: int = 12) -> list[dict]:
        data = self.movies_df.sort_values(
            ["vote_average", "popularity"], ascending=False
        ).head(k)
        movies = []
        for movie_id in data["movieId"]:
            movie = self.movie(int(movie_id))
            if movie is not None:
                movies.append(movie)
        return movies

    @staticmethod
    def _manual_ids(manual_ratings: list[dict]) -> set[int]:
        return {int(row["movie_id"]) for row in manual_ratings}

    def _liked_movie_ids(self, user_id: int, manual_ratings: list[dict]) -> list[int]:
        liked = set(
            get_liked_movie_ids_from_ratings(self.ratings_df, user_id, split="train")
        )
        liked.update(
            int(row["movie_id"])
            for row in manual_ratings
            if float(row["rating"]) >= 3.5
        )
        return sorted(liked)

    @staticmethod
    def clean_weights(requested: Optional[dict]) -> dict[str, float]:
        values = {"collaborative": 0.50, "genre": 0.18, "language": 0.12}
        if requested:
            for key in values:
                if key in requested:
                    values[key] = max(0.0, min(float(requested[key]), 0.9))
        values["content"] = max(0.05, 1.0 - sum(values.values()))
        total = sum(values.values())
        return {key: round(value / total, 4) for key, value in values.items()}

    def _localized_scores(
        self,
        user_id: int,
        liked: list[int],
        profile: UserProfile,
        weights: Optional[dict],
    ) -> tuple[np.ndarray, dict[str, float]]:
        if weights is None:
            raw = {
                "collaborative": LOCALIZED_HYBRID_WEIGHTS.cf,
                "content": LOCALIZED_HYBRID_WEIGHTS.cbf,
                "genre": LOCALIZED_HYBRID_WEIGHTS.localization * 0.6,
                "language": LOCALIZED_HYBRID_WEIGHTS.localization * 0.4,
            }
            total = sum(raw.values())
            applied = {key: value / total for key, value in raw.items()}
        else:
            applied = self.clean_weights(weights)
        cf = min_max_normalize(self.engine.get_cf_scores(user_id, liked))
        cbf = min_max_normalize(self.engine.get_cbf_scores(liked))
        # Split localization into its actual genre and language components so the
        # two controls directly affect the proposed localized model.
        genre = min_max_normalize(
            self.engine._genre_onehot @ build_genre_weight_vector(list(profile.genres))
        )
        language = min_max_normalize(
            compute_language_preference_scores(
                self.engine._movie_language, set(profile.languages)
            )
        )
        score = (
            applied["collaborative"] * cf
            + applied["content"] * cbf
            + applied["genre"] * genre
            + applied["language"] * language
        )
        return score.astype(np.float32), applied

    def _reason(
        self, model: str, movie: dict, profile: dict, liked_count: int, weights: dict
    ) -> str:
        movie_genres = set(movie["genres"].split("|"))
        matches = sorted(movie_genres & set(profile["preferred_genres"]))
        language_match = movie["language"] in profile["preferred_language"]
        if model == "cf":
            return f"SVD latent-factor score using a profile built from {liked_count} positively rated movies."
        if model == "cbf":
            return f"Content similarity to your positively rated movies; matches {', '.join(matches) or 'their metadata profile'}."
        if model == "hybrid":
            return f"Combined SVD ({STANDARD_HYBRID_WEIGHTS.cf:.3g}) and content similarity ({STANDARD_HYBRID_WEIGHTS.cbf:.3g})."
        parts = []
        if matches:
            parts.append(f"genre match: {', '.join(matches)}")
        if language_match:
            parts.append(f"language match: {movie['language']}")
        suffix = "; ".join(parts) or "localized profile contribution"
        return f"{suffix}; localized weights CF {weights['collaborative']:.2f}, genre {weights['genre']:.2f}, language {weights['language']:.2f}."

    def recommend(
        self,
        user_id: int,
        model: str,
        manual_ratings: Optional[list[dict]] = None,
        weights: Optional[dict] = None,
        k: int = 10,
    ) -> tuple[list[dict], dict[str, float]]:
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{model}'.")
        profile = self.get_user_profile(user_id)
        if profile is None:
            raise ValueError("Synthetic user not found.")
        manual_ratings = manual_ratings or []
        liked = self._liked_movie_ids(user_id, manual_ratings)
        if model == "cf":
            scores, applied = (
                self.engine.get_cf_scores(user_id, liked),
                self.clean_weights(None),
            )
        elif model == "cbf":
            scores, applied = (
                self.engine.get_cbf_scores(liked),
                self.clean_weights(None),
            )
        elif model == "hybrid":
            scores, applied = (
                self.engine.get_standard_hybrid_scores(user_id, liked),
                self.clean_weights(None),
            )
        else:
            scores, applied = self._localized_scores(
                user_id, liked, self._profile_obj(user_id), weights
            )
        exclude = set(liked) | self._manual_ids(manual_ratings)
        raw = self.engine.recommend_from_scores(
            scores, exclude_movie_ids=list(exclude), k=k
        )
        recommendations = []
        for rank, item in enumerate(raw, start=1):
            movie = self.movie(item.movie_id) or {
                "movieId": item.movie_id,
                "title": item.title,
                "genres": item.genres,
                "language": item.language,
                "poster_url": "",
                "year": None,
                "average_rating": None,
            }
            movie.update(
                {
                    "rank": rank,
                    "score": round(float(item.score), 4),
                    "reason": self._reason(model, movie, profile, len(liked), applied),
                }
            )
            recommendations.append(movie)
        return recommendations, applied

    def live_metrics(
        self, user_id: int, recommendations: list[dict]
    ) -> dict[str, float]:
        profile = self.get_user_profile(user_id)
        if profile is None:
            return {}
        top = [item["movieId"] for item in recommendations]
        positives = set(
            self.ratings_df.loc[
                (self.ratings_df.userId == user_id)
                & (self.ratings_df.split == "holdout")
                & (self.ratings_df.rating >= 3.5),
                "movieId",
            ].astype(int)
        )
        hits = [movie_id for movie_id in top if movie_id in positives]
        dcg = sum(
            1 / np.log2(index + 2)
            for index, movie_id in enumerate(top)
            if movie_id in positives
        )
        ideal = sum(1 / np.log2(index + 2) for index in range(min(10, len(positives))))
        genres = {
            genre
            for item in recommendations
            for genre in item["genres"].split("|")
            if genre
        }
        languages = {item["language"] for item in recommendations if item["language"]}
        outside_profile = sum(
            1
            for item in recommendations
            if not (set(item["genres"].split("|")) & set(profile["preferred_genres"]))
            and item["language"] not in profile["preferred_language"]
        )
        return {
            "precision_at_10": round(len(hits) / 10, 4),
            "recall_at_10": round(len(hits) / len(positives), 4) if positives else 0.0,
            "ndcg_at_10": float(round(dcg / ideal, 4)) if ideal else 0.0,
            "coverage": round(len(set(top)) / self.engine.n_movies, 4),
            "genre_diversity": len(genres),
            "language_diversity": len(languages),
            "diversity": len(genres) + len(languages),
            "novelty": round(outside_profile / max(len(recommendations), 1), 4),
        }

    def get_user_metrics(self, user_id: int, model: str) -> Optional[dict]:
        """Return the reproducible precomputed metric for legacy RQ pages."""
        df = self.user_level_metrics
        if df is None:
            return None
        user_col, model_col = (
            _first_present(df, ["UserId", "userId", "user_id"]),
            _first_present(df, ["model", "model_name", "Model"]),
        )
        if user_col is None or model_col is None:
            return None
        row = df[
            (df[user_col] == user_id)
            & df[model_col].isin(PROTO_TO_EVAL_MODELS.get(model, (model,)))
        ]
        if row.empty:
            return None
        value = row.iloc[0]
        return {
            key: round(float(value[column]), 4)
            if (column := _first_present(df, aliases))
            else None
            for key, aliases in METRIC_COLUMN_ALIASES.items()
        }


_service: Optional[RecommenderService] = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service
