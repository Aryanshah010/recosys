from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from engine.hybrid_fusion import (
    HybridRecommender,
    UserProfile,
    get_liked_movie_ids_from_ratings,
)
from evaluation.evaluation_metrics import (
    filter_bubble_score,
    genre_diversity,
    language_diversity,
)
from src.localization_config import (
    AFFINITY_ONLY_WEIGHTS,
    LOCALIZED_HYBRID_WEIGHTS,
    STANDARD_HYBRID_WEIGHTS,
)

load_dotenv()

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")
POSTER_CACHE_PATH = PROCESSED_DIR / "poster_cache.json"
POSTER_TIMEOUT_SECONDS = 2
POSTER_MAX_WORKERS = 8
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


@dataclass(frozen=True, slots=True)
class ModelSpec:

    key: str
    label: str
    evaluation_name: str
    weights: dict[str, float]
    description: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "popularity",
        "Popularity",
        "Popularity",
        {},
        "Bayesian weighted-rating baseline; identical for every user.",
    ),
    ModelSpec(
        "cf",
        "CF Cold-Start",
        "CF_ColdStart",
        {"cf": 1.0},
        "SVD latent factors. The cohort is unseen by the trainset by design, so "
        "scores come from averaged item factors of the user's liked titles.",
    ),
    ModelSpec(
        "cbf",
        "Content-Based",
        "CBF",
        {"cbf": 1.0},
        "TF-IDF cosine similarity over genres and synopses.",
    ),
    ModelSpec(
        "affinity",
        "Affinity Only",
        "Affinity_Only",
        {"affinity": AFFINITY_ONLY_WEIGHTS.localization},
        "Ablation: cohort affinity alone, with no CF or CBF contribution.",
    ),
    ModelSpec(
        "hybrid",
        "Standard Hybrid",
        "NonLocal_Hybrid",
        {"cf": STANDARD_HYBRID_WEIGHTS.cf, "cbf": STANDARD_HYBRID_WEIGHTS.cbf},
        "CF and CBF blended at 5:3, with no cohort signal.",
    ),
    ModelSpec(
        "localized",
        "Localized Hybrid",
        "Localized_Hybrid",
        {
            "cf": LOCALIZED_HYBRID_WEIGHTS.cf,
            "cbf": LOCALIZED_HYBRID_WEIGHTS.cbf,
            "affinity": LOCALIZED_HYBRID_WEIGHTS.localization,
        },
        "The standard hybrid plus a cohort-affinity term. Exactly "
        "0.8 x standard + 0.2 x affinity, so any difference is attributable to "
        "affinity.",
    ),
)

# Derived views, kept so callers and tests read the same names as before.
MODEL_ORDER = [spec.key for spec in MODEL_SPECS]
MODEL_LABELS = {spec.key: spec.label for spec in MODEL_SPECS}
EVALUATION_MODEL_NAMES = {spec.key: spec.evaluation_name for spec in MODEL_SPECS}
MODEL_WEIGHTS = {spec.key: spec.weights for spec in MODEL_SPECS}
MODEL_DESCRIPTIONS = {spec.key: spec.description for spec in MODEL_SPECS}


class RecommenderService:
    def __init__(self) -> None:
        logger.info("Loading trained engines and cohort data...")
        self.engine = HybridRecommender()
        self.ratings_df = pd.read_csv(PROCESSED_DIR / "synthetic_ratings.csv")
        self.users_df = pd.read_csv(PROCESSED_DIR / "synthetic_users.csv")
        self.movies_df = pd.read_csv(PROCESSED_DIR / "movies_final.csv")
        self.movies_by_id = self.movies_df.set_index("movieId", drop=False)
        self._poster_cache = self._load_poster_cache()
        self._profiles = self._build_profiles()
        logger.info("RecommenderService ready.")

    @property
    def catalogue_size(self) -> int:
        return len(self.movies_df)

    def _build_profiles(self) -> dict[int, dict]:
        
        counts = self.ratings_df["userId"].value_counts()
        return {
            int(row["userId"]): {
                "userId": int(row["userId"]),
                "age": int(row["age"]),
                "gender": str(row["gender"]),
                "education": str(row["education"]),
                "archetype": str(row["archetype"]),
                "preferred_genres": [
                    x for x in str(row["preferred_genres"]).split("|") if x
                ],
                "preferred_language": [
                    x for x in str(row["preferred_language"]).split("|") if x
                ],
                "base_rating_count": int(counts.get(row["userId"], 0)),
            }
            for _, row in self.users_df.iterrows()
        }

    def get_user_profile(self, user_id: int) -> dict | None:
        profile = self._profiles.get(int(user_id))
        return dict(profile) if profile else None

    def _profile_obj(self, user_id: int) -> UserProfile:
        profile = self._profiles.get(int(user_id))
        if profile is None:
            raise ValueError("Cohort user not found.")
        return UserProfile(
            tuple(profile["preferred_genres"]),
            tuple(profile["preferred_language"]),
            profile["archetype"],
        )

    def title(self, movie_id: int) -> str:
        """Just the title. Building a full movie dict for this was wasteful."""
        row = self._row(movie_id)
        return self._text(row.get("title")) if row is not None else str(movie_id)

    def _row(self, movie_id: int) -> pd.Series | None:
        """One catalogue row, tolerating a duplicated index."""
        if movie_id not in self.movies_by_id.index:
            return None
        selected = self.movies_by_id.loc[movie_id]
        row = selected.iloc[0] if isinstance(selected, pd.DataFrame) else selected
        return row if isinstance(row, pd.Series) else None

    def movie(self, movie_id: int) -> dict | None:
        row = self._row(movie_id)
        if row is None:
            return None
        catalogue_id = self._optional_int(row.get("movieId"))
        if catalogue_id is None:
            return None
        tmdb_id = self._optional_int(row.get("tmdbId"))
        return {
            "movieId": catalogue_id,
            "tmdbId": tmdb_id,
            "title": self._text(row.get("title")),
            "genres": self._text(row.get("clean_genres")),
            "language": self._text(row.get("language")),
            "overview": self._text(row.get("overview")),
            "year": self._optional_int(row.get("release_year")),
            "average_rating": self._optional_float(row.get("vote_average")),
            "vote_count": self._optional_int(row.get("vote_count")),
            "popularity": self._optional_float(row.get("popularity")) or 0.0,
            "poster_url": self._cached_poster(tmdb_id),
        }

    @staticmethod
    def _text(value: object) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    @classmethod
    def _optional_float(cls, value: object) -> float | None:
        text = cls._text(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def _optional_int(cls, value: object) -> int | None:
        number = cls._optional_float(value)
        return None if number is None else int(number)

    def _load_poster_cache(self) -> dict[str, str]:
        if not POSTER_CACHE_PATH.exists():
            return {}
        try:
            with open(POSTER_CACHE_PATH) as fh:
                return {str(k): str(v) for k, v in json.load(fh).items()}
        except OSError, ValueError, TypeError:
            logger.warning("Could not read %s; starting empty.", POSTER_CACHE_PATH)
            return {}

    def _save_poster_cache(self) -> None:
        try:
            POSTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(POSTER_CACHE_PATH, "w") as fh:
                json.dump(self._poster_cache, fh)
        except OSError:
            logger.warning("Could not write %s.", POSTER_CACHE_PATH)

    def _cached_poster(self, tmdb_id: int | None) -> str:
        if tmdb_id is None:
            return ""
        return self._poster_cache.get(str(tmdb_id), "")

    def _fetch_poster(self, tmdb_id: int) -> str:
        api_key = os.getenv("TMDB_API_KEY", "").strip()
        if not api_key:
            return ""
        base_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        if api_key.count(".") == 2:
            request = urllib.request.Request(
                base_url, headers={"Authorization": f"Bearer {api_key}"}
            )
        else:
            request = urllib.request.Request(f"{base_url}?api_key={api_key}")
        try:
            with urllib.request.urlopen(
                request, timeout=POSTER_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError, urllib.error.URLError, ValueError:
            return ""
        poster_path = payload.get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""

    def _tmdb_ids_for(self, movie_ids: list[int]) -> list[int]:
        ids = []
        for movie_id in movie_ids:
            row = self._row(movie_id)
            if row is None:
                continue
            tmdb_id = self._optional_int(row.get("tmdbId"))
            if tmdb_id is not None:
                ids.append(tmdb_id)
        return ids

    def warm_posters(self, movie_ids: list[int]) -> None:
        pending = sorted(
            {
                t
                for t in self._tmdb_ids_for(movie_ids)
                if str(t) not in self._poster_cache
            }
        )
        if not pending or not os.getenv("TMDB_API_KEY", "").strip():
            return
        with ThreadPoolExecutor(max_workers=POSTER_MAX_WORKERS) as pool:
            resolved = list(pool.map(self._fetch_poster, pending))
        for tmdb_id, url in zip(pending, resolved, strict=True):
            self._poster_cache[str(tmdb_id)] = url
        self._save_poster_cache()

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

    def _model_scores(
        self, model: str, user_id: int, liked: list[int], profile: UserProfile
    ) -> np.ndarray:
        if model == "popularity":
            return self.engine.get_popularity_scores()
        if model == "cf":
            return self.engine.get_cf_scores(user_id, liked)
        if model == "cbf":
            return self.engine.get_cbf_scores(liked)
        if model == "affinity":
            return self.engine.get_affinity_only_scores(profile)
        if model == "hybrid":
            return self.engine.get_standard_hybrid_scores(user_id, liked)
        return self.engine.get_localized_hybrid_scores(user_id, liked, profile)

    @staticmethod
    def _reason(model: str, movie: dict, profile: dict, liked_count: int) -> str:
        movie_genres = {g for g in movie["genres"].split("|") if g}
        matches = sorted(movie_genres & set(profile["preferred_genres"]))
        language_match = movie["language"] in profile["preferred_language"]

        if model == "popularity":
            return (
                f"Bayesian weighted rating {movie.get('average_rating') or 0:.1f} "
                f"over {movie.get('vote_count') or 0} votes. No personalisation."
            )
        if model == "cf":
            return (
                f"SVD item factors averaged over {liked_count} liked titles "
                "(the cohort is held out of the trainset)."
            )
        if model == "cbf":
            shared = ", ".join(matches) if matches else "its TF-IDF profile"
            return f"Content similarity to the user's liked titles via {shared}."

        signals = []
        if matches:
            signals.append(f"genre affinity ({', '.join(matches)})")
        if language_match:
            signals.append(f"language affinity ({movie['language']})")
        affinity = "; ".join(signals) or "no cohort signal on this title"

        if model == "affinity":
            return f"Cohort affinity only: {affinity}."
        if model == "hybrid":
            return (
                f"CF {STANDARD_HYBRID_WEIGHTS.cf:.3g} + CBF "
                f"{STANDARD_HYBRID_WEIGHTS.cbf:.3g}; no cohort term."
            )
        return (
            f"CF {LOCALIZED_HYBRID_WEIGHTS.cf:.2f} + CBF "
            f"{LOCALIZED_HYBRID_WEIGHTS.cbf:.2f} + affinity "
            f"{LOCALIZED_HYBRID_WEIGHTS.localization:.2f} — {affinity}."
        )

    def recommend(
        self,
        user_id: int,
        model: str,
        manual_ratings: list[dict] | None = None,
        k: int = 10,
    ) -> list[dict]:
        if model not in MODEL_LABELS:
            raise ValueError(f"Unknown model '{model}'.")
        profile = self.get_user_profile(user_id)
        if profile is None:
            raise ValueError("Cohort user not found.")

        manual_ratings = manual_ratings or []
        liked = self._liked_movie_ids(user_id, manual_ratings)
        scores = self._model_scores(model, user_id, liked, self._profile_obj(user_id))
        exclude = set(liked) | self._manual_ids(manual_ratings)
        ranked = self.engine.recommend_from_scores(
            scores, exclude_movie_ids=list(exclude), k=k
        )
        self.warm_posters([item.movie_id for item in ranked])

        recommendations = []
        for rank, item in enumerate(ranked, start=1):
            movie = self.movie(item.movie_id) or {
                "movieId": item.movie_id,
                "title": item.title,
                "genres": item.genres,
                "language": item.language,
                "poster_url": "",
                "year": None,
                "average_rating": None,
                "vote_count": None,
            }
            movie.update(
                {
                    "rank": rank,
                    "score": round(float(item.score), 4),
                    "reason": self._reason(model, movie, profile, len(liked)),
                }
            )
            recommendations.append(movie)
        return recommendations

    def live_metrics(
        self, user_id: int, recommendations: list[dict]
    ) -> dict[str, float]:
     
        profile = self._profiles.get(int(user_id))
        if profile is None or not recommendations:
            return {}

        top_k = [int(item["movieId"]) for item in recommendations]
        movie_index = {mid: i for i, mid in enumerate(top_k)}
        clean_genres = [str(item["genres"]) for item in recommendations]
        language = [str(item["language"]) for item in recommendations]

        bubble = filter_bubble_score(
            top_k,
            profile["preferred_genres"],
            profile["preferred_language"],
            movie_index,
            clean_genres,
            language,
        )
        return {
            "language_diversity": language_diversity(top_k, movie_index, language),
            "genre_diversity": genre_diversity(top_k, movie_index, clean_genres),
            "filter_bubble_score": round(bubble, 4),
            "preference_escape_at_10": round(1.0 - bubble, 4),
        }


_service: RecommenderService | None = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
    return _service
