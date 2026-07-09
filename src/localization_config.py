from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mappings import CANONICAL_GENRES, CANONICAL_LANGUAGE_MAP

GENRE_LOCALIZATION_WEIGHT: dict[str, float] = {
    "Sci-Fi": 1.00,
    "Action": 0.95,
    "Thriller": 0.90,
    "Adventure": 0.85,
    "Fantasy": 0.80,
    "Crime": 0.75,
    "Animation": 0.70,
    "Comedy": 0.65,
    "Drama": 0.60,
    "Mystery": 0.55,
    "Romance": 0.40,
    "Horror": 0.40,
    "Family": 0.35,
    "History": 0.30,
    "War": 0.30,
    "Documentary": 0.20,
    "Music": 0.20,
    "Western": 0.15,
    "TV": 0.15,
}

LANGUAGE_LOCALIZATION_WEIGHT: dict[str, float] = {
    "English": 1.00,
    "Hindi": 0.90,
    "Japanese": 0.85,
    "Korean": 0.80,
    "Nepali": 0.75,
}

LOCALIZATION_GENRE_WEIGHT = 0.6
LOCALIZATION_LANGUAGE_WEIGHT = 0.4


@dataclass(frozen=True, slots=True)
class HybridWeights:
    cf: float
    cbf: float
    localization: float = 0.0

    def sum(self) -> float:
        return self.cf + self.cbf + self.localization


STANDARD_HYBRID_WEIGHTS = HybridWeights(cf=0.60, cbf=0.40)
LOCALIZED_HYBRID_WEIGHTS = HybridWeights(cf=0.45, cbf=0.35, localization=0.20)

GENRE_INDEX: dict[str, int] = {g: i for i, g in enumerate(CANONICAL_GENRES)}


def _assert_weight_tables_match_canonical() -> None:
    genre_keys = set(GENRE_LOCALIZATION_WEIGHT)
    if genre_keys != set(CANONICAL_GENRES):
        raise ValueError(
            f"GENRE_LOCALIZATION_WEIGHT keys do not match CANONICAL_GENRES. "
            f"Missing: {set(CANONICAL_GENRES) - genre_keys}, "
            f"Extra: {genre_keys - set(CANONICAL_GENRES)}"
        )
    canonical_langs = set(CANONICAL_LANGUAGE_MAP.values())
    lang_keys = set(LANGUAGE_LOCALIZATION_WEIGHT)
    if not lang_keys.issubset(canonical_langs):
        raise ValueError(
            f"LANGUAGE_LOCALIZATION_WEIGHT has keys not in canonical "
            f"language vocabulary: {lang_keys - canonical_langs}"
        )
    if abs(STANDARD_HYBRID_WEIGHTS.sum() - 1.0) > 1e-6:
        raise ValueError("STANDARD_HYBRID_WEIGHTS must sum to 1.0.")
    if abs(LOCALIZED_HYBRID_WEIGHTS.sum() - 1.0) > 1e-6:
        raise ValueError("LOCALIZED_HYBRID_WEIGHTS must sum to 1.0.")


_assert_weight_tables_match_canonical()


def build_genre_onehot_from_list(clean_genres: list[str]) -> np.ndarray[Any, Any]:
    onehot = np.zeros((len(clean_genres), len(CANONICAL_GENRES)), dtype=np.float32)
    for row_idx, genre_str in enumerate(clean_genres):
        for g in genre_str.split("|"):
            col = GENRE_INDEX.get(g)
            if col is not None:
                onehot[row_idx, col] = 1.0
    return onehot


def build_genre_weight_vector(preferred_genres: list[str]) -> np.ndarray[Any, Any]:
    return np.array(
        [
            GENRE_LOCALIZATION_WEIGHT[g] if g in preferred_genres else 0.0
            for g in CANONICAL_GENRES
        ],
        dtype=np.float32,
    )


def compute_language_preference_scores(
    movie_language: np.ndarray[Any, Any], preferred_langs: set[str]
) -> np.ndarray[Any, Any]:
    return np.array(
        [
            LANGUAGE_LOCALIZATION_WEIGHT.get(lang, 0.0)
            if lang in preferred_langs
            else 0.0
            for lang in movie_language
        ],
        dtype=np.float32,
    )
