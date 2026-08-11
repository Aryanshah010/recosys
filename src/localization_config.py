"""Cohort-affinity scoring configuration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mappings import CANONICAL_GENRES, CANONICAL_LANGUAGE_MAP

logger = logging.getLogger(__name__)

DERIVED_AFFINITY_PATH = os.path.join("data", "processed", "cohort_affinity_tables.json")

PRIOR_GENRE_AFFINITY: dict[str, float] = {
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

PRIOR_LANGUAGE_AFFINITY: dict[str, float] = {
    "English": 1.00,
    "Hindi": 0.90,
    "Japanese": 0.85,
    "Korean": 0.80,
    "Nepali": 0.75,
}


def _load_derived_tables() -> tuple[dict[str, float], dict[str, float], str]:
    """Load data-derived affinity tables, falling back to the prior."""
    if not os.path.exists(DERIVED_AFFINITY_PATH):
        return dict(PRIOR_GENRE_AFFINITY), dict(PRIOR_LANGUAGE_AFFINITY), "prior"

    try:
        with open(DERIVED_AFFINITY_PATH) as fh:
            payload = json.load(fh)
        genre = {str(k): float(v) for k, v in payload["genre"].items()}
        language = {str(k): float(v) for k, v in payload["language"].items()}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Could not read derived affinity tables from %s (%s); using prior.",
            DERIVED_AFFINITY_PATH,
            exc,
        )
        return dict(PRIOR_GENRE_AFFINITY), dict(PRIOR_LANGUAGE_AFFINITY), "prior"

    for g in CANONICAL_GENRES:
        genre.setdefault(g, PRIOR_GENRE_AFFINITY.get(g, 0.0))
    for lang in PRIOR_LANGUAGE_AFFINITY:
        language.setdefault(lang, PRIOR_LANGUAGE_AFFINITY[lang])

    return genre, language, "derived"


COHORT_GENRE_AFFINITY, COHORT_LANGUAGE_AFFINITY, AFFINITY_SOURCE = (
    _load_derived_tables()
)

GENRE_LOCALIZATION_WEIGHT = COHORT_GENRE_AFFINITY
LANGUAGE_LOCALIZATION_WEIGHT = COHORT_LANGUAGE_AFFINITY

COHORT_GENRE_COMPONENT = 0.6
COHORT_LANGUAGE_COMPONENT = 0.4
LOCALIZATION_GENRE_WEIGHT = COHORT_GENRE_COMPONENT
LOCALIZATION_LANGUAGE_WEIGHT = COHORT_LANGUAGE_COMPONENT


@dataclass(frozen=True, slots=True)
class HybridWeights:
    cf: float
    cbf: float
    localization: float = 0.0

    def sum(self) -> float:
        return self.cf + self.cbf + self.localization


STANDARD_HYBRID_WEIGHTS = HybridWeights(cf=0.625, cbf=0.375)
LOCALIZED_HYBRID_WEIGHTS = HybridWeights(cf=0.50, cbf=0.30, localization=0.20)

AFFINITY_ONLY_WEIGHTS = HybridWeights(cf=0.0, cbf=0.0, localization=1.0)

GENRE_INDEX: dict[str, int] = {g: i for i, g in enumerate(CANONICAL_GENRES)}


def _assert_weight_tables_match_canonical() -> None:
    genre_keys = set(COHORT_GENRE_AFFINITY)
    if not set(CANONICAL_GENRES).issubset(genre_keys):
        raise ValueError(
            f"COHORT_GENRE_AFFINITY is missing canonical genres: "
            f"{set(CANONICAL_GENRES) - genre_keys}"
        )

    canonical_langs = set(CANONICAL_LANGUAGE_MAP.values())
    lang_keys = set(COHORT_LANGUAGE_AFFINITY)
    if not lang_keys.issubset(canonical_langs):
        raise ValueError(
            f"COHORT_LANGUAGE_AFFINITY has keys outside the canonical language "
            f"vocabulary: {lang_keys - canonical_langs}"
        )

    for name, w in [
        ("STANDARD_HYBRID_WEIGHTS", STANDARD_HYBRID_WEIGHTS),
        ("LOCALIZED_HYBRID_WEIGHTS", LOCALIZED_HYBRID_WEIGHTS),
        ("AFFINITY_ONLY_WEIGHTS", AFFINITY_ONLY_WEIGHTS),
    ]:
        if abs(w.sum() - 1.0) > 1e-6:
            raise ValueError(f"{name} must sum to 1.0, got {w.sum()}.")

    if abs(COHORT_GENRE_COMPONENT + COHORT_LANGUAGE_COMPONENT - 1.0) > 1e-6:
        raise ValueError("Cohort genre/language components must sum to 1.0.")


_assert_weight_tables_match_canonical()


def build_genre_onehot_from_list(clean_genres: list[str]) -> np.ndarray[Any, Any]:
    onehot = np.zeros((len(clean_genres), len(CANONICAL_GENRES)), dtype=np.float32)
    for row_idx, genre_str in enumerate(clean_genres):
        for g in str(genre_str).split("|"):
            col = GENRE_INDEX.get(g)
            if col is not None:
                onehot[row_idx, col] = 1.0
    return onehot


def build_genre_weight_vector(preferred_genres: list[str]) -> np.ndarray[Any, Any]:
    pref = set(preferred_genres)
    return np.array(
        [
            COHORT_GENRE_AFFINITY.get(g, 0.0) if g in pref else 0.0
            for g in CANONICAL_GENRES
        ],
        dtype=np.float32,
    )


def compute_genre_affinity_scores(
    genre_onehot: np.ndarray[Any, Any], genre_vec: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Mean affinity across a title's matching preferred genres."""
    matched = genre_onehot * genre_vec[None, :]
    match_counts = (genre_onehot > 0).sum(axis=1)
    n_matched = (matched > 0).sum(axis=1)
    totals = matched.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(n_matched > 0, totals / np.maximum(n_matched, 1), 0.0)

    del match_counts
    return scores.astype(np.float32)


def compute_language_preference_scores(
    movie_language: np.ndarray[Any, Any], preferred_langs: set[str]
) -> np.ndarray[Any, Any]:
    return np.array(
        [
            COHORT_LANGUAGE_AFFINITY.get(lang, 0.0) if lang in preferred_langs else 0.0
            for lang in movie_language
        ],
        dtype=np.float32,
    )
