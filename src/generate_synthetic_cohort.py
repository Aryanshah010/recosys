from __future__ import annotations

import logging
import os
from typing import TypedDict

import numpy as np
import pandas as pd

from .mappings import CANONICAL_GENRES
from .localization_config import (
    GENRE_LOCALIZATION_WEIGHT,
    LANGUAGE_LOCALIZATION_WEIGHT,
    build_genre_weight_vector,
    build_genre_onehot_from_list,
    compute_language_preference_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MOVIES_FINAL_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
SYNTHETIC_USERS_PATH = os.path.join(PROCESSED_DIR, "synthetic_users.csv")
SYNTHETIC_RATINGS_PATH = os.path.join(PROCESSED_DIR, "synthetic_ratings.csv")

N_USERS = 400
USER_ID_OFFSET = 900_000
AGE_RANGE = (19, 26)
GENDER = "Male"

EDUCATION_WEIGHTS: dict[str, float] = {
    "BSc Computing": 0.60,
    "BIT": 0.20,
    "BCA": 0.15,
    "BE Software": 0.05,
}

MIN_RATINGS_PER_USER = 30
MAX_RATINGS_PER_USER = 60
TRAIN_SPLIT_RATIO = 0.8
RATING_NOISE_STD = 0.5
RANDOM_STATE = 42


class Archetype(TypedDict):
    name: str
    prob: float
    core_languages: list[str]
    optional_languages: list[str]
    optional_language_prob: float
    core_genres: list[str]
    optional_genres: list[str]
    n_optional_genres: tuple[int, int]


ARCHETYPES: list[Archetype] = [
    {
        "name": "hollywood",
        "prob": 0.35,
        "core_languages": ["English"],
        "optional_languages": ["Hindi"],
        "optional_language_prob": 0.25,
        "core_genres": ["Action", "Sci-Fi"],
        "optional_genres": ["Adventure", "Crime", "Thriller", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "anime",
        "prob": 0.20,
        "core_languages": ["English", "Japanese"],
        "optional_languages": ["Korean"],
        "optional_language_prob": 0.20,
        "core_genres": ["Animation", "Fantasy"],
        "optional_genres": ["Action", "Sci-Fi", "Adventure", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "bollywood",
        "prob": 0.20,
        "core_languages": ["English", "Hindi"],
        "optional_languages": ["Nepali"],
        "optional_language_prob": 0.30,
        "core_genres": ["Drama", "Comedy"],
        "optional_genres": ["Action", "Romance", "Thriller"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "kdrama",
        "prob": 0.15,
        "core_languages": ["English", "Korean"],
        "optional_languages": ["Japanese"],
        "optional_language_prob": 0.15,
        "core_genres": ["Drama", "Romance"],
        "optional_genres": ["Mystery", "Thriller", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "mixed",
        "prob": 0.10,
        "core_languages": ["English", "Hindi"],
        "optional_languages": ["Nepali", "Japanese"],
        "optional_language_prob": 0.20,
        "core_genres": ["Action", "Comedy"],
        "optional_genres": ["Drama", "Adventure", "Thriller"],
        "n_optional_genres": (1, 2),
    },
]

GENRE_SCORE_COEF = 0.5
LANGUAGE_SCORE_COEF = 0.3
POPULARITY_SCORE_COEF = 0.1
GUMBEL_TEMPERATURE = 0.3

REQUIRED_MOVIE_COLUMNS = ["movieId", "clean_genres", "language", "vote_count"]


def _assert_archetypes_valid() -> None:
    genre_keys = set(GENRE_LOCALIZATION_WEIGHT)
    lang_keys = set(LANGUAGE_LOCALIZATION_WEIGHT)

    prob_sum = sum(a["prob"] for a in ARCHETYPES)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"ARCHETYPES probabilities must sum to 1.0, got {prob_sum}")

    for a in ARCHETYPES:
        used_genres = set(a["core_genres"]) | set(a["optional_genres"])
        used_langs = set(a["core_languages"]) | set(a["optional_languages"])
        bad_g = used_genres - genre_keys
        bad_l = used_langs - lang_keys
        if bad_g or bad_l:
            raise ValueError(
                f"Archetype '{a['name']}' uses undefined genres {bad_g} "
                f"or languages {bad_l}."
            )


_assert_archetypes_valid()


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Run clean_data.py first to produce movies_final.csv."
        )


def load_movies_final(path: str = MOVIES_FINAL_PATH) -> pd.DataFrame:
    check_file_exists(path)
    movies_df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MOVIE_COLUMNS if c not in movies_df.columns]
    if missing:
        raise ValueError(f"movies_final.csv is missing required columns: {missing}")
    logger.info("Loaded %d movies.", len(movies_df))
    return movies_df


def build_user_languages(archetype: Archetype, rng: np.random.Generator) -> list[str]:
    langs = list(archetype["core_languages"])
    for lang in archetype["optional_languages"]:
        if rng.random() < archetype["optional_language_prob"]:
            langs.append(lang)
    return langs


def build_user_genres(archetype: Archetype, rng: np.random.Generator) -> list[str]:
    genres = list(archetype["core_genres"])
    lo, hi = archetype["n_optional_genres"]
    n_optional = int(rng.integers(lo, hi + 1))
    pool = archetype["optional_genres"]
    n_optional = min(n_optional, len(pool))
    if n_optional > 0:
        picked = rng.choice(pool, size=n_optional, replace=False)
        genres.extend(picked.tolist())
    return genres


def generate_user_profiles(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    education_items = list(EDUCATION_WEIGHTS.keys())
    education_probs = list(EDUCATION_WEIGHTS.values())
    archetype_probs = [a["prob"] for a in ARCHETYPES]

    rows = []
    for i in range(n_users):
        archetype = ARCHETYPES[rng.choice(len(ARCHETYPES), p=archetype_probs)]
        genres = build_user_genres(archetype, rng)
        langs = build_user_languages(archetype, rng)
        rows.append(
            {
                "userId": USER_ID_OFFSET + i,
                "age": int(rng.integers(AGE_RANGE[0], AGE_RANGE[1])),
                "gender": GENDER,
                "education": rng.choice(education_items, p=education_probs),
                "archetype": archetype["name"],
                "preferred_genres": "|".join(genres),
                "preferred_language": "|".join(langs),
            }
        )
    logger.info("Generated %d synthetic user profiles.", len(rows))
    return pd.DataFrame(rows)


def build_popularity_norm(movies_df: pd.DataFrame) -> np.ndarray:
    vote_count = pd.to_numeric(movies_df["vote_count"], errors="coerce").fillna(0.0)
    log_vc = np.log1p(vote_count.to_numpy(dtype=np.float64))
    max_log_vc = log_vc.max() if log_vc.max() > 0 else 1.0
    return (log_vc / max_log_vc).astype(np.float32)


def generate_ratings(
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:

    movie_ids = movies_df["movieId"].to_numpy()
    movie_language = movies_df["language"].fillna("").to_numpy()
    genre_onehot = build_genre_onehot_from_list(
        movies_df["clean_genres"].fillna("").tolist()
    )
    popularity_norm = build_popularity_norm(movies_df)
    n_movies = len(movies_df)

    all_rows = []
    for _, user in users_df.iterrows():
        pref_genres_str = str(user["preferred_genres"])
        pref_langs_str = str(user["preferred_language"])
        pref_genres = pref_genres_str.split("|") if pref_genres_str else []
        pref_langs = set(pref_langs_str.split("|")) if pref_langs_str else set()

        genre_vec = build_genre_weight_vector(pref_genres)
        lang_score = compute_language_preference_scores(movie_language, pref_langs)

        base_score = (
            GENRE_SCORE_COEF * (genre_onehot @ genre_vec)
            + LANGUAGE_SCORE_COEF * lang_score
            + POPULARITY_SCORE_COEF * popularity_norm
        )

        gumbel_noise = -np.log(-np.log(rng.random(n_movies) + 1e-12) + 1e-12)
        keys = base_score + GUMBEL_TEMPERATURE * gumbel_noise

        k = int(rng.integers(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1))
        k = min(k, n_movies)
        top_idx = np.argpartition(-keys, k - 1)[:k]

        selected_scores = base_score[top_idx]
        s_min, s_max = selected_scores.min(), selected_scores.max()
        norm = (selected_scores - s_min) / (s_max - s_min + 1e-8)

        ratings = 1.5 + 3.5 * norm + rng.normal(0, RATING_NOISE_STD, size=k)
        ratings = np.clip(ratings, 0.5, 5.0)
        ratings = np.round(ratings * 2) / 2

        n_train = int(round(k * TRAIN_SPLIT_RATIO))
        shuffle_order = rng.permutation(k)
        split_labels = np.where(np.argsort(shuffle_order) < n_train, "train", "holdout")

        selected_movie_ids = movie_ids[top_idx]
        for mid, r, split in zip(selected_movie_ids, ratings, split_labels):
            all_rows.append(
                {
                    "userId": user["userId"],
                    "movieId": int(mid),
                    "rating": float(r),
                    "split": split,
                }
            )

    ratings_df = pd.DataFrame(all_rows)
    logger.info(
        "Generated %d synthetic ratings for %d users.", len(ratings_df), len(users_df)
    )
    return ratings_df


def validate_synthetic_users(users_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    if not users_df["userId"].is_unique:
        raise ValueError("synthetic_users.csv contains duplicate userId values.")
    if bool(users_df["preferred_genres"].eq("").any()):
        raise ValueError("synthetic_users.csv contains empty preferred_genres.")
    if bool(users_df["preferred_language"].eq("").any()):
        raise ValueError("synthetic_users.csv contains empty preferred_language.")

    all_genres = set(CANONICAL_GENRES)
    bad_genres = set()
    for g_str in users_df["preferred_genres"]:
        bad_genres |= set(g_str.split("|")) - all_genres
    if bad_genres:
        raise ValueError(f"Non-canonical genres found in synthetic_users: {bad_genres}")

    allowed_langs = set(LANGUAGE_LOCALIZATION_WEIGHT)
    bad_langs = set()
    for l_str in users_df["preferred_language"]:
        bad_langs |= set(l_str.split("|")) - allowed_langs
    if bad_langs:
        raise ValueError(
            f"Languages outside cohort vocabulary found in synthetic_users: {bad_langs}"
        )

    valid_archetypes = {a["name"] for a in ARCHETYPES}
    if not bool(users_df["archetype"].isin(list(valid_archetypes)).all()):
        raise ValueError("synthetic_users.csv contains an unrecognized archetype.")


def validate_synthetic_ratings(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    if not bool(ratings_df["rating"].between(0.5, 5.0).all()):
        raise ValueError(
            "synthetic_ratings.csv contains rating values outside [0.5, 5.0]."
        )

    valid_movie_ids = set(movies_df["movieId"].unique())
    orphans = ~ratings_df["movieId"].isin(list(valid_movie_ids))
    if bool(orphans.any()):
        raise ValueError(
            f"synthetic_ratings.csv references {orphans.sum()} movieId(s) "
            "not present in movies_final.csv."
        )

    if not bool(ratings_df["split"].isin(["train", "holdout"]).all()):
        raise ValueError("synthetic_ratings.csv contains invalid split labels.")

    dup = ratings_df.duplicated(subset=["userId", "movieId"]).sum()
    if dup > 0:
        raise ValueError(
            f"synthetic_ratings.csv contains {dup} duplicate (userId, movieId) pairs."
        )


def main() -> None:
    logger.info("Starting generate_synthetic_cohort.py pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    rng = np.random.default_rng(RANDOM_STATE)
    movies_df = load_movies_final()

    users_df = generate_user_profiles(N_USERS, rng)
    validate_synthetic_users(users_df, movies_df)

    ratings_df = generate_ratings(users_df, movies_df, rng)
    validate_synthetic_ratings(ratings_df, movies_df)

    users_df.to_csv(SYNTHETIC_USERS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", SYNTHETIC_USERS_PATH, len(users_df))

    ratings_df.to_csv(SYNTHETIC_RATINGS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", SYNTHETIC_RATINGS_PATH, len(ratings_df))

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
