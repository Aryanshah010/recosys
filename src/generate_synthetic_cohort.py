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

GENRE_SCORE_COEF = 0.5
LANGUAGE_SCORE_COEF = 0.3
POPULARITY_SCORE_COEF = 0.1
GUMBEL_TEMPERATURE = 0.3
GENRE_WEIGHT_JITTER = 0.15

REQUIRED_MOVIE_COLUMNS = [
    "movieId",
    "clean_genres",
    "language",
    "vote_count",
    "vote_average",
    "release_year",
]

ARCHETYPE_BY_NAME: dict[str, "Archetype"] = {}


class Archetype(TypedDict):
    name: str
    prob: float
    primary_language: str | None
    secondary_languages: list[str]
    secondary_language_prob: float
    preferred_pool_genres: list[str]
    preferred_pool_ratio: float
    preferred_pool_ratio_jitter: tuple[float, float]
    require_animation: bool
    core_genres: list[str]
    optional_genres: list[str]
    n_optional_genres: tuple[int, int]


ARCHETYPES: list[Archetype] = [
    {
        "name": "hollywood",
        "prob": 0.35,
        "primary_language": "English",
        "secondary_languages": ["Hindi"],
        "secondary_language_prob": 0.25,
        "preferred_pool_genres": [
            "Action",
            "Sci-Fi",
            "Adventure",
            "Thriller",
            "Comedy",
            "Drama",
        ],
        "preferred_pool_ratio": 0.90,
        "preferred_pool_ratio_jitter": (0.85, 0.95),
        "require_animation": False,
        "core_genres": ["Action", "Sci-Fi"],
        "optional_genres": ["Adventure", "Crime", "Thriller", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "anime",
        "prob": 0.20,
        "primary_language": "Japanese",
        "secondary_languages": ["English", "Korean"],
        "secondary_language_prob": 0.20,
        "preferred_pool_genres": [
            "Animation",
            "Fantasy",
            "Adventure",
            "Action",
            "Sci-Fi",
        ],
        "preferred_pool_ratio": 0.80,
        "preferred_pool_ratio_jitter": (0.70, 0.90),
        "require_animation": True,
        "core_genres": ["Animation", "Fantasy"],
        "optional_genres": ["Action", "Sci-Fi", "Adventure", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "bollywood",
        "prob": 0.20,
        "primary_language": "Hindi",
        "secondary_languages": ["English", "Nepali"],
        "secondary_language_prob": 0.30,
        "preferred_pool_genres": [
            "Drama",
            "Romance",
            "Comedy",
            "Action",
            "Family",
        ],
        "preferred_pool_ratio": 0.80,
        "preferred_pool_ratio_jitter": (0.70, 0.90),
        "require_animation": False,
        "core_genres": ["Drama", "Comedy"],
        "optional_genres": ["Action", "Romance", "Thriller"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "kdrama",
        "prob": 0.15,
        "primary_language": "Korean",
        "secondary_languages": ["English", "Japanese"],
        "secondary_language_prob": 0.15,
        "preferred_pool_genres": ["Drama", "Romance", "Mystery", "Thriller"],
        "preferred_pool_ratio": 0.80,
        "preferred_pool_ratio_jitter": (0.70, 0.90),
        "require_animation": False,
        "core_genres": ["Drama", "Romance"],
        "optional_genres": ["Mystery", "Thriller", "Comedy"],
        "n_optional_genres": (1, 2),
    },
    {
        "name": "mixed",
        "prob": 0.10,
        "primary_language": None,
        "secondary_languages": ["English", "Hindi", "Japanese", "Korean", "Nepali"],
        "secondary_language_prob": 0.50,
        "preferred_pool_genres": [
            "Action",
            "Comedy",
            "Drama",
            "Adventure",
            "Thriller",
        ],
        "preferred_pool_ratio": 0.40,
        "preferred_pool_ratio_jitter": (0.35, 0.45),
        "require_animation": False,
        "core_genres": ["Action", "Comedy"],
        "optional_genres": ["Drama", "Adventure", "Thriller"],
        "n_optional_genres": (1, 2),
    },
]

ARCHETYPE_BY_NAME = {a["name"]: a for a in ARCHETYPES}


def _assert_archetypes_valid() -> None:
    genre_keys = set(GENRE_LOCALIZATION_WEIGHT)
    lang_keys = set(LANGUAGE_LOCALIZATION_WEIGHT)

    prob_sum = sum(a["prob"] for a in ARCHETYPES)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"ARCHETYPES probabilities must sum to 1.0, got {prob_sum}")

    for a in ARCHETYPES:
        used_genres = (
            set(a["core_genres"])
            | set(a["optional_genres"])
            | set(a["preferred_pool_genres"])
        )
        used_langs = set(a["secondary_languages"])
        if a["primary_language"]:
            used_langs.add(a["primary_language"])
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


def _movie_has_genre(genre_str: str, genres: set[str]) -> bool:
    return bool(set(genre_str.split("|")) & genres)


def build_archetype_pools(
    movies_df: pd.DataFrame,
    archetype: Archetype,
) -> tuple[np.ndarray, np.ndarray]:
    languages = movies_df["language"].fillna("").to_numpy()
    genre_strs = movies_df["clean_genres"].fillna("").tolist()
    n = len(movies_df)
    all_idx = np.arange(n, dtype=np.int64)
    preferred_genres = set(archetype["preferred_pool_genres"])

    secondary_langs = set(archetype["secondary_languages"])
    secondary_mask = np.array(
        [lang in secondary_langs for lang in languages],
        dtype=bool,
    )

    if archetype["primary_language"] is None:
        preferred_mask = np.array(
            [_movie_has_genre(g, preferred_genres) for g in genre_strs],
            dtype=bool,
        )
    else:
        primary = archetype["primary_language"]
        preferred_mask = np.array(
            [
                lang == primary
                and _movie_has_genre(g, preferred_genres)
                and (
                    not archetype["require_animation"]
                    or "Animation" in str(g).split("|")
                )
                for lang, g in zip(languages, genre_strs)
            ],
            dtype=bool,
        )

    preferred_idx = all_idx[preferred_mask]
    global_idx = all_idx[~preferred_mask & secondary_mask]
    return preferred_idx, global_idx


def build_user_languages(archetype: Archetype, rng: np.random.Generator) -> list[str]:
    if archetype["primary_language"]:
        langs = [archetype["primary_language"]]
        for lang in archetype["secondary_languages"]:
            if rng.random() < archetype["secondary_language_prob"]:
                langs.append(lang)
        return langs

    pool = archetype["secondary_languages"]
    # For mixed archetype, pick 1 or 2 random preferred languages so their
    # 'preference bubble' isn't huge, making their diverse watching habits
    # accurately reflect low language alignment.
    n = int(rng.integers(1, 3))
    return rng.choice(pool, size=n, replace=False).tolist()


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


def build_jittered_genre_vector(
    pref_genres: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    vec = build_genre_weight_vector(pref_genres).copy()
    for i in range(len(vec)):
        if vec[i] > 0:
            vec[i] *= rng.uniform(1.0 - GENRE_WEIGHT_JITTER, 1.0 + GENRE_WEIGHT_JITTER)
    return vec


def sample_candidate_indices(
    preferred_idx: np.ndarray,
    global_idx: np.ndarray,
    k: int,
    preferred_ratio: float,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    n_preferred = round(k * preferred_ratio)
    n_global = k - n_preferred

    n_preferred = min(n_preferred, len(preferred_idx))
    n_global = min(n_global, len(global_idx))

    selected: list[np.ndarray] = []
    used: set[int] = set()

    def _sample(pool_idx: np.ndarray, n: int) -> np.ndarray:
        if weights is not None:
            p = weights[pool_idx]
            if p.sum() > 0:
                p = p / p.sum()
            else:
                p = None
        else:
            p = None
        return rng.choice(pool_idx, size=n, replace=False, p=p)

    if n_preferred > 0 and len(preferred_idx) > 0:
        pref_pick = _sample(preferred_idx, n_preferred)
        selected.append(pref_pick)
        used.update(pref_pick.tolist())

    if n_global > 0 and len(global_idx) > 0:
        available_global = global_idx[~np.isin(global_idx, list(used))]
        n_global = min(n_global, len(available_global))
        if n_global > 0:
            glob_pick = _sample(available_global, n_global)
            selected.append(glob_pick)
            used.update(glob_pick.tolist())

    if not selected:
        fallback = np.concatenate([preferred_idx, global_idx])
        if len(fallback) == 0:
            raise ValueError("No candidate movies available for sampling.")
        pick_size = min(k, len(fallback))
        return _sample(fallback, pick_size)

    candidate_idx = np.concatenate(selected)
    shortfall = k - len(candidate_idx)
    if shortfall > 0:
        remaining = np.concatenate([preferred_idx, global_idx])
        remaining = remaining[~np.isin(remaining, list(used))]
        if len(remaining) > 0:
            extra = _sample(remaining, min(shortfall, len(remaining)))
            candidate_idx = np.concatenate([candidate_idx, extra])

    return candidate_idx


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


def build_bayesian_popularity(movies_df: pd.DataFrame) -> np.ndarray:
    v = (
        pd.to_numeric(movies_df["vote_count"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    R = (
        pd.to_numeric(movies_df["vote_average"], errors="coerce")
        .fillna(5.0)
        .to_numpy(dtype=np.float64)
    )
    C = R.mean()
    m = np.percentile(v[v > 0], 50) if np.any(v > 0) else 1.0
    bayes = (v / (v + m)) * R + (m / (v + m)) * C
    return bayes.astype(np.float32)


def build_recency_score(movies_df: pd.DataFrame) -> np.ndarray:
    years = (
        pd.to_numeric(movies_df["release_year"], errors="coerce")
        .fillna(2000)
        .to_numpy()
    )
    max_year = years.max()
    decay = np.exp(-0.05 * (max_year - years))
    return decay.astype(np.float32)


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
    bayesian_popularity = build_bayesian_popularity(movies_df)
    recency_score = build_recency_score(movies_df)

    pools_cache = {a["name"]: build_archetype_pools(movies_df, a) for a in ARCHETYPES}
    for name, (pref, glob) in pools_cache.items():
        logger.info(
            "Archetype '%s': preferred pool=%d, global pool=%d.",
            name,
            len(pref),
            len(glob),
        )

    all_rows = []
    for _, user in users_df.iterrows():
        archetype = ARCHETYPE_BY_NAME[str(user["archetype"])]
        preferred_idx, global_idx = pools_cache[archetype["name"]]

        pref_genres = str(user["preferred_genres"]).split("|")
        pref_langs = set(str(user["preferred_language"]).split("|"))

        lo, hi = archetype["preferred_pool_ratio_jitter"]
        preferred_ratio = rng.uniform(lo, hi)

        genre_vec = build_jittered_genre_vector(pref_genres, rng)
        genre_score = genre_onehot @ genre_vec

        # sample proportionally to genre score * Bayesian popularity * recency
        sampling_weights = (
            np.clip(genre_score, 0.01, None) * bayesian_popularity * recency_score
        )

        k = int(rng.integers(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1))
        candidate_idx = sample_candidate_indices(
            preferred_idx, global_idx, k, preferred_ratio, rng, weights=sampling_weights
        )
        k = len(candidate_idx)

        lang_score = compute_language_preference_scores(movie_language, pref_langs)

        # Mixed archetype represents users without strong cultural/language boundaries.
        # Zeroing out the language coefficient ensures their ratings are driven solely
        # by genres, making their ground-truth holdout set culturally diverse.
        user_lang_coef = LANGUAGE_SCORE_COEF if archetype["name"] != "mixed" else 0.0

        base_score = (
            GENRE_SCORE_COEF * genre_score
            + user_lang_coef * lang_score
            + POPULARITY_SCORE_COEF * popularity_norm
        )

        selected_scores = base_score[candidate_idx]
        gumbel_noise = -np.log(-np.log(rng.random(len(candidate_idx)) + 1e-12) + 1e-12)
        keys = selected_scores + GUMBEL_TEMPERATURE * gumbel_noise
        order = np.argsort(-keys)
        candidate_idx = candidate_idx[order]
        selected_scores = selected_scores[order]

        s_min, s_max = selected_scores.min(), selected_scores.max()
        norm = (selected_scores - s_min) / (s_max - s_min + 1e-8)

        ratings = 1.5 + 3.5 * norm + rng.normal(0, RATING_NOISE_STD, size=k)
        ratings = np.clip(ratings, 0.5, 5.0)
        ratings = np.round(ratings * 2) / 2

        n_train = round(k * TRAIN_SPLIT_RATIO)
        shuffle_order = rng.permutation(k)
        split_labels = np.where(np.argsort(shuffle_order) < n_train, "train", "holdout")

        selected_movie_ids = movie_ids[candidate_idx]
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
