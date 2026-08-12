from __future__ import annotations

import logging
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from .cohort_spec import (
    ARCHETYPE_NAMES,
    COHORT_AGE_RANGE,
    COHORT_EDUCATION_WEIGHTS,
    COHORT_GENDER,
    COHORT_LANGUAGES,
    LIKE_THRESHOLD,
    MAX_INFERRED_LANGUAGES,
    MIN_INFERRED_LANGUAGE_SHARE,
    N_COHORT_USERS,
    N_INFERRED_GENRES,
    RANDOM_STATE,
    SYNTHETIC_USER_ID_OFFSET,
    TRAIN_SPLIT_RATIO,
    target_counts,
)
from .mappings import CANONICAL_GENRES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MOVIES_FINAL_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
ARCHETYPE_AFFINITY_PATH = os.path.join(PROCESSED_DIR, "archetype_affinity.pkl")
SYNTHETIC_USERS_PATH = os.path.join(PROCESSED_DIR, "synthetic_users.csv")
SYNTHETIC_RATINGS_PATH = os.path.join(PROCESSED_DIR, "synthetic_ratings.csv")

MIN_RATINGS_PER_USER = 30
MAX_RATINGS_PER_USER = 60
RATING_NOISE_STD = 0.4

AFFINITY_JITTER_SIGMA = 0.55

EXPLORATION_RATIO = 0.15

RECENCY_DECAY = 0.04

REQUIRED_MOVIE_COLUMNS = [
    "movieId",
    "clean_genres",
    "language",
    "vote_count",
    "vote_average",
    "release_year",
]


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Run clean_data.py and build_real_cohort.py first."
        )


def load_movies_final(path: str = MOVIES_FINAL_PATH) -> pd.DataFrame:
    check_file_exists(path)
    movies_df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MOVIE_COLUMNS if c not in movies_df.columns]
    if missing:
        raise ValueError(f"movies_final.csv is missing required columns: {missing}")
    logger.info("Loaded %d movies.", len(movies_df))
    return movies_df


def load_archetype_affinity(movies_df: pd.DataFrame) -> dict[str, np.ndarray]:
    check_file_exists(ARCHETYPE_AFFINITY_PATH)
    payload = joblib.load(ARCHETYPE_AFFINITY_PATH)

    affinity = payload["affinity"]
    stored_ids = np.asarray(payload["movie_ids"])
    current_ids = movies_df["movieId"].to_numpy()

    if len(stored_ids) != len(current_ids) or not np.array_equal(
        stored_ids, current_ids
    ):
        raise ValueError(
            "archetype_affinity.pkl was built against a different catalogue "
            "ordering than movies_final.csv. Re-run build_real_cohort.py."
        )

    missing = set(ARCHETYPE_NAMES) - set(affinity)
    if missing:
        raise ValueError(f"archetype_affinity.pkl is missing archetypes: {missing}")

    logger.info("Loaded affinity vectors for %d archetypes.", len(affinity))
    return affinity


def build_recency_score(movies_df: pd.DataFrame) -> np.ndarray:
    years = (
        pd.to_numeric(movies_df["release_year"], errors="coerce")
        .fillna(2000)
        .to_numpy()
    )
    return np.exp(-RECENCY_DECAY * (years.max() - years)).astype(np.float64)


def assign_archetypes(n_users: int, rng: np.random.Generator) -> list[str]:
    quotas = target_counts(n_users)
    labels: list[str] = []
    for name in ARCHETYPE_NAMES:
        labels.extend([name] * quotas[name])
    labels = labels[:n_users]
    rng.shuffle(labels)
    return labels


def sample_user_history(
    base_affinity: np.ndarray,
    recency: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.flatnonzero(base_affinity > 0)
    if len(support) == 0:
        raise ValueError("Archetype affinity vector has empty support.")

    jitter = np.exp(rng.normal(0.0, AFFINITY_JITTER_SIGMA, size=len(support)))
    strength = base_affinity[support] * jitter * recency[support]

    n_explore = round(k * EXPLORATION_RATIO)
    n_core = max(1, k - n_explore)
    n_core = min(n_core, len(support))

    p = strength / strength.sum()
    core_pick = rng.choice(support, size=n_core, replace=False, p=p)

    picked = [core_pick]
    used = set(core_pick.tolist())

    if n_explore > 0:
        outside = np.setdiff1d(
            np.arange(len(base_affinity), dtype=np.int64),
            np.fromiter(used, dtype=np.int64, count=len(used)),
            assume_unique=False,
        )
        if len(outside) > 0:
            explore_pick = rng.choice(
                outside, size=min(n_explore, len(outside)), replace=False
            )
            picked.append(explore_pick)
            used.update(explore_pick.tolist())

    idx = np.concatenate(picked)

    full_strength = np.zeros(len(base_affinity), dtype=np.float64)
    full_strength[support] = strength
    return idx, full_strength[idx]


def strengths_to_ratings(strengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(strengths)
    if n == 1:
        norm = np.array([0.75])
    else:
        order = np.argsort(np.argsort(strengths))
        norm = order / (n - 1)

    ratings = 1.5 + 3.5 * norm + rng.normal(0.0, RATING_NOISE_STD, size=n)
    ratings = np.clip(ratings, 0.5, 5.0)
    return np.round(ratings * 2) / 2


def infer_preferences_from_history(
    movie_ids: np.ndarray,
    ratings: np.ndarray,
    splits: np.ndarray,
    genres_by_movie: dict[int, str],
    lang_by_movie: dict[int, str],
) -> tuple[str, str]:
    canonical = set(CANONICAL_GENRES)
    mask = (splits == "train") & (ratings >= LIKE_THRESHOLD)

    genre_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()

    for mid in movie_ids[mask]:
        for g in str(genres_by_movie.get(int(mid), "")).split("|"):
            if g in canonical:
                genre_counter[g] += 1
        lang = lang_by_movie.get(int(mid))
        if lang:
            lang_counter[lang] += 1

    top_genres = [g for g, _ in genre_counter.most_common(N_INFERRED_GENRES)]

    total = sum(lang_counter.values()) or 1
    langs = [
        lang
        for lang, c in lang_counter.most_common()
        if c / total >= MIN_INFERRED_LANGUAGE_SHARE and lang in COHORT_LANGUAGES
    ][:MAX_INFERRED_LANGUAGES]

    if not langs:
        common = lang_counter.most_common(1)
        langs = [common[0][0]] if common else ["English"]

    return "|".join(top_genres) if top_genres else "Drama", "|".join(langs)


def generate_cohort(
    movies_df: pd.DataFrame,
    affinity: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie_ids = movies_df["movieId"].to_numpy()
    recency = build_recency_score(movies_df)
    genres_by_movie = dict(zip(movies_df["movieId"], movies_df["clean_genres"]))
    lang_by_movie = dict(zip(movies_df["movieId"], movies_df["language"]))

    education_items = list(COHORT_EDUCATION_WEIGHTS)
    education_probs = list(COHORT_EDUCATION_WEIGHTS.values())

    archetypes = assign_archetypes(N_COHORT_USERS, rng)

    user_rows = []
    rating_rows = []

    for i, arch in enumerate(archetypes):
        uid = SYNTHETIC_USER_ID_OFFSET + i
        k = int(rng.integers(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1))

        idx, strengths = sample_user_history(affinity[arch], recency, k, rng)
        ratings = strengths_to_ratings(strengths, rng)

        n = len(idx)
        n_train = max(1, round(n * TRAIN_SPLIT_RATIO))
        perm = rng.permutation(n)
        splits = np.empty(n, dtype=object)
        splits[perm[:n_train]] = "train"
        splits[perm[n_train:]] = "holdout"

        selected_ids = movie_ids[idx]

        pref_genres, pref_langs = infer_preferences_from_history(
            selected_ids, ratings, splits, genres_by_movie, lang_by_movie
        )

        user_rows.append(
            {
                "userId": uid,
                "age": int(rng.integers(COHORT_AGE_RANGE[0], COHORT_AGE_RANGE[1] + 1)),
                "gender": COHORT_GENDER,
                "education": rng.choice(education_items, p=education_probs),
                "archetype": arch,
                "preferred_genres": pref_genres,
                "preferred_language": pref_langs,
            }
        )

        for mid, r, sp in zip(selected_ids, ratings, splits):
            rating_rows.append(
                {
                    "userId": uid,
                    "movieId": int(mid),
                    "rating": float(r),
                    "split": sp,
                }
            )

    users_df = pd.DataFrame(user_rows)
    ratings_df = pd.DataFrame(rating_rows)

    logger.info(
        "Generated %d users and %d ratings (%d train / %d holdout).",
        len(users_df),
        len(ratings_df),
        int((ratings_df["split"] == "train").sum()),
        int((ratings_df["split"] == "holdout").sum()),
    )
    return users_df, ratings_df


def validate_synthetic_users(users_df: pd.DataFrame) -> None:
    if not users_df["userId"].is_unique:
        raise ValueError("synthetic_users.csv contains duplicate userId values.")
    if users_df["preferred_genres"].eq("").any():
        raise ValueError("synthetic_users.csv contains empty preferred_genres.")
    if users_df["preferred_language"].eq("").any():
        raise ValueError("synthetic_users.csv contains empty preferred_language.")

    all_genres = set(CANONICAL_GENRES)
    bad = set()
    for g_str in users_df["preferred_genres"]:
        bad |= set(str(g_str).split("|")) - all_genres
    if bad:
        raise ValueError(f"Non-canonical genres in synthetic_users: {bad}")

    if not users_df["archetype"].isin(list(ARCHETYPE_NAMES)).all():
        raise ValueError("synthetic_users.csv contains an unrecognised archetype.")

    observed = users_df["archetype"].value_counts().to_dict()
    logger.info("Archetype composition: %s", observed)


def validate_synthetic_ratings(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    if not ratings_df["rating"].between(0.5, 5.0).all():
        raise ValueError("synthetic_ratings.csv has ratings outside [0.5, 5.0].")

    valid = set(movies_df["movieId"].unique())
    if not ratings_df["movieId"].isin(valid).all():
        raise ValueError("synthetic_ratings.csv references unknown movieIds.")

    if not ratings_df["split"].isin(["train", "holdout"]).all():
        raise ValueError("synthetic_ratings.csv has invalid split labels.")

    dup = ratings_df.duplicated(subset=["userId", "movieId"]).sum()
    if dup > 0:
        raise ValueError(f"synthetic_ratings.csv has {dup} duplicate pairs.")

    per_user = ratings_df.groupby("userId")["split"].agg(
        lambda s: (s == "holdout").sum()
    )
    if (per_user == 0).any():
        raise ValueError(f"{int((per_user == 0).sum())} users have no holdout ratings.")


def main() -> None:
    logger.info("Starting generate_synthetic_cohort.py pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    rng = np.random.default_rng(RANDOM_STATE)
    movies_df = load_movies_final()
    affinity = load_archetype_affinity(movies_df)

    users_df, ratings_df = generate_cohort(movies_df, affinity, rng)

    validate_synthetic_users(users_df)
    validate_synthetic_ratings(ratings_df, movies_df)

    users_df.to_csv(SYNTHETIC_USERS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", SYNTHETIC_USERS_PATH, len(users_df))

    ratings_df.to_csv(SYNTHETIC_RATINGS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", SYNTHETIC_RATINGS_PATH, len(ratings_df))

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
