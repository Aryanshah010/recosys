"""Select a real MovieLens proxy cohort matching the target consumption profile."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from .cohort_spec import (
    ARCHETYPE_NAMES,
    COHORT_DESIGN_NOTE,
    COHORT_LANGUAGES,
    LIKE_THRESHOLD,
    MAX_INFERRED_LANGUAGES,
    MIN_INFERRED_LANGUAGE_SHARE,
    MIN_LIKED_RATINGS,
    N_COHORT_USERS,
    N_INFERRED_GENRES,
    RANDOM_STATE,
    TRAIN_SPLIT_RATIO,
    label_from_language_shares,
    target_counts,
)
from .mappings import CANONICAL_GENRES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MOVIES_FINAL_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
RATINGS_FINAL_PATH = os.path.join(PROCESSED_DIR, "ratings_final.csv")

REAL_USERS_PATH = os.path.join(PROCESSED_DIR, "real_cohort_users.csv")
REAL_RATINGS_PATH = os.path.join(PROCESSED_DIR, "real_cohort_ratings.csv")
HELDOUT_USER_IDS_PATH = os.path.join(PROCESSED_DIR, "cf_excluded_user_ids.json")
ARCHETYPE_AFFINITY_PATH = os.path.join(PROCESSED_DIR, "archetype_affinity.pkl")
BENCHMARK_PROFILE_PATH = os.path.join(RESULTS_DIR, "benchmark_population_profile.csv")

RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

MAX_SAMPLING_FRACTION = 0.90

MIN_GROUP_SUPPORT = 3


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. Run clean_data.py first."
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    check_file_exists(MOVIES_FINAL_PATH)
    check_file_exists(RATINGS_FINAL_PATH)

    movies_df = pd.read_csv(
        MOVIES_FINAL_PATH, usecols=["movieId", "clean_genres", "language"]
    )

    logger.info("Loading ratings (28M rows, this takes a moment)...")
    ratings_df = pd.read_csv(RATINGS_FINAL_PATH, dtype=RATINGS_DTYPES)  # pyright: ignore[reportArgumentType]

    if "timestamp" not in ratings_df.columns:
        raise ValueError(
            "ratings_final.csv has no timestamp column. Re-run clean_data.py: "
            "temporal splitting requires it."
        )

    logger.info(
        "Loaded %d movies and %d ratings from %d users.",
        len(movies_df),
        len(ratings_df),
        ratings_df["userId"].nunique(),
    )
    return movies_df, ratings_df


def compute_language_shares(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-user share of liked ratings by movie language."""
    lang_by_movie = dict(zip(movies_df["movieId"], movies_df["language"]))

    liked = ratings_df.loc[
        ratings_df["rating"] >= LIKE_THRESHOLD, ["userId", "movieId"]
    ]
    liked = liked.assign(language=liked["movieId"].map(lang_by_movie))

    counts = pd.crosstab(liked["userId"], liked["language"])
    totals = counts.sum(axis=1)

    eligible = totals >= MIN_LIKED_RATINGS
    counts, totals = counts[eligible], totals[eligible]

    shares = counts.div(totals, axis=0)
    logger.info("Users with >= %d liked ratings: %d", MIN_LIKED_RATINGS, len(shares))
    return shares, totals


def label_users(shares: pd.DataFrame) -> pd.Series:
    share_dicts = shares.to_dict(orient="index")
    labels = {
        uid: label_from_language_shares({str(k): float(v) for k, v in row.items()})
        for uid, row in share_dicts.items()
    }
    return pd.Series(labels, name="archetype")


def report_benchmark_population(
    labels: pd.Series, shares: pd.DataFrame
) -> pd.DataFrame:
    """Quantify how unlike the target cohort the benchmark population is."""
    counts = labels.value_counts()
    pct = (counts / len(labels) * 100).round(3)

    rows = []
    for name in ARCHETYPE_NAMES:
        rows.append(
            {
                "Archetype": name,
                "Benchmark_Users": int(counts.get(name, 0)),
                "Benchmark_Share_%": float(pct.get(name, 0.0)),
            }
        )
    profile = pd.DataFrame(rows)

    english_share = shares["English"] if "English" in shares.columns else pd.Series(0.0, index=shares.index)
    english_dominant = float((english_share >= 0.85).mean() * 100)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    profile.to_csv(BENCHMARK_PROFILE_PATH, index=False)

    logger.info("Benchmark population profile (MovieLens 32M):")
    for _, r in profile.iterrows():
        logger.info(
            "  %-10s %7d users (%.3f%%)",
            r["Archetype"],
            r["Benchmark_Users"],
            r["Benchmark_Share_%"],
        )
    logger.info(
        "  English-dominant users: %.2f%% of eligible population.", english_dominant
    )
    return profile


def sample_cohort(
    labels: pd.Series, rng: np.random.Generator
) -> tuple[pd.Series, pd.DataFrame]:
    """Stratified quota sample matching the target cohort's archetype mix."""
    quotas = target_counts(N_COHORT_USERS)
    picked: list[int] = []
    audit_rows = []

    for name in ARCHETYPE_NAMES:
        available = labels.index[labels == name].to_numpy()
        want = quotas[name]
        cap = int(len(available) * MAX_SAMPLING_FRACTION)
        take = min(want, cap)

        if take < want:
            logger.warning(
                "Archetype '%s': wanted %d users but only %d available "
                "(cap %.0f%%); taking %d.",
                name,
                want,
                len(available),
                MAX_SAMPLING_FRACTION * 100,
                take,
            )

        chosen = rng.choice(available, size=take, replace=False)
        picked.extend(int(u) for u in chosen)

        audit_rows.append(
            {
                "Archetype": name,
                "Quota": want,
                "Available": len(available),
                "Sampled": take,
                "Sampling_Fraction": round(take / len(available), 4)
                if len(available)
                else 0.0,
            }
        )

    audit = pd.DataFrame(audit_rows)
    logger.info("Stratified sampling audit:\n%s", audit.to_string(index=False))
    return pd.Series(picked, name="userId"), audit


def temporal_split(group: pd.DataFrame) -> np.ndarray:
    """Leave-last-n split by timestamp."""
    n = len(group)
    n_train = max(1, round(n * TRAIN_SPLIT_RATIO))
    order = np.argsort(group["timestamp"].to_numpy(), kind="stable")
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:]] = "holdout"
    return labels


def build_cohort_ratings(
    ratings_df: pd.DataFrame, cohort_ids: pd.Series
) -> pd.DataFrame:
    subset = ratings_df[ratings_df["userId"].isin(list(cohort_ids))].copy()
    subset = subset.sort_values(["userId", "timestamp"], kind="stable")

    splits = []
    for _, grp in subset.groupby("userId", sort=False):
        splits.append(pd.Series(temporal_split(grp), index=grp.index))
    subset["split"] = pd.concat(splits)

    logger.info(
        "Cohort ratings: %d rows (%d train / %d holdout) across %d users.",
        len(subset),
        int((subset["split"] == "train").sum()),
        int((subset["split"] == "holdout").sum()),
        subset["userId"].nunique(),
    )
    return subset[["userId", "movieId", "rating", "timestamp", "split"]]


def infer_preferences(
    cohort_ratings: pd.DataFrame, movies_df: pd.DataFrame
) -> pd.DataFrame:
    """Derive stated preferences from the TRAIN split only."""
    genres_by_movie = dict(zip(movies_df["movieId"], movies_df["clean_genres"]))
    lang_by_movie = dict(zip(movies_df["movieId"], movies_df["language"]))
    canonical = set(CANONICAL_GENRES)

    train_liked = cohort_ratings[
        (cohort_ratings["split"] == "train")
        & (cohort_ratings["rating"] >= LIKE_THRESHOLD)
    ]

    rows = []
    for uid, grp in train_liked.groupby("userId", sort=False):
        genre_counter: Counter[str] = Counter()
        lang_counter: Counter[str] = Counter()

        for mid in grp["movieId"]:
            for g in str(genres_by_movie.get(mid, "")).split("|"):
                if g in canonical:
                    genre_counter[g] += 1
            lang = lang_by_movie.get(mid)
            if lang:
                lang_counter[lang] += 1

        top_genres = [g for g, _ in genre_counter.most_common(N_INFERRED_GENRES)]

        total_lang = sum(lang_counter.values()) or 1
        ranked_langs = [
            lang
            for lang, c in lang_counter.most_common()
            if c / total_lang >= MIN_INFERRED_LANGUAGE_SHARE
            and lang in COHORT_LANGUAGES
        ][:MAX_INFERRED_LANGUAGES]

        if not ranked_langs:
            most_common = lang_counter.most_common(1)
            ranked_langs = [most_common[0][0]] if most_common else ["English"]

        rows.append(
            {
                "userId": int(uid),  # pyright: ignore[reportArgumentType]
                "preferred_genres": "|".join(top_genres) if top_genres else "Drama",
                "preferred_language": "|".join(ranked_langs),
            }
        )

    prefs = pd.DataFrame(rows)
    logger.info("Inferred preferences for %d users from train split only.", len(prefs))
    return prefs


def build_archetype_affinity(
    cohort_ratings: pd.DataFrame,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Per-archetype item affinity, measured from real in-group consumption."""
    movie_ids = movies_df["movieId"].to_numpy()
    index_of = {int(m): i for i, m in enumerate(movie_ids)}

    archetype_of = dict(zip(users_df["userId"], users_df["archetype"]))
    train_liked = cohort_ratings[
        (cohort_ratings["split"] == "train")
        & (cohort_ratings["rating"] >= LIKE_THRESHOLD)
    ]

    affinity: dict[str, np.ndarray] = {}
    for name in ARCHETYPE_NAMES:
        vec = np.zeros(len(movie_ids), dtype=np.float64)
        member_ids = {u for u, a in archetype_of.items() if a == name}
        grp = train_liked[train_liked["userId"].isin(list(member_ids))]

        support = grp["movieId"].value_counts()
        support = support[support >= MIN_GROUP_SUPPORT]

        for mid, cnt in support.items():
            idx = index_of.get(int(mid))  # pyright: ignore[reportArgumentType]
            if idx is not None:
                vec[idx] = float(cnt)

        if vec.sum() <= 0:
            logger.warning(
                "Archetype '%s' produced an empty affinity vector; falling back "
                "to uniform over the catalogue.",
                name,
            )
            vec = np.ones(len(movie_ids), dtype=np.float64)

        affinity[name] = (vec / vec.sum()).astype(np.float32)
        logger.info(
            "Archetype '%s': affinity over %d titles from %d real users.",
            name,
            int((affinity[name] > 0).sum()),
            len(member_ids),
        )

    return affinity


def derive_affinity_tables(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    labels: pd.Series,
    cohort_ids: set[int],
) -> dict[str, dict[str, float]]:
    """Estimate genre and language affinity tables from real consumption."""
    from .cohort_spec import ARCHETYPE_BY_NAME

    estim_ids = [int(u) for u in labels.index if int(u) not in cohort_ids]
    estim_labels = labels.loc[estim_ids]
    logger.info(
        "Deriving affinity tables from %d held-out users (cohort's %d excluded).",
        len(estim_ids),
        len(cohort_ids),
    )

    observed = estim_labels.value_counts(normalize=True).to_dict()
    user_weight = {}
    for uid, arch in estim_labels.items():
        obs = observed.get(arch, 0.0)
        target = ARCHETYPE_BY_NAME[arch]["target_share"]
        user_weight[int(uid)] = (target / obs) if obs > 0 else 0.0  # pyright: ignore[reportArgumentType]

    liked = ratings_df[
        (ratings_df["rating"] >= LIKE_THRESHOLD)
        & (ratings_df["userId"].isin(estim_ids))
    ]
    liked = liked.assign(w=liked["userId"].map(user_weight).astype("float64"))

    genres_by_movie = dict(zip(movies_df["movieId"], movies_df["clean_genres"]))
    lang_by_movie = dict(zip(movies_df["movieId"], movies_df["language"]))

    canonical = set(CANONICAL_GENRES)
    arch_of = estim_labels.to_dict()

    genre_overall: dict[str, float] = {g: 0.0 for g in CANONICAL_GENRES}
    lang_overall: dict[str, float] = {lang: 0.0 for lang in COHORT_LANGUAGES}
    genre_by_arch: dict[str, dict[str, float]] = {
        a: {g: 0.0 for g in CANONICAL_GENRES} for a in ARCHETYPE_NAMES
    }
    lang_by_arch: dict[str, dict[str, float]] = {
        a: {lang: 0.0 for lang in COHORT_LANGUAGES} for a in ARCHETYPE_NAMES
    }

    uids = liked["userId"].to_numpy()
    mids = liked["movieId"].to_numpy()
    weights = liked["w"].to_numpy()

    for uid, mid, w in zip(uids, mids, weights):
        if not np.isfinite(w) or w <= 0:
            continue
        arch = arch_of.get(int(uid))  # pyright: ignore[reportArgumentType]
        if arch is None:
            continue

        for g in str(genres_by_movie.get(mid, "")).split("|"):
            if g in canonical:
                genre_overall[g] += w
                genre_by_arch[arch][g] += w

        lang = lang_by_movie.get(mid)
        if lang in lang_overall:
            lang_overall[lang] += w
            lang_by_arch[arch][lang] += w

    def _shares(mass: dict[str, float]) -> dict[str, float]:
        total = sum(mass.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in mass.items()}

    def _lift_table(
        overall: dict[str, float], by_arch: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Max lift of each category across archetypes, log-compressed to [0,1]."""
        pop = _shares(overall)
        arch_shares = {a: _shares(m) for a, m in by_arch.items()}

        lifts: dict[str, float] = {}
        for key, base in pop.items():
            if base <= 0:
                lifts[key] = 0.0
                continue
            lifts[key] = max(
                (arch_shares[a].get(key, 0.0) / base for a in ARCHETYPE_NAMES),
                default=0.0,
            )

        compressed = {k: float(np.log1p(max(v, 0.0))) for k, v in lifts.items()}
        peak = max(compressed.values()) if compressed else 0.0
        if peak <= 0:
            return {k: 0.0 for k in compressed}

        return {
            k: round(float(0.25 + 0.75 * (v / peak)), 4) for k, v in compressed.items()
        }

    tables = {
        "genre": _lift_table(genre_overall, genre_by_arch),
        "language": _lift_table(lang_overall, lang_by_arch),
    }

    logger.info(
        "Derived genre affinity (top 8): %s",
        sorted(tables["genre"].items(), key=lambda kv: -kv[1])[:8],
    )
    logger.info("Derived language affinity: %s", tables["language"])

    with open(os.path.join(PROCESSED_DIR, "cohort_affinity_tables.json"), "w") as fh:
        json.dump(tables, fh, indent=2)
    logger.info("Wrote data/processed/cohort_affinity_tables.json.")

    pd.DataFrame(
        [{"Type": "genre", "Key": k, "Affinity": v} for k, v in tables["genre"].items()]
        + [
            {"Type": "language", "Key": k, "Affinity": v}
            for k, v in tables["language"].items()
        ]
    ).to_csv(os.path.join(RESULTS_DIR, "derived_affinity_tables.csv"), index=False)

    return tables


def validate_outputs(
    users_df: pd.DataFrame, cohort_ratings: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    if not users_df["userId"].is_unique:
        raise ValueError("real_cohort_users.csv contains duplicate userIds.")

    if users_df["preferred_genres"].eq("").any():
        raise ValueError("real_cohort_users.csv contains empty preferred_genres.")

    if users_df["preferred_language"].eq("").any():
        raise ValueError("real_cohort_users.csv contains empty preferred_language.")

    if not cohort_ratings["split"].isin(["train", "holdout"]).all():
        raise ValueError("real_cohort_ratings.csv contains invalid split labels.")

    if cohort_ratings.duplicated(subset=["userId", "movieId"]).any():
        raise ValueError(
            "real_cohort_ratings.csv contains duplicate (user, movie) pairs."
        )

    valid_movies = set(movies_df["movieId"])
    if not cohort_ratings["movieId"].isin(list(valid_movies)).all():
        raise ValueError("real_cohort_ratings.csv references unknown movieIds.")

    per_user = cohort_ratings.groupby("userId")["split"].agg(
        lambda s: (s == "holdout").sum()
    )
    if (per_user == 0).any():
        n = int((per_user == 0).sum())
        raise ValueError(f"{n} cohort users have no holdout ratings.")

    logger.info("Validation passed for real cohort artifacts.")


def main() -> None:
    logger.info("Starting build_real_cohort.py...")
    logger.info("Cohort design context:\n%s", COHORT_DESIGN_NOTE)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    movies_df, ratings_df = load_inputs()

    shares, _ = compute_language_shares(ratings_df, movies_df)
    labels = label_users(shares)
    report_benchmark_population(labels, shares)

    cohort_ids, audit = sample_cohort(labels, rng)
    cohort_ratings = build_cohort_ratings(ratings_df, cohort_ids)

    prefs = infer_preferences(cohort_ratings, movies_df)
    users_df = prefs.merge(
        labels.rename("archetype").reset_index().rename(columns={"index": "userId"}),
        on="userId",
        how="left",
    )

    cohort_ratings = cohort_ratings[
        cohort_ratings["userId"].isin(users_df["userId"].tolist())
    ]

    validate_outputs(users_df, cohort_ratings, movies_df)

    derive_affinity_tables(
        ratings_df, movies_df, labels, {int(u) for u in users_df["userId"]}
    )

    affinity = build_archetype_affinity(cohort_ratings, users_df, movies_df)

    users_df.to_csv(REAL_USERS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", REAL_USERS_PATH, len(users_df))

    cohort_ratings.to_csv(REAL_RATINGS_PATH, index=False)
    logger.info("Wrote %s (%d rows).", REAL_RATINGS_PATH, len(cohort_ratings))

    audit.to_csv(
        os.path.join(RESULTS_DIR, "real_cohort_sampling_audit.csv"), index=False
    )

    with open(HELDOUT_USER_IDS_PATH, "w") as fh:
        json.dump(sorted(int(u) for u in users_df["userId"]), fh)
    logger.info(
        "Wrote %s (%d ids to exclude from CF).", HELDOUT_USER_IDS_PATH, len(users_df)
    )

    joblib.dump(
        {"affinity": affinity, "movie_ids": movies_df["movieId"].to_numpy()},
        ARCHETYPE_AFFINITY_PATH,
        compress=3,
    )
    logger.info("Wrote %s.", ARCHETYPE_AFFINITY_PATH)

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
