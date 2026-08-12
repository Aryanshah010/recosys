from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

SCORER_FUNCTIONS = {
    "build_genre_weight_vector",
    "build_genre_onehot_from_list",
    "compute_language_preference_scores",
    "compute_genre_affinity_scores",
}


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "localization_config" in node.module
        ):
            names.update(alias.name for alias in node.names)
    return names


def test_generator_does_not_import_scorer_functions():
    generator = PROJECT_ROOT / "src" / "generate_synthetic_cohort.py"
    leaked = _imported_names(generator) & SCORER_FUNCTIONS
    assert not leaked, (
        f"generate_synthetic_cohort.py imports scorer functions {leaked} from "
        "localization_config. Ground truth and the ranking model would then "
        "share a formula, making RQ1 circular."
    )


SHARED_CONSTANTS = {
    "COHORT_GENRE_AFFINITY",
    "COHORT_LANGUAGE_AFFINITY",
    "STANDARD_HYBRID_WEIGHTS",
    "LOCALIZED_HYBRID_WEIGHTS",
    "AFFINITY_ONLY_WEIGHTS",
    "COHORT_GENRE_COMPONENT",
    "COHORT_LANGUAGE_COMPONENT",
}

CONSUMER_MODULES = [
    Path("evaluation") / "evaluation_metrics.py",
    Path("evaluation") / "sensitivity_analysis.py",
    Path("engine") / "hybrid_fusion.py",
    Path("api") / "recommender_service.py",
]


@pytest.mark.parametrize("relative", CONSUMER_MODULES, ids=lambda p: p.name)
def test_consumers_import_constants_rather_than_copying(relative: Path):
    text = (PROJECT_ROOT / relative).read_text()
    assert "localization_config import" in text, (
        f"{relative} does not import from localization_config."
    )

    tree = ast.parse(text)
    redefined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in SHARED_CONSTANTS:
                    redefined.add(target.id)
    assert not redefined, f"{relative} re-defines imported constants: {redefined}"

    for literal in ("0.625", "0.375"):
        assert literal not in text, (
            f"{relative} hard-codes the hybrid weight {literal} instead of "
            "importing STANDARD_HYBRID_WEIGHTS. A future reweighting would then "
            "silently desynchronise this module from the evaluated model."
        )


@pytest.mark.skipif(
    not (PROCESSED / "cf_excluded_user_ids.json").exists(),
    reason="pipeline artifacts not built",
)
def test_evaluation_cohort_excluded_from_cf_training():
    from surprise import dump

    with open(PROCESSED / "cf_excluded_user_ids.json") as fh:
        excluded = {int(u) for u in json.load(fh)}

    _, svd = dump.load(str(PROCESSED / "svd_model.pkl"))
    trainset = svd.trainset

    leaked = []
    for uid in excluded:
        try:
            trainset.to_inner_uid(uid)
            leaked.append(uid)
        except ValueError:
            pass

    assert not leaked, f"{len(leaked)} evaluation-cohort users are in the CF trainset."


@pytest.mark.skipif(
    not (PROCESSED / "real_cohort_ratings.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_inferred_preferences_use_train_split_only():
    ratings = pd.read_csv(PROCESSED / "real_cohort_ratings.csv")
    users = pd.read_csv(PROCESSED / "real_cohort_users.csv")

    liked_train = ratings[(ratings["split"] == "train") & (ratings["rating"] >= 3.5)]
    users_with_train_signal = set(liked_train["userId"].unique())

    missing = set(users["userId"]) - users_with_train_signal
    assert not missing, (
        f"{len(missing)} users have inferred preferences but no liked training "
        "items, so their profile must have used holdout data."
    )


@pytest.mark.skipif(
    not (PROCESSED / "real_cohort_ratings.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_temporal_split_is_actually_temporal():
    ratings = pd.read_csv(PROCESSED / "real_cohort_ratings.csv")
    assert "timestamp" in ratings.columns

    violations = 0
    for _, grp in ratings.groupby("userId"):
        train = grp[grp["split"] == "train"]["timestamp"]
        holdout = grp[grp["split"] == "holdout"]["timestamp"]
        if len(train) and len(holdout) and holdout.min() < train.max():
            violations += 1

    assert violations == 0, f"{violations} users have a non-temporal split."


@pytest.mark.skipif(
    not (PROCESSED / "synthetic_ratings.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_no_duplicate_user_item_pairs():
    for name in ("synthetic_ratings.csv", "real_cohort_ratings.csv"):
        path = PROCESSED / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        dupes = df.duplicated(subset=["userId", "movieId"]).sum()
        assert dupes == 0, f"{name} has {dupes} duplicate (user, item) pairs."


def test_hybrid_weight_presets_are_convex_and_comparable():
    from src.localization_config import (
        AFFINITY_ONLY_WEIGHTS,
        LOCALIZED_HYBRID_WEIGHTS,
        STANDARD_HYBRID_WEIGHTS,
    )

    for w in (STANDARD_HYBRID_WEIGHTS, LOCALIZED_HYBRID_WEIGHTS, AFFINITY_ONLY_WEIGHTS):
        assert abs(w.sum() - 1.0) < 1e-6

    std_ratio = STANDARD_HYBRID_WEIGHTS.cf / STANDARD_HYBRID_WEIGHTS.cbf
    loc_ratio = LOCALIZED_HYBRID_WEIGHTS.cf / LOCALIZED_HYBRID_WEIGHTS.cbf
    assert abs(std_ratio - loc_ratio) < 1e-6, (
        "The localized hybrid changed the CF:CBF ratio, which confounds the "
        "comparison against the standard hybrid."
    )


@pytest.mark.skipif(
    not (PROCESSED / "svd_model.pkl").exists(),
    reason="pipeline artifacts not built",
)
def test_served_model_matches_evaluated_model():
    import numpy as np

    from api.recommender_service import EVALUATION_MODEL_NAMES, get_service

    svc = get_service()
    user_id = int(svc.users_df["userId"].iloc[0])
    liked = svc._liked_movie_ids(user_id, [])
    profile = svc._profile_obj(user_id)

    expected = {
        "popularity": svc.engine.get_popularity_scores(),
        "cf": svc.engine.get_cf_scores(user_id, liked),
        "cbf": svc.engine.get_cbf_scores(liked),
        "affinity": svc.engine.get_affinity_only_scores(profile),
        "hybrid": svc.engine.get_standard_hybrid_scores(user_id, liked),
        "localized": svc.engine.get_localized_hybrid_scores(user_id, liked, profile),
    }

    assert set(expected) == set(EVALUATION_MODEL_NAMES), (
        "The demo model set has drifted from the evaluated model set."
    )

    for model, reference in expected.items():
        served = svc._model_scores(model, user_id, liked, profile)
        assert np.allclose(served, reference), (
            f"Demo model '{model}' ({EVALUATION_MODEL_NAMES[model]}) does not "
            "match engine/hybrid_fusion.py. Note this pins demo == engine, NOT "
            "engine == evaluation harness: the harness scores a sampled "
            "candidate set and normalises over it, so its rankings are close "
            "but not identical. What both share is the weights and affinity "
            "tables, pinned by test_consumers_import_constants_rather_than_copying."
        )


def test_cohort_spec_is_internally_consistent():
    from src.cohort_spec import ARCHETYPE_SPECS, target_counts

    assert abs(sum(a["target_share"] for a in ARCHETYPE_SPECS) - 1.0) < 1e-6
    counts = target_counts(400)
    assert sum(counts.values()) == 400


def test_television_is_not_a_genre():
    from src.mappings import CANONICAL_GENRE_MAP, CANONICAL_GENRES

    assert "TV" not in CANONICAL_GENRES, (
        "'TV' is a canonical genre again. This is a movie recommender; TMDB's "
        "'TV Movie' must fall through to 'Other' rather than becoming a "
        "user-visible label."
    )
    assert "TV Movie" not in CANONICAL_GENRE_MAP, (
        "TMDB's 'TV Movie' genre is being mapped in again."
    )


@pytest.mark.skipif(
    not (PROCESSED / "movies_final.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_catalogue_contains_no_television():
    movies = pd.read_csv(
        PROCESSED / "movies_final.csv", usecols=["movieId", "tmdbId", "clean_genres"]
    )

    tagged = (
        movies["clean_genres"].astype(str).str.split("|").apply(lambda g: "TV" in g)
    )
    assert not tagged.any(), (
        f"{int(tagged.sum())} catalogue titles carry a 'TV' genre token."
    )

    exclusions_path = PROCESSED / "television_exclusions.json"
    if not exclusions_path.exists():
        pytest.skip("television_exclusions.json not present")

    with open(exclusions_path) as fh:
        excluded = {int(t) for t in json.load(fh)["excluded_tmdb_ids"]}

    leaked = movies[movies["tmdbId"].isin(excluded)]
    assert leaked.empty, (
        f"{len(leaked)} titles TMDB classifies as television survived into the "
        f"catalogue, e.g. tmdbId {leaked['tmdbId'].head(5).tolist()}. "
        "Is drop_television() still wired into build_unified_catalog()?"
    )


@pytest.mark.skipif(
    not (PROCESSED / "real_cohort_ratings.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_archetype_labels_do_not_use_holdout_data():
    
    from src.cohort_spec import LIKE_THRESHOLD, label_from_language_shares

    ratings = pd.read_csv(PROCESSED / "real_cohort_ratings.csv")
    users = pd.read_csv(PROCESSED / "real_cohort_users.csv")
    movies = pd.read_csv(
        PROCESSED / "movies_final.csv", usecols=["movieId", "language"]
    )
    language_by_movie = dict(zip(movies["movieId"], movies["language"]))

    liked_train = ratings[
        (ratings["split"] == "train") & (ratings["rating"] >= LIKE_THRESHOLD)
    ].copy()
    liked_train["language"] = liked_train["movieId"].map(language_by_movie)

    counts = pd.crosstab(liked_train["userId"], liked_train["language"])
    shares = counts.div(counts.sum(axis=1), axis=0)

    stored = dict(zip(users["userId"], users["archetype"]))
    mismatched = [
        uid
        for uid, row in shares.iterrows()
        if uid in stored and label_from_language_shares(row.to_dict()) != stored[uid] # type: ignore
    ]

    ratio = len(mismatched) / max(len(stored), 1)
    assert ratio < 0.05, (
        f"{len(mismatched)} of {len(stored)} archetype labels ({ratio:.1%}) are "
        "not reproducible from the train split. Cohort selection is reading "
        "holdout data -- see the train_mask() call in build_real_cohort.main()."
    )


def test_sensitivity_mixes_match_the_archetype_spec():
    from evaluation.sensitivity_analysis import ALTERNATIVE_MIXES
    from src.cohort_spec import ARCHETYPE_NAMES

    for name, mix in ALTERNATIVE_MIXES.items():
        assert set(mix) == set(ARCHETYPE_NAMES), (
            f"Sensitivity mix '{name}' has keys {sorted(set(mix))}, but the "
            f"cohort spec defines {sorted(ARCHETYPE_NAMES)}. Renaming an "
            "archetype silently desynchronises the sweep."
        )
        assert abs(sum(mix.values()) - 1.0) < 1e-6, f"Mix '{name}' must sum to 1.0."


@pytest.mark.skipif(
    not (PROCESSED / "movies_final.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_nepali_catalogue_scarcity_is_still_true():
    movies = pd.read_csv(PROCESSED / "movies_final.csv", usecols=["language"])
    n_nepali = int((movies["language"] == "Nepali").sum())
    assert n_nepali < 10, (
        f"Catalogue now has {n_nepali} Nepali titles. The thesis argues "
        "localization cannot mean national-cinema retrieval because the "
        "catalogue contains almost none - revisit that argument."
    )
