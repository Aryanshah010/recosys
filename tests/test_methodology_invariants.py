"""Regression guards for the methodological properties the thesis claims."""

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
    """The synthetic generator must not share scoring functions with the model."""
    generator = PROJECT_ROOT / "src" / "generate_synthetic_cohort.py"
    leaked = _imported_names(generator) & SCORER_FUNCTIONS
    assert not leaked, (
        f"generate_synthetic_cohort.py imports scorer functions {leaked} from "
        "localization_config. Ground truth and the ranking model would then "
        "share a formula, making RQ1 circular."
    )


def test_evaluation_imports_constants_rather_than_copying():
    """Evaluation must not hand-copy weights that live in localization_config."""
    text = (PROJECT_ROOT / "evaluation" / "evaluation_metrics.py").read_text()
    assert "from src.localization_config import" in text

    tree = ast.parse(text)
    redefined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "COHORT_GENRE_AFFINITY",
                    "COHORT_LANGUAGE_AFFINITY",
                    "STANDARD_HYBRID_WEIGHTS",
                    "LOCALIZED_HYBRID_WEIGHTS",
                }:
                    redefined.add(target.id)
    assert not redefined, (
        f"evaluation_metrics.py re-defines imported constants: {redefined}"
    )


@pytest.mark.skipif(
    not (PROCESSED / "cf_excluded_user_ids.json").exists(),
    reason="pipeline artifacts not built",
)
def test_evaluation_cohort_excluded_from_cf_training():
    """No real-cohort user may appear in the SVD trainset."""
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
    """Preferences must be derivable from the train split alone."""
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
    """Every holdout item must be no earlier than the user's latest train item."""
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
    """The two hybrids must differ only by the affinity term."""
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


def test_cohort_spec_is_internally_consistent():
    from src.cohort_spec import ARCHETYPE_SPECS, target_counts

    assert abs(sum(a["target_share"] for a in ARCHETYPE_SPECS) - 1.0) < 1e-6
    counts = target_counts(400)
    assert sum(counts.values()) == 400


@pytest.mark.skipif(
    not (PROCESSED / "movies_final.csv").exists(),
    reason="pipeline artifacts not built",
)
def test_nepali_catalogue_scarcity_is_still_true():
    """Pin the finding that motivates the audience-localization reframe.

    This guard encodes a deliberate scope decision, not merely an observed data
    fact. Supplementing the catalogue with Nepali titles from the TMDB API was
    considered and rejected, because it could not produce a measurable result:

      1. No cohort user prefers Nepali (0 of 400 real, 0 of 400 synthetic).
         Preferences are inferred from observed consumption, and there is no
         Nepali consumption in MovieLens to infer from.
      2. No MovieLens rating exists for any Nepali film, so an added title can
         never be a held-out positive - only a negative-pool distractor. Adding
         titles cannot raise HR/NDCG/MRR; it can only leave them flat or lower
         them through new false positives.
      3. Language affinity is estimated as consumption *lift*, so Nepali stays
         pinned at the 0.25 floor however much catalogue is added. Adding rows
         does not create consumption.

    So do not "fix" this test by adding Nepali titles. Doing so would weaken the
    catalogue's quality floor without making any new claim testable.
    """
    movies = pd.read_csv(PROCESSED / "movies_final.csv", usecols=["language"])
    n_nepali = int((movies["language"] == "Nepali").sum())
    assert n_nepali < 10, (
        f"Catalogue now has {n_nepali} Nepali titles. The thesis argues "
        "localization cannot mean national-cinema retrieval because the "
        "catalogue contains almost none - revisit that argument."
    )
