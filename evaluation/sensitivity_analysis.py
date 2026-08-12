from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

import evaluation.evaluation_metrics as em
from src.cohort_spec import ARCHETYPE_BY_NAME, ARCHETYPE_NAMES
from src.frames import as_frame, as_series, mean_frame, to_float
from src.localization_config import (
    AFFINITY_SOURCE,
    COHORT_GENRE_AFFINITY,
    COHORT_LANGUAGE_AFFINITY,
    PRIOR_GENRE_AFFINITY,
    PRIOR_LANGUAGE_AFFINITY,
    STANDARD_HYBRID_WEIGHTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"

LAMBDA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

ALTERNATIVE_MIXES: dict[str, dict[str, float]] = {
    "baseline_design": {
        name: ARCHETYPE_BY_NAME[name]["target_share"] for name in ARCHETYPE_NAMES
    },
    "hollywood_heavy": {
        "hollywood": 0.55,
        "anime": 0.15,
        "bollywood": 0.15,
        "korean": 0.10,
        "mixed": 0.05,
    },
    "regional_heavy": {
        "hollywood": 0.20,
        "anime": 0.20,
        "bollywood": 0.30,
        "korean": 0.20,
        "mixed": 0.10,
    },
    "high_diversity": {
        "hollywood": 0.25,
        "anime": 0.15,
        "bollywood": 0.15,
        "korean": 0.15,
        "mixed": 0.30,
    },
    "uniform": {name: 0.20 for name in ARCHETYPE_NAMES},
}


def _rank_top_k(scored: list[tuple[int, float]], k: int) -> list[int]:
    return [mid for mid, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]


def affinity_weight_sweep(track: str = "real") -> pd.DataFrame:
    shared = em.load_shared_artifacts()
    (svd, cbf_matrix, movie_ids, movie_index, clean_genres, language, quality) = shared

    ratings_df, users_df = em.load_track(track)
    splits = em.build_holdout_split(ratings_df)
    profiles = users_df.set_index("userId").to_dict("index")
    pop_bins = em.build_popularity_bins(quality)

    eval_uids = em.select_eval_users(splits)
    logger.info("Lambda sweep on track '%s' over %d users.", track, len(eval_uids))

    records = []
    for i, uid in enumerate(eval_uids):
        if (i + 1) % 50 == 0:
            logger.info("  ... %d / %d users", i + 1, len(eval_uids))

        train_ids, train_pos, holdout_pos = splits[uid]
        profile = profiles.get(uid, {})
        pref_genres = em._parse_pipe_list(profile.get("preferred_genres", ""))
        pref_languages = em._parse_pipe_list(profile.get("preferred_language", ""))
        archetype = str(profile.get("archetype", "unknown"))

        candidates = em.build_candidate_set(
            train_ids, holdout_pos, movie_ids, movie_index, pop_bins, uid
        )
        if not candidates:
            continue

        valid_train = [movie_index[m] for m in train_pos if m in movie_index]
        if valid_train:
            from sklearn.metrics.pairwise import linear_kernel

            sims = linear_kernel(cbf_matrix[valid_train], cbf_matrix).mean(axis=0)
            cbf_arr = em._normalise(np.asarray(sims, dtype=np.float32).ravel())
        else:
            cbf_arr = np.zeros(len(movie_index), dtype=np.float32)

        cf_norm = em._cold_start_cf_scores(svd, train_pos, candidates)
        aff_norm = em.compute_affinity_score(
            candidates, movie_index, clean_genres, language, pref_genres, pref_languages
        )

        for lam in LAMBDA_GRID:
            scale = 1.0 - lam
            scored = []
            for mid in candidates:
                idx = movie_index.get(mid)
                if idx is None:
                    continue
                score = (
                    scale * STANDARD_HYBRID_WEIGHTS.cf * float(cf_norm.get(mid, 0.0))
                    + scale * STANDARD_HYBRID_WEIGHTS.cbf * float(cbf_arr[idx])
                    + lam * float(aff_norm.get(mid, 0.0))
                )
                scored.append((mid, score))

            top_k = _rank_top_k(scored, em.K)
            metrics = em.compute_metrics(
                top_k,
                holdout_pos,
                pref_genres,
                pref_languages,
                movie_index,
                clean_genres,
                language,
            )
            records.append(
                {
                    "Track": track,
                    "Lambda": lam,
                    "UserId": uid,
                    "Archetype": archetype,
                    **metrics,
                }
            )

    per_user = pd.DataFrame(records)
    per_user.to_csv(
        f"{RESULTS_DIR}/sensitivity_lambda_user_level_{track}.csv", index=False
    )

    summary_columns = [
        "Precision@10",
        "Recall@10",
        "HR@10",
        "MRR@10",
        "NDCG@10",
        "Language_Diversity",
        "Genre_Diversity",
        "Filter_Bubble_Score",
        "Preference_Escape@10",
    ]
    summary = mean_frame(per_user, "Lambda", summary_columns).reset_index()

    print(
        f"\n{'=' * 78}\n  LAMBDA SWEEP - accuracy vs diversity trade-off [{track}]\n{'=' * 78}"
    )
    print(summary.to_string(index=False))
    summary.to_csv(f"{RESULTS_DIR}/sensitivity_lambda_summary_{track}.csv", index=False)

    return per_user


def archetype_mix_sweep(per_user: pd.DataFrame, track: str = "real") -> pd.DataFrame:
    user_level_path = f"{RESULTS_DIR}/evaluation_user_level.csv"
    if not os.path.exists(user_level_path):
        logger.warning("No %s; skipping archetype mix sweep.", user_level_path)
        return pd.DataFrame()

    df = pd.read_csv(user_level_path)
    df = df[df["Track"] == track]
    if df.empty:
        logger.warning("No rows for track '%s'; skipping mix sweep.", track)
        return pd.DataFrame()

    unique_users = as_frame(df.drop_duplicates(subset="UserId"))
    observed = as_series(unique_users["Archetype"]).value_counts(normalize=True)

    rows = []
    for mix_name, mix in ALTERNATIVE_MIXES.items():
        for model, grp in df.groupby("Model"):
            weighted_ndcg = 0.0
            weighted_novelty = 0.0
            total_w = 0.0
            for arch, sub in grp.groupby("Archetype"):
                share = mix.get(str(arch), 0.0)
                obs = to_float(observed.get(arch, 0.0))
                if share <= 0 or obs <= 0:
                    continue
                weighted_ndcg += share * to_float(sub["NDCG@10"].mean())
                weighted_novelty += share * to_float(sub["Preference_Escape@10"].mean())
                total_w += share
            if total_w > 0:
                rows.append(
                    {
                        "Mix": mix_name,
                        "Model": model,
                        "NDCG@10": round(weighted_ndcg / total_w, 4),
                        "Preference_Escape@10": round(weighted_novelty / total_w, 4),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    pivot = out.pivot(index="Model", columns="Mix", values="NDCG@10")
    print(f"\n{'=' * 78}\n  ARCHETYPE MIX SENSITIVITY - NDCG@10 [{track}]\n{'=' * 78}")
    print(pivot.round(4).to_string())
    print(
        "\n  If Localized_Hybrid's ranking relative to the baselines holds across\n"
        "  every column, the absent survey does not threaten the conclusion."
    )

    out.to_csv(f"{RESULTS_DIR}/sensitivity_archetype_mix_{track}.csv", index=False)
    return out


def affinity_table_comparison() -> pd.DataFrame:
    rows = []
    for key in sorted(set(PRIOR_GENRE_AFFINITY) | set(COHORT_GENRE_AFFINITY)):
        rows.append(
            {
                "Type": "genre",
                "Key": key,
                "Prior_HandSet": PRIOR_GENRE_AFFINITY.get(key),
                "Derived_FromData": COHORT_GENRE_AFFINITY.get(key),
            }
        )
    for key in sorted(set(PRIOR_LANGUAGE_AFFINITY) | set(COHORT_LANGUAGE_AFFINITY)):
        rows.append(
            {
                "Type": "language",
                "Key": key,
                "Prior_HandSet": PRIOR_LANGUAGE_AFFINITY.get(key),
                "Derived_FromData": COHORT_LANGUAGE_AFFINITY.get(key),
            }
        )

    out = pd.DataFrame(rows)
    print(
        f"\n{'=' * 78}\n  AFFINITY TABLES: hand-set prior vs data-derived\n{'=' * 78}"
    )
    print(f"  Active source: {AFFINITY_SOURCE}")
    print(out.to_string(index=False))
    out.to_csv(f"{RESULTS_DIR}/sensitivity_affinity_tables.csv", index=False)
    return out


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    affinity_table_comparison()

    for track, cfg in em.TRACKS.items():
        if not os.path.exists(cfg["ratings"]):
            logger.warning("Track '%s' ratings missing; skipping.", track)
            continue
        per_user = affinity_weight_sweep(track)
        archetype_mix_sweep(per_user, track)

    logger.info("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
