"""Master evaluation harness for RQ1 (accuracy) and RQ2 (diversity / bias)."""

from __future__ import annotations

import contextlib
import logging
import os
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import linear_kernel
from surprise import dump

from src.frames import agg_frame, mean_frame, mean_series, pivot_mean
from src.localization_config import (
    AFFINITY_ONLY_WEIGHTS,
    AFFINITY_SOURCE,
    COHORT_GENRE_AFFINITY,
    COHORT_GENRE_COMPONENT,
    COHORT_LANGUAGE_AFFINITY,
    COHORT_LANGUAGE_COMPONENT,
    LOCALIZED_HYBRID_WEIGHTS,
    STANDARD_HYBRID_WEIGHTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SVD_MODEL_PATH = os.path.join(PROCESSED_DIR, "svd_model.pkl")
CBF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "cbf_matrix.pkl")
CBF_META_PATH = os.path.join(PROCESSED_DIR, "cbf_metadata.pkl")

TRACKS = {
    "real": {
        "ratings": os.path.join(PROCESSED_DIR, "real_cohort_ratings.csv"),
        "users": os.path.join(PROCESSED_DIR, "real_cohort_users.csv"),
        "label": "Real proxy cohort (real human ratings)",
    },
    "synthetic": {
        "ratings": os.path.join(PROCESSED_DIR, "synthetic_ratings.csv"),
        "users": os.path.join(PROCESSED_DIR, "synthetic_users.csv"),
        "label": "Synthetic cohort (controlled simulation)",
    },
}

K = 10
MAX_EVAL_USERS = 400
MIN_HOLDOUT_POS = 1
HOLDOUT_RATING_THRESHOLD = 3.5
RANDOM_SEED = 42
N_NEGATIVE_CANDIDATES = 1_000

NEGATIVE_SAMPLING = "popularity_stratified"
N_POPULARITY_BINS = 10

MODEL_ORDER = [
    "Popularity",
    "CF_ColdStart",
    "CBF",
    "Affinity_Only",
    "NonLocal_Hybrid",
    "Localized_Hybrid",
]

ALPHA = 0.05


def _check(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")


def _parse_pipe_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split("|") if x.strip()]


def load_shared_artifacts() -> tuple:
    logger.info("Loading SVD model ...")
    _check(SVD_MODEL_PATH)
    _, svd = dump.load(SVD_MODEL_PATH)

    logger.info("Loading CBF matrix and metadata ...")
    _check(CBF_MATRIX_PATH)
    _check(CBF_META_PATH)
    cbf_matrix = joblib.load(CBF_MATRIX_PATH)
    cbf_meta = joblib.load(CBF_META_PATH)

    quality = cbf_meta.get("quality_scores")
    if quality is None:
        quality = np.zeros(cbf_meta["n_movies"], dtype=np.float32)

    logger.info(
        "Artifacts loaded: %d CBF movies. Affinity tables source: %s.",
        len(cbf_meta["movie_ids"]),
        AFFINITY_SOURCE,
    )
    return (
        svd,
        cbf_matrix,
        cbf_meta["movie_ids"],
        cbf_meta["movie_index"],
        cbf_meta["clean_genres"],
        cbf_meta["language"],
        np.asarray(quality, dtype=np.float32),
    )


def load_track(track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = TRACKS[track]
    _check(cfg["ratings"])
    _check(cfg["users"])
    ratings_df = pd.read_csv(cfg["ratings"])
    users_df = pd.read_csv(cfg["users"])
    users_df["userId"] = users_df["userId"].astype(int)
    logger.info(
        "Track '%s': %d ratings, %d users.", track, len(ratings_df), len(users_df)
    )
    return ratings_df, users_df


def build_holdout_split(
    ratings_df: pd.DataFrame,
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    splits: dict[int, tuple[list[int], list[int], list[int]]] = {}
    for uid, grp in ratings_df.groupby("userId"):
        train_ids = grp.loc[grp["split"] == "train", "movieId"].astype(int).tolist()
        train_pos = (
            grp.loc[
                (grp["split"] == "train") & (grp["rating"] >= HOLDOUT_RATING_THRESHOLD),
                "movieId",
            ]
            .astype(int)
            .tolist()
        )
        holdout_pos = (
            grp.loc[
                (grp["split"] == "holdout")
                & (grp["rating"] >= HOLDOUT_RATING_THRESHOLD),
                "movieId",
            ]
            .astype(int)
            .tolist()
        )
        if len(holdout_pos) >= MIN_HOLDOUT_POS and train_pos:
            splits[int(uid)] = (train_ids, train_pos, holdout_pos) # pyright: ignore[reportArgumentType]
    return splits


def build_popularity_bins(quality: np.ndarray) -> np.ndarray:
    """Assign each catalogue item to a popularity decile."""
    ranks = stats.rankdata(quality, method="average") / max(len(quality), 1) # pyright: ignore[reportAttributeAccessIssue]
    return np.clip((ranks * N_POPULARITY_BINS).astype(int), 0, N_POPULARITY_BINS - 1)


def build_candidate_set(
    train_ids: list[int],
    holdout_pos: list[int],
    movie_ids: np.ndarray,
    movie_index: dict,
    pop_bins: np.ndarray,
    user_id: int,
) -> list[int]:
    """Sample negatives, optionally matched to the positives' popularity band."""
    excluded = set(train_ids) | set(holdout_pos)
    rng = np.random.default_rng(RANDOM_SEED + user_id)

    pool_mask = np.array([mid not in excluded for mid in movie_ids])
    negative_pool = movie_ids[pool_mask]
    if len(negative_pool) == 0:
        return []

    n_negatives = min(N_NEGATIVE_CANDIDATES, len(negative_pool))

    if NEGATIVE_SAMPLING != "popularity_stratified":
        negatives = rng.choice(negative_pool, size=n_negatives, replace=False).tolist()
        return sorted([*holdout_pos, *negatives])

    pos_bins = [pop_bins[movie_index[m]] for m in holdout_pos if m in movie_index]
    if not pos_bins:
        negatives = rng.choice(negative_pool, size=n_negatives, replace=False).tolist()
        return sorted([*holdout_pos, *negatives])

    target = np.bincount(pos_bins, minlength=N_POPULARITY_BINS).astype(float)
    target = target / target.sum()

    pool_bins = np.array([pop_bins[movie_index[m]] for m in negative_pool])

    chosen: list[int] = []
    for b in range(N_POPULARITY_BINS):
        want = round(target[b] * n_negatives)
        if want <= 0:
            continue
        in_bin = negative_pool[pool_bins == b]
        if len(in_bin) == 0:
            continue
        take = min(want, len(in_bin))
        chosen.extend(rng.choice(in_bin, size=take, replace=False).tolist())

    if len(chosen) < n_negatives:
        remaining = np.setdiff1d(
            negative_pool, np.array(chosen, dtype=negative_pool.dtype)
        )
        if len(remaining) > 0:
            top_up = min(n_negatives - len(chosen), len(remaining))
            chosen.extend(rng.choice(remaining, size=top_up, replace=False).tolist())

    return sorted([*holdout_pos, *chosen])


def _normalise(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _cold_start_cf_scores(
    svd, train_pos: list[int], candidates: list[int]
) -> dict[int, float]:
    trainset = svd.trainset
    global_mean = float(trainset.global_mean)

    qi_liked: list[np.ndarray] = []
    for mid in train_pos:
        with contextlib.suppress(ValueError):
            qi_liked.append(svd.qi[trainset.to_inner_iid(mid)])

    p_hat = np.mean(qi_liked, axis=0) if qi_liked else np.zeros(svd.qi.shape[1])

    raw = np.empty(len(candidates), dtype=np.float32)
    for i, mid in enumerate(candidates):
        try:
            inner = trainset.to_inner_iid(mid)
            raw[i] = global_mean + svd.bi[inner] + float(svd.qi[inner] @ p_hat)
        except ValueError:
            raw[i] = global_mean

    return dict(zip(candidates, _normalise(raw)))


def compute_affinity_score(
    candidates: list[int],
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
    pref_genres: list[str],
    pref_languages: list[str],
) -> dict[int, float]:
    """Cohort-affinity score."""
    pref_lang_set = set(pref_languages)
    pref_genre_set = set(pref_genres)

    raw = np.empty(len(candidates), dtype=np.float32)
    for i, mid in enumerate(candidates):
        idx = movie_index.get(mid)
        if idx is None:
            raw[i] = 0.0
            continue

        matched = [
            COHORT_GENRE_AFFINITY.get(g, 0.0)
            for g in str(clean_genres[idx]).split("|")
            if g and g in pref_genre_set
        ]
        genre_score = float(np.mean(matched)) if matched else 0.0

        m_lang = language[idx]
        lang_score = (
            COHORT_LANGUAGE_AFFINITY.get(m_lang, 0.0)
            if m_lang in pref_lang_set
            else 0.0
        )

        raw[i] = (
            COHORT_GENRE_COMPONENT * genre_score
            + COHORT_LANGUAGE_COMPONENT * lang_score
        )

    return dict(zip(candidates, _normalise(raw)))


def score_all_models(
    train_pos: list[int],
    candidates: list[int],
    pref_genres: list[str],
    pref_languages: list[str],
    svd,
    cbf_matrix,
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
    quality: np.ndarray,
) -> dict[str, list[tuple[int, float]]]:
    valid_train = [movie_index[m] for m in train_pos if m in movie_index]
    if valid_train:
        sims = linear_kernel(cbf_matrix[valid_train], cbf_matrix).mean(axis=0)
        cbf_arr = _normalise(np.asarray(sims, dtype=np.float32).ravel())
    else:
        cbf_arr = np.zeros(len(movie_index), dtype=np.float32)

    cf_norm = _cold_start_cf_scores(svd, train_pos, candidates)
    aff_norm = compute_affinity_score(
        candidates, movie_index, clean_genres, language, pref_genres, pref_languages
    )

    scores: dict[str, list[tuple[int, float]]] = {m: [] for m in MODEL_ORDER}

    for mid in candidates:
        idx = movie_index.get(mid)
        if idx is None:
            continue

        cf = float(cf_norm.get(mid, 0.0))
        cbf = float(cbf_arr[idx])
        aff = float(aff_norm.get(mid, 0.0))
        pop = float(quality[idx])

        scores["Popularity"].append((mid, pop))
        scores["CF_ColdStart"].append((mid, cf))
        scores["CBF"].append((mid, cbf))
        scores["Affinity_Only"].append((mid, AFFINITY_ONLY_WEIGHTS.localization * aff))
        scores["NonLocal_Hybrid"].append(
            (mid, STANDARD_HYBRID_WEIGHTS.cf * cf + STANDARD_HYBRID_WEIGHTS.cbf * cbf)
        )
        scores["Localized_Hybrid"].append(
            (
                mid,
                LOCALIZED_HYBRID_WEIGHTS.cf * cf
                + LOCALIZED_HYBRID_WEIGHTS.cbf * cbf
                + LOCALIZED_HYBRID_WEIGHTS.localization * aff,
            )
        )

    return scores


def precision_at_k(top_k: list[int], ground_truth: list[int]) -> float:
    return len(set(top_k) & set(ground_truth)) / K


def precision_ceiling(ground_truth: list[int]) -> float:
    """Maximum attainable Precision@K given how few holdout items exist."""
    return min(len(ground_truth), K) / K


def recall_at_k(top_k: list[int], ground_truth: list[int]) -> float:
    if not ground_truth:
        return 0.0
    return len(set(top_k) & set(ground_truth)) / len(ground_truth)


def hit_rate_at_k(top_k: list[int], ground_truth: list[int]) -> float:
    """1 if any holdout positive is retrieved. Not capped by |holdout|."""
    return 1.0 if set(top_k) & set(ground_truth) else 0.0


def mrr_at_k(top_k: list[int], ground_truth: list[int]) -> float:
    gt = set(ground_truth)
    for rank, mid in enumerate(top_k, start=1):
        if mid in gt:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(top_k: list[int], ground_truth: list[int]) -> float:
    gt = set(ground_truth)
    dcg = sum(1.0 / np.log2(r + 2) for r, m in enumerate(top_k) if m in gt)
    ideal = min(len(ground_truth), K)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def language_diversity(top_k: list[int], movie_index: dict, language: list[str]) -> int:
    return len({language[movie_index[m]] for m in top_k if m in movie_index})


def genre_diversity(
    top_k: list[int], movie_index: dict, clean_genres: list[str]
) -> int:
    genres: set[str] = set()
    for mid in top_k:
        idx = movie_index.get(mid)
        if idx is not None:
            genres.update(g for g in str(clean_genres[idx]).split("|") if g)
    return len(genres)


def filter_bubble_score(
    top_k: list[int],
    pref_genres: list[str],
    pref_languages: list[str],
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
) -> float:
    """Fraction of the top-K matching BOTH preferred language and genre."""
    pref_lang_set, pref_genre_set = set(pref_languages), set(pref_genres)
    matches = 0
    for mid in top_k:
        idx = movie_index.get(mid)
        if idx is None:
            continue
        m_genres = {g for g in str(clean_genres[idx]).split("|") if g}
        if language[idx] in pref_lang_set and (m_genres & pref_genre_set):
            matches += 1
    return matches / K


def compute_metrics(
    top_k: list[int],
    ground_truth: list[int],
    pref_genres: list[str],
    pref_languages: list[str],
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
) -> dict:
    bubble = round(
        filter_bubble_score(
            top_k, pref_genres, pref_languages, movie_index, clean_genres, language
        ),
        4,
    )
    return {
        "Precision@10": round(precision_at_k(top_k, ground_truth), 4),
        "Precision@10_Ceiling": round(precision_ceiling(ground_truth), 4),
        "Recall@10": round(recall_at_k(top_k, ground_truth), 4),
        "HR@10": round(hit_rate_at_k(top_k, ground_truth), 4),
        "MRR@10": round(mrr_at_k(top_k, ground_truth), 4),
        "NDCG@10": round(ndcg_at_k(top_k, ground_truth), 4),
        "Language_Diversity": language_diversity(top_k, movie_index, language),
        "Genre_Diversity": genre_diversity(top_k, movie_index, clean_genres),
        "Filter_Bubble_Score": bubble,
        "Novelty@10": round(1.0 - bubble, 4),
    }


def run_evaluation(track: str, shared: tuple) -> pd.DataFrame:
    (svd, cbf_matrix, movie_ids, movie_index, clean_genres, language, quality) = shared

    ratings_df, users_df = load_track(track)
    splits = build_holdout_split(ratings_df)
    logger.info("Track '%s': %d users with usable splits.", track, len(splits))

    profiles = users_df.set_index("userId").to_dict("index")
    pop_bins = build_popularity_bins(quality)

    rng = np.random.default_rng(RANDOM_SEED)
    eval_uids = sorted(splits)
    if len(eval_uids) > MAX_EVAL_USERS:
        eval_uids = sorted(
            int(u) for u in rng.choice(eval_uids, size=MAX_EVAL_USERS, replace=False)
        )

    rows = []
    for i, uid in enumerate(eval_uids):
        if (i + 1) % 50 == 0:
            logger.info("  [%s] %d / %d users", track, i + 1, len(eval_uids))

        train_ids, train_pos, holdout_pos = splits[uid]
        profile = profiles.get(uid, {})
        pref_genres = _parse_pipe_list(profile.get("preferred_genres", ""))
        pref_languages = _parse_pipe_list(profile.get("preferred_language", ""))
        archetype = str(profile.get("archetype", "unknown"))

        candidates = build_candidate_set(
            train_ids, holdout_pos, movie_ids, movie_index, pop_bins, uid
        )
        if not candidates:
            continue

        model_scores = score_all_models(
            train_pos,
            candidates,
            pref_genres,
            pref_languages,
            svd,
            cbf_matrix,
            movie_index,
            clean_genres,
            language,
            quality,
        )

        for name in MODEL_ORDER:
            top_k = [
                mid
                for mid, _ in sorted(
                    model_scores[name], key=lambda x: x[1], reverse=True
                )[:K]
            ]
            rows.append(
                {
                    "Track": track,
                    "UserId": uid,
                    "Archetype": archetype,
                    "Model": name,
                    "Train_Count": len(train_ids),
                    "Holdout_Pos": len(holdout_pos),
                    "Candidate_Count": len(candidates),
                    **compute_metrics(
                        top_k,
                        holdout_pos,
                        pref_genres,
                        pref_languages,
                        movie_index,
                        clean_genres,
                        language,
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def report_rq1(df: pd.DataFrame, track: str) -> None:
    print_section(f"[{track}] TABLE 1: MODEL PERFORMANCE (RQ1)")

    metrics = ["Precision@10", "Recall@10", "HR@10", "MRR@10", "NDCG@10"]
    summary = agg_frame(df, "Model", metrics, ["mean", "std"]).reindex(MODEL_ORDER)
    summary = summary.reset_index()

    ceiling = mean_series(df, "Model", "Precision@10_Ceiling")
    summary["Precision@10_Ceiling_mean"] = summary["Model"].map(ceiling)

    print(summary.to_string(index=False))
    print(
        f"\n  NOTE: mean attainable Precision@10 is {ceiling.iloc[0]:.4f}, "
        "not 1.0, because most users have fewer than 10 holdout positives.\n"
        "  HR@10 and MRR@10 are not subject to this ceiling."
    )
    summary.to_csv(f"{RESULTS_DIR}/rq1_model_performance_{track}.csv", index=False)


def report_significance(df: pd.DataFrame, track: str) -> None:
    print_section(f"[{track}] TABLE 2: SIGNIFICANCE ON NDCG@10 (H1)")

    rows = []
    for a, b in combinations(MODEL_ORDER, 2):
        paired = (
            df[df["Model"].isin([a, b])]
            .pivot(index="UserId", columns="Model", values="NDCG@10")
            .dropna()
        )
        if paired.empty:
            continue

        xa = paired[a].to_numpy(float)
        xb = paired[b].to_numpy(float)
        diff = xa - xb
        n = len(paired)

        t_stat = p_t = d = float("nan")
        if n > 1 and diff.std(ddof=1) > 0:
            t_stat, p_t = stats.ttest_rel(xa, xb) # pyright: ignore[reportAttributeAccessIssue]
            d = float(diff.mean() / diff.std(ddof=1))

        p_w = float("nan")
        if np.any(diff != 0):
            with contextlib.suppress(ValueError):
                _, p_w = stats.wilcoxon(xa, xb, zero_method="wilcox") # pyright: ignore[reportAttributeAccessIssue]

        rows.append(
            {
                "Model_A": a,
                "Model_B": b,
                "N": n,
                "Mean_A": round(float(xa.mean()), 4),
                "Mean_B": round(float(xb.mean()), 4),
                "t_statistic": round(float(t_stat), 4),
                "p_ttest": float(p_t),
                "p_wilcoxon": float(p_w),
                "Cohens_d": round(float(d), 4),
                "B_wins": int((xb > xa).sum()),
                "Ties": int((xb == xa).sum()),
                "B_losses": int((xb < xa).sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return

    out["p_ttest_holm"] = holm_bonferroni(out["p_ttest"].fillna(1.0).tolist())
    out["p_wilcoxon_holm"] = holm_bonferroni(out["p_wilcoxon"].fillna(1.0).tolist())
    out["Significant_holm"] = np.where(out["p_ttest_holm"] < ALPHA, "yes", "no")

    print(out.to_string(index=False))
    print(
        "\n  Win/tie/loss counts are reported because a mean improvement can be "
        "driven\n  by a minority of users while most are unchanged."
    )
    out.to_csv(f"{RESULTS_DIR}/rq1_significance_tests_{track}.csv", index=False)


def report_confidence_intervals(df: pd.DataFrame, track: str) -> None:
    print_section(f"[{track}] TABLE 3: 95% CONFIDENCE INTERVALS")

    rows = []
    for metric in ["Precision@10", "Recall@10", "HR@10", "MRR@10", "NDCG@10"]:
        for model in MODEL_ORDER:
            vals = df.loc[df["Model"] == model, metric].to_numpy(float)
            if len(vals) < 2:
                continue
            mean = float(vals.mean())
            sem = float(stats.sem(vals)) # pyright: ignore[reportAttributeAccessIssue]
            lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem) # type: ignore
            rows.append(
                {
                    "Metric": metric,
                    "Model": model,
                    "Mean": round(mean, 4),
                    "CI_lower": round(float(lo), 4),
                    "CI_upper": round(float(hi), 4),
                }
            )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(f"{RESULTS_DIR}/rq1_confidence_intervals_{track}.csv", index=False)


def report_rq2_diversity(df: pd.DataFrame, track: str) -> None:
    print_section(f"[{track}] TABLE 4: DIVERSITY AND FILTER BUBBLE (RQ2 / H2)")

    columns = [
        "Language_Diversity",
        "Genre_Diversity",
        "Filter_Bubble_Score",
        "Novelty@10",
    ]
    out = mean_frame(df, "Model", columns, decimals=3).reindex(MODEL_ORDER)
    out = out.reset_index()
    print(out.to_string(index=False))
    out.to_csv(f"{RESULTS_DIR}/rq2_diversity_by_model_{track}.csv", index=False)


def report_by_archetype(df: pd.DataFrame, track: str) -> None:
    print_section(f"[{track}] TABLE 5: PERFORMANCE AND BUBBLE BY ARCHETYPE")

    perf = mean_frame(
        df, ["Archetype", "Model"], ["Precision@10", "HR@10", "NDCG@10"]
    ).reset_index()
    print(perf.to_string(index=False))
    perf.to_csv(f"{RESULTS_DIR}/rq2_performance_by_archetype_{track}.csv", index=False)

    bubble = pivot_mean(
        df, index="Model", columns="Archetype", values="Filter_Bubble_Score"
    )
    bubble = bubble.reindex(MODEL_ORDER).reset_index()
    print("\n" + bubble.to_string(index=False))
    bubble.to_csv(
        f"{RESULTS_DIR}/rq2_filter_bubble_by_archetype_{track}.csv", index=False
    )


def main() -> None:
    shared = load_shared_artifacts()

    all_frames = []
    for track in ("real", "synthetic"):
        cfg = TRACKS[track]
        if not os.path.exists(cfg["ratings"]):
            logger.warning(
                "Track '%s' artifacts missing (%s); skipping.", track, cfg["ratings"]
            )
            continue

        print_section(f"EVALUATION TRACK: {cfg['label']}")
        df = run_evaluation(track, shared)
        if df.empty:
            logger.warning("Track '%s' produced no rows.", track)
            continue

        report_rq1(df, track)
        report_significance(df, track)
        report_confidence_intervals(df, track)
        report_rq2_diversity(df, track)
        report_by_archetype(df, track)
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No evaluation track produced results.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(f"{RESULTS_DIR}/evaluation_user_level.csv", index=False)
    logger.info("Wrote %s/evaluation_user_level.csv", RESULTS_DIR)


if __name__ == "__main__":
    main()
