from __future__ import annotations

import logging
import os
import warnings
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from surprise import dump

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SVD_MODEL_PATH = "data/processed/svd_model.pkl"
SYNTH_RATINGS = "data/processed/synthetic_ratings.csv"
SYNTH_USERS = "data/processed/synthetic_users.csv"
CBF_MATRIX_PATH = "data/processed/cbf_matrix.pkl"
CBF_META_PATH = "data/processed/cbf_metadata.pkl"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

K = 10
MAX_EVAL_USERS = 400
MIN_HOLDOUT_POS = 1  
HOLDOUT_RATING_THRESHOLD = 3.5
RANDOM_SEED = 42
N_NEGATIVE_CANDIDATES = 1_000

MODEL_ORDER = ["CF_ColdStart", "CBF", "NonLocal_Hybrid", "Localized_Hybrid"]

STANDARD_CF_W = 0.625
STANDARD_CBF_W = 0.375
LOCAL_CF_W = 0.50
LOCAL_CBF_W = 0.30
LOCAL_LOC_W = 0.20

LOC_GENRE_W = 0.60
LOC_LANG_W = 0.40

LANGUAGE_PREF_SCORE: dict[str, float] = {
    "English": 1.00,
    "Hindi": 0.90,
    "Japanese": 0.85,
    "Korean": 0.80,
    "Nepali": 0.75,
}

GENRE_LOC_WEIGHT: dict[str, float] = {
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


def _check(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def load_all_artifacts() -> tuple:

    logger.info("Loading SVD model …")
    _check(SVD_MODEL_PATH)
    _, svd = dump.load(SVD_MODEL_PATH)

    logger.info("Loading synthetic cohort …")
    _check(SYNTH_RATINGS)
    _check(SYNTH_USERS)
    ratings_df = pd.read_csv(SYNTH_RATINGS)
    users_df = pd.read_csv(SYNTH_USERS)

    logger.info("Loading CBF matrix and metadata …")
    _check(CBF_MATRIX_PATH)
    _check(CBF_META_PATH)
    cbf_matrix = joblib.load(CBF_MATRIX_PATH)
    cbf_meta = joblib.load(CBF_META_PATH)

    movie_ids = cbf_meta["movie_ids"]
    movie_index = cbf_meta["movie_index"]
    clean_genres = cbf_meta["clean_genres"]
    language = cbf_meta["language"]
    logger.info(
        "Artifacts loaded: %d CBF movies, %d synthetic ratings, %d synthetic users.",
        len(movie_ids),
        len(ratings_df),
        len(users_df),
    )
    return (
        svd,
        ratings_df,
        users_df,
        cbf_matrix,
        movie_ids,
        movie_index,
        clean_genres,
        language,
    )


def _parse_pipe_list(s: str) -> list[str]:
    return [x.strip() for x in s.split("|") if x.strip()]


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
        if len(holdout_pos) >= MIN_HOLDOUT_POS:
            splits[int(uid)] = (train_ids, train_pos, holdout_pos)  # type: ignore
    return splits


def build_candidate_set(
    train_ids: list[int],
    holdout_pos: list[int],
    movie_ids: np.ndarray,
    user_id: int,
) -> list[int]:
    
    excluded = set(train_ids) | set(holdout_pos)
    negative_pool = np.array([mid for mid in movie_ids if mid not in excluded])
    n_negatives = min(N_NEGATIVE_CANDIDATES, len(negative_pool))
    rng = np.random.default_rng(RANDOM_SEED + user_id)
    negatives = rng.choice(negative_pool, size=n_negatives, replace=False).tolist()
    return sorted([*holdout_pos, *negatives])


def _cold_start_cf_scores(
    svd,
    train_pos: list[int],
    candidates: list[int],
) -> dict[int, float]:
    
    trainset = svd.trainset
    global_mean = float(trainset.global_mean)

    qi_liked: list[np.ndarray] = []
    for mid in train_pos:
        try:
            inner_iid = trainset.to_inner_iid(mid)
            qi_liked.append(svd.qi[inner_iid])
        except ValueError:
            pass  

    p_hat: np.ndarray = (
        np.mean(qi_liked, axis=0) if qi_liked else np.zeros(svd.qi.shape[1])
    )

    raw: dict[int, float] = {}
    for mid in candidates:
        try:
            inner_iid = trainset.to_inner_iid(mid)
            raw[mid] = (
                global_mean + svd.bi[inner_iid] + float(svd.qi[inner_iid] @ p_hat)
            )
        except ValueError:
            raw[mid] = global_mean  

    vals = np.array([raw[m] for m in candidates], dtype=np.float32)
    lo, hi = vals.min(), vals.max()
    if hi - lo > 1e-8:
        vals = (vals - lo) / (hi - lo)
    return {m: float(v) for m, v in zip(candidates, vals)}


def compute_localization_score(
    movie_ids_cands: list[int],
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
    pref_genres: list[str],
    pref_languages: list[str],
) -> dict[int, float]:

    pref_lang_set = set(pref_languages)
    scores: dict[int, float] = {}
    for mid in movie_ids_cands:
        idx = movie_index.get(mid)
        if idx is None:
            scores[mid] = 0.0
            continue

        movie_genre_list = [g for g in str(clean_genres[idx]).split("|") if g]
        genre_score = 0.0
        if pref_genres and movie_genre_list:
            matched = [
                GENRE_LOC_WEIGHT.get(g, 0.0)
                for g in movie_genre_list
                if g in pref_genres
            ]
            genre_score = float(np.mean(matched)) if matched else 0.0

        m_lang = language[idx]
        lang_score = (
            LANGUAGE_PREF_SCORE.get(m_lang, 0.0) if m_lang in pref_lang_set else 0.0
        )
        scores[mid] = LOC_GENRE_W * genre_score + LOC_LANG_W * lang_score
    return scores


def score_all_models(
    uid: int,  
    train_pos: list[int],
    candidates: list[int],
    pref_genres: list[str],
    pref_languages: list[str],
    svd,
    cbf_matrix,
    movie_index: dict,
    clean_genres: list[str],
    language: list[str],
) -> dict[str, list[tuple[int, float]]]:

    valid_train_idxs = [movie_index[m] for m in train_pos if m in movie_index]
    if valid_train_idxs:
        from sklearn.metrics.pairwise import linear_kernel

        raw_sims = linear_kernel(cbf_matrix[valid_train_idxs], cbf_matrix).mean(axis=0)
        lo, hi = raw_sims.min(), raw_sims.max()
        norm_sims = (
            (raw_sims - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(raw_sims)
        )
        cbf_arr = np.asarray(norm_sims, dtype=np.float32).ravel()
    else:
        cbf_arr = np.zeros(len(movie_index), dtype=np.float32)

    cf_norm = _cold_start_cf_scores(svd, train_pos, candidates)

    loc_scores = compute_localization_score(
        candidates, movie_index, clean_genres, language, pref_genres, pref_languages
    )

    loc_vals = np.array([loc_scores[m] for m in candidates], dtype=np.float32)
    lo, hi = loc_vals.min(), loc_vals.max()
    if hi - lo > 1e-8:
        loc_vals = (loc_vals - lo) / (hi - lo)
    loc_norm = {m: float(v) for m, v in zip(candidates, loc_vals)}

    model_scores: dict[str, list[tuple[int, float]]] = {
        name: [] for name in MODEL_ORDER
    }

    for mid in candidates:
        idx = movie_index.get(mid)
        if idx is None:
            continue

        cf_pred = cf_norm.get(mid, 0.0)
        cbf_s = float(cbf_arr[idx])
        loc_s = loc_norm.get(mid, 0.0)

        model_scores["CF_ColdStart"].append((mid, cf_pred))
        model_scores["CBF"].append((mid, cbf_s))
        model_scores["NonLocal_Hybrid"].append(
            (mid, STANDARD_CF_W * cf_pred + STANDARD_CBF_W * cbf_s)
        )
        model_scores["Localized_Hybrid"].append(
            (mid, LOCAL_CF_W * cf_pred + LOCAL_CBF_W * cbf_s + LOCAL_LOC_W * loc_s)
        )

    return model_scores


def precision_at_k(top_k: list[int], ground_truth: list[int]) -> float:

    hits = len(set(top_k) & set(ground_truth))
    return hits / K


def recall_at_k(top_k: list[int], ground_truth: list[int]) -> float:

    if not ground_truth:
        return 0.0
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth)


def ndcg_at_k(top_k: list[int], ground_truth: list[int]) -> float:

    gt_set = set(ground_truth)
    dcg = sum(
        1.0 / np.log2(rank + 2) for rank, mid in enumerate(top_k) if mid in gt_set
    )
    ideal_hits = min(len(ground_truth), K)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def language_diversity(top_k: list[int], movie_index: dict, language: list[str]) -> int:

    langs = {language[movie_index[m]] for m in top_k if m in movie_index}
    return len(langs)


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
    
    pref_lang_set = set(pref_languages)
    pref_genre_set = set(pref_genres)
    matches = 0
    for mid in top_k:
        idx = movie_index.get(mid)
        if idx is None:
            continue
        m_lang = language[idx]
        m_genres = {g for g in str(clean_genres[idx]).split("|") if g}

        if m_lang in pref_lang_set and bool(m_genres & pref_genre_set):
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
        # RQ1
        "Precision@10": round(precision_at_k(top_k, ground_truth), 4),
        "Recall@10": round(recall_at_k(top_k, ground_truth), 4),
        "NDCG@10": round(ndcg_at_k(top_k, ground_truth), 4),
        # RQ2
        "Language_Diversity": language_diversity(top_k, movie_index, language),
        "Genre_Diversity": genre_diversity(top_k, movie_index, clean_genres),
        "Filter_Bubble_Score": bubble,
        # Novelty@10: complement of Filter_Bubble_Score.
        # Fraction of recommendations *outside* the user's preference bubble.
        # Higher Novelty@10 → more serendipitous / diverse recommendations.
        "Novelty@10": round(1.0 - bubble, 4),
    }


def run_evaluation() -> pd.DataFrame:

    (
        svd,
        ratings_df,
        users_df,
        cbf_matrix,
        movie_ids,
        movie_index,
        clean_genres,
        language,
    ) = load_all_artifacts()

    splits = build_holdout_split(ratings_df)
    logger.info("Users with ≥%d holdout positive: %d", MIN_HOLDOUT_POS, len(splits))

    users_df["userId"] = users_df["userId"].astype(int)
    user_profile = users_df.set_index("userId").to_dict("index")

    rng = np.random.default_rng(RANDOM_SEED)
    eval_uids = sorted(splits.keys())
    if len(eval_uids) > MAX_EVAL_USERS:
        eval_uids = sorted(
            rng.choice(eval_uids, size=MAX_EVAL_USERS, replace=False).tolist()
        )
    logger.info("Evaluating %d users …", len(eval_uids))

    rows = []
    for i, uid in enumerate(eval_uids):
        if (i + 1) % 50 == 0:
            logger.info("  … %d / %d users done", i + 1, len(eval_uids))

        train_ids, train_pos, holdout_pos = splits[uid]
        profile = user_profile.get(uid, {})
        pref_genres = _parse_pipe_list(profile.get("preferred_genres", ""))
        pref_languages = _parse_pipe_list(profile.get("preferred_language", ""))
        archetype = str(profile.get("archetype", "unknown"))

        candidates = build_candidate_set(
            train_ids,
            holdout_pos,
            movie_ids,
            uid,
        )
        if not candidates:
            logger.warning("uid=%d: empty candidate set, skipping.", uid)
            continue

        model_scores = score_all_models(
            uid,
            train_pos,
            candidates,
            pref_genres,
            pref_languages,
            svd,
            cbf_matrix,
            movie_index,
            clean_genres,
            language,
        )

        for model_name in MODEL_ORDER:
            top_k = [
                mid
                for mid, _ in sorted(
                    model_scores[model_name], key=lambda x: x[1], reverse=True
                )[:K]
            ]
            metrics = compute_metrics(
                top_k,
                holdout_pos,
                pref_genres,
                pref_languages,
                movie_index,
                clean_genres,
                language,
            )
            rows.append(
                {
                    "UserId": uid,
                    "Archetype": archetype,
                    "Model": model_name,
                    "Train_Count": len(train_ids),
                    "Holdout_Pos": len(holdout_pos),
                    "Candidate_Count": len(candidates),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def report_rq1(df: pd.DataFrame) -> None:

    print_section("TABLE 1: MODEL PERFORMANCE (RQ1 — Precision, Recall, NDCG)")

    summary = (
        df.groupby("Model")[["Precision@10", "Recall@10", "NDCG@10"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reindex(MODEL_ORDER).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(f"{RESULTS_DIR}/rq1_model_performance.csv", index=False)


def report_significance(df: pd.DataFrame) -> None:

    print_section("TABLE 2: STATISTICAL SIGNIFICANCE — Paired t-test on NDCG@10 (H1)")
    print("  α = 0.05   |   Cohen's d = effect size")
    print(
        f"  {'Model A':<20} {'Model B':<20} {'N':>4} "
        f"{'Mean A':>8} {'Mean B':>8} {'t':>8} {'p':>10} {'Cohen d':>9} {'Sig?':>5}"
    )
    print("  " + "-" * 95)

    sig_rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        paired = (
            df[df["Model"].isin([model_a, model_b])]
            .pivot(index="UserId", columns="Model", values="NDCG@10")
            .dropna()
        )

        a_scores = paired[model_a].to_numpy(dtype=float)
        b_scores = paired[model_b].to_numpy(dtype=float)
        diff = a_scores - b_scores
        n = len(paired)
        mean_a = a_scores.mean()
        mean_b = b_scores.mean()

        t_stat = p_val = cohen_d = float("nan")
        if n > 1 and diff.std(ddof=1) > 0:
            t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
            cohen_d = diff.mean() / diff.std(ddof=1)

        sig = "✓" if (not np.isnan(p_val) and p_val < 0.05) else "✗"
        p_display = f"{p_val:.3g}" if not np.isnan(p_val) else "N/A"
        print(
            f"  {model_a:<20} {model_b:<20} {n:>4} "
            f"{mean_a:>8.4f} {mean_b:>8.4f} "
            f"{t_stat:>8.3f} {p_display:>10} {cohen_d:>9.3f} {sig:>5}"
        )
        sig_rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "N": n,
                "Mean_A": round(mean_a, 4),
                "Mean_B": round(mean_b, 4),
                "t_statistic": round(t_stat, 4) if not np.isnan(t_stat) else "",
                "p_value": p_val if not np.isnan(p_val) else "",
                "Cohens_d": round(cohen_d, 4) if not np.isnan(cohen_d) else "",
                "Significant_p<0.05": sig,
            }
        )

    pd.DataFrame(sig_rows).to_csv(
        f"{RESULTS_DIR}/rq1_significance_tests.csv", index=False
    )


def report_confidence_intervals(df: pd.DataFrame) -> None:

    print_section("TABLE 3: 95% CONFIDENCE INTERVALS — NDCG@10")
    ci_rows = []
    for model in MODEL_ORDER:
        scores = df[df["Model"] == model]["NDCG@10"].to_numpy(dtype=float)
        mean = scores.mean()
        if len(scores) > 1:
            ci = stats.t.interval(
                0.95, df=len(scores) - 1, loc=mean, scale=stats.sem(scores)
            )
            print(f"  {model:<22}: {mean:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            ci_rows.append(
                {
                    "Model": model,
                    "Mean_NDCG@10": round(mean, 4),
                    "CI_lower": round(ci[0], 4),
                    "CI_upper": round(ci[1], 4),
                }
            )
        else:
            print(f"  {model:<22}: {mean:.4f}  95% CI [N/A]")
            ci_rows.append({"Model": model, "Mean_NDCG@10": round(mean, 4)})
    pd.DataFrame(ci_rows).to_csv(
        f"{RESULTS_DIR}/rq1_confidence_intervals.csv", index=False
    )


def report_rq2_diversity(df: pd.DataFrame) -> None:

    print_section("TABLE 4: DIVERSITY & NOVELTY METRICS BY MODEL (RQ2)")
    div = (
        df.groupby("Model")[["Language_Diversity", "Genre_Diversity", "Novelty@10"]]
        .mean()
        .round(3)
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    print(div.to_string(index=False))
    div.to_csv(f"{RESULTS_DIR}/rq2_diversity_by_model.csv", index=False)


def report_rq2_filter_bubble(df: pd.DataFrame) -> None:

    print_section(
        "TABLE 5: FILTER BUBBLE SCORE — Model × Archetype (RQ2 / H2)\n"
        "  Higher = more preference-locked (stronger bubble effect)"
    )
    bubble = (
        df.groupby(["Model", "Archetype"])["Filter_Bubble_Score"]
        .mean()
        .unstack("Archetype")
        .round(3)
        .reindex(MODEL_ORDER)
    )
    print(bubble.to_string())
    bubble.reset_index().to_csv(
        f"{RESULTS_DIR}/rq2_filter_bubble_by_archetype.csv", index=False
    )
    print(
        "\n  Interpretation for Hypothesis 2:\n"
        "  • Localized_Hybrid scoring highest → preference-reinforcing effect.\n"
        "  • Compare Language_Diversity / Genre_Diversity across models (Table 4).\n"
        "  • If diversity drops alongside bubble rise, H2 is supported."
    )


def report_fairness(df: pd.DataFrame) -> None:

    print_section("TABLE 6: PERFORMANCE BY ARCHETYPE (Fairness / RQ2)")
    arch_perf = (
        df.groupby(["Archetype", "Model"])[["Precision@10", "NDCG@10"]]
        .mean()
        .round(4)
        .reset_index()
    )
    print(arch_perf.to_string(index=False))
    arch_perf.to_csv(f"{RESULTS_DIR}/rq2_performance_by_archetype.csv", index=False)

    print("\n  Fairness Gap (max − min NDCG@10 across archetypes, per model):")
    for model in MODEL_ORDER:
        sub = arch_perf[arch_perf["Model"] == model]
        gap = sub["NDCG@10"].max() - sub["NDCG@10"].min()
        best = sub.loc[sub["NDCG@10"].idxmax(), "Archetype"]
        worst = sub.loc[sub["NDCG@10"].idxmin(), "Archetype"]
        print(f"  {model:<22}: gap={gap:.4f}  best={best}  worst={worst}")


def main() -> None:
    print_section("STARTING EVALUATION")

    results_df = run_evaluation()
    if results_df.empty:
        print("No results generated. Check data alignment.")
        return

    user_path = f"{RESULTS_DIR}/evaluation_user_level.csv"
    results_df.to_csv(user_path, index=False)
    logger.info("User-level results saved → %s  (%d rows)", user_path, len(results_df))

    report_rq1(results_df)
    report_significance(results_df)
    report_confidence_intervals(results_df)
    report_rq2_diversity(results_df)
    report_rq2_filter_bubble(results_df)
    report_fairness(results_df)

    print_section("EVALUATION COMPLETE")
    print(f"  All CSV files written to: {RESULTS_DIR}/")
    print("  Files produced:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".csv"):
            print(f"    • {f}")


if __name__ == "__main__":
    main()
