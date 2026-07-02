import os
import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from surprise import dump

warnings.filterwarnings("ignore")

SVD_MODEL = "data/processed/svd_model.pkl"
SYNTH_RATINGS = "data/processed/synthetic_interactions.csv"
SYNTH_PROFILES = "data/processed/synthetic_user_profiles.csv"
CBF_MATRIX = "data/processed/cbf_matrix.pkl"
CBF_META = "data/processed/cbf_metadata.pkl"
MOVIES_FINAL = "data/processed/movies_final.csv"
RESULTS_DIR = "results"

K = 10
MAX_EVAL_USERS = 150
SUPPORT_RATIO = 0.7
RANDOM_SEED = 42
MODEL_ORDER = ["CF", "CBF", "NonLocal_Hybrid", "Localized_Hybrid"]

os.makedirs(RESULTS_DIR, exist_ok=True)

GENRE_MAPPING = {
    "Sci-Fi": "ScienceFiction",
    "Science Fiction": "ScienceFiction",
    "Anime": "Animation",
    "Children's": "Family",
    "Childrens": "Family",
    "Bollywood": "Drama",
    "Tollywood": "Action",
}
LANGUAGE_MAPPING = {
    "HINDI": "HI",
    "ENGLISH": "EN",
    "NEPALI": "NE",
    "JAPANESE": "JA",
    "KOREAN": "KO",
    "FRENCH": "FR",
    "SPANISH": "ES",
    "GERMAN": "DE",
    "CHINESE": "ZH",
}


def normalize_prefs(profile):
    langs_raw = str(profile.get("preferred_languages", "")).split("|")
    genres_raw = str(profile.get("preferred_genres", "")).split("|")

    langs = {
        LANGUAGE_MAPPING.get(lang.strip().upper(), lang.strip().upper())
        for lang in langs_raw
        if lang.strip()
    }
    genres = {
        GENRE_MAPPING.get(genre.strip(), genre.strip().replace(" ", ""))
        for genre in genres_raw
        if genre.strip()
    }
    return langs, genres


def get_matrix_idx(cbf_indices, movie_id):
    idx_val = cbf_indices.get(movie_id)
    if idx_val is None:
        return None
    if isinstance(idx_val, pd.Series):
        return int(idx_val.iloc[0])
    return int(idx_val)


def split_support_holdout(user_rows):
    rows = user_rows.sort_values(["rating", "movieId"], ascending=[False, True])
    positives = rows[rows["rating"] >= 4.0]

    if len(positives) < 2:
        return [], []

    holdout_count = max(1, int(round(len(positives) * (1 - SUPPORT_RATIO))))
    holdout = positives.tail(holdout_count)
    support = rows.drop(index=holdout.index)

    support_history = support["movieId"].astype(int).tolist()
    ground_truth = holdout["movieId"].astype(int).tolist()
    return support_history, ground_truth


def preference_score(movie_row, pref_langs, pref_genres):
    score = 0.0
    m_lang = str(movie_row.get("language", "")).upper()
    m_genres = set(str(movie_row.get("clean_genres", "")).split())

    if m_lang in pref_langs:
        score += 0.6
    if not m_genres.isdisjoint(pref_genres):
        score += 0.4
    return score


def adaptive_weights(history_len):
    if history_len == 0:
        return {"cf": 0.0, "cbf": 0.7, "pref": 0.3}
    if history_len < 5:
        return {"cf": 0.2, "cbf": 0.5, "pref": 0.3}
    if history_len < 15:
        return {"cf": 0.4, "cbf": 0.4, "pref": 0.2}
    return {"cf": 0.6, "cbf": 0.2, "pref": 0.2}


def build_candidates(
    valid_history,
    support_history,
    ground_truth,
    cosine_sim,
    cbf_meta,
    cbf_indices,
    movies_meta,
):
    candidates = set()

    if valid_history:
        safe_idxs = [
            get_matrix_idx(cbf_indices, movie_id) for movie_id in valid_history
        ]
        safe_idxs = [idx for idx in safe_idxs if idx is not None]

        if safe_idxs:
            sim_scores = np.mean(cosine_sim[np.unique(safe_idxs), :], axis=0)
            top_count = min(500, len(sim_scores))
            top_cbf_idx = np.argsort(sim_scores)[-top_count:]
            candidates.update(
                cbf_meta.iloc[top_cbf_idx]["movieId"].astype(int).tolist()
            )

    candidates.update(
        movies_meta.nlargest(500, "vote_count")["movieId"].astype(int).tolist()
    )
    # Rank held-out positives among realistic negatives. The held-out ratings
    # are not used for scoring; they only define relevance for metrics.
    candidates.update(ground_truth)
    return list(candidates - set(support_history))


def calculate_metrics(top_k, ground_truth, cbf_meta, pref_langs, pref_genres):
    hits = len(set(top_k).intersection(set(ground_truth)))
    precision = hits / K
    recall = hits / len(ground_truth) if ground_truth else 0

    relevance = [1 if movie_id in ground_truth else 0 for movie_id in top_k]
    dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))
    ideal_relevance = sorted([1] * min(len(ground_truth), K), reverse=True)
    idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_relevance))
    ndcg = dcg / idcg if idcg > 0 else 0

    top_k_meta = cbf_meta[cbf_meta["movieId"].isin(top_k)]
    unique_langs = top_k_meta["language"].nunique() if len(top_k_meta) > 0 else 0
    all_genres = " ".join(top_k_meta["clean_genres"].fillna("").astype(str)).split()
    unique_genres = len(set(all_genres))

    bubble_matches = 0
    novelty_matches = 0
    for movie_id in top_k:
        rows = top_k_meta[top_k_meta["movieId"] == movie_id]
        if rows.empty:
            novelty_matches += 1
            continue

        row = rows.iloc[0]
        movie_lang = str(row.get("language", "")).upper()
        movie_genres = set(str(row.get("clean_genres", "")).split())
        matches_pref = movie_lang in pref_langs or not movie_genres.isdisjoint(
            pref_genres
        )

        if matches_pref:
            bubble_matches += 1
        else:
            novelty_matches += 1

    return {
        "Precision@10": round(precision, 4),
        "Recall@10": round(recall, 4),
        "NDCG@10": round(ndcg, 4),
        "Novelty@10": round(novelty_matches / K, 4),
        "Language_Diversity": unique_langs,
        "Genre_Diversity": unique_genres,
        "Filter_Bubble_Score": round(bubble_matches / K, 4),
        "Hits@10": hits,
    }


def main():
    print("Loading real-only SVD model and evaluation artifacts...")
    if not os.path.exists(SVD_MODEL):
        raise FileNotFoundError(
            "Missing real-only SVD model. Run 'uv run python engine/collaborative_filter.py' first."
        )

    _, svd = dump.load(SVD_MODEL)
    synth_df = pd.read_csv(SYNTH_RATINGS)
    profiles_df = pd.read_csv(SYNTH_PROFILES)
    movies_meta = pd.read_csv(MOVIES_FINAL)

    synth_df["userId"] = synth_df["userId"].astype(str)
    profiles_df["user_id"] = profiles_df["user_id"].astype(str)

    print("Loading CBF artifacts...")
    with open(CBF_MATRIX, "rb") as f:
        cosine_sim = pickle.load(f)
    with open(CBF_META, "rb") as f:
        cbf_meta = pickle.load(f)

    cbf_indices = pd.Series(cbf_meta.index, index=cbf_meta["movieId"])
    cbf_indices = cbf_indices[~cbf_indices.index.duplicated(keep="first")]

    print("Running synthetic support/holdout evaluation.")
    print("SVD source: real MovieLens ratings only.")
    print(
        "Synthetic cohort source: evaluation support history + held-out positives only."
    )

    rng = np.random.default_rng(RANDOM_SEED)
    candidate_users = []
    user_splits = {}

    for uid, user_rows in synth_df.groupby("userId"):
        support_history, ground_truth = split_support_holdout(user_rows)
        if len(support_history) == 0 or len(ground_truth) == 0:
            continue
        candidate_users.append(uid)
        user_splits[uid] = (support_history, ground_truth)

    eval_users = sorted(candidate_users)
    if len(eval_users) > MAX_EVAL_USERS:
        eval_users = sorted(rng.choice(eval_users, size=MAX_EVAL_USERS, replace=False))

    results = []

    for uid in eval_users:
        profile_match = profiles_df[profiles_df["user_id"] == str(uid)]
        if profile_match.empty:
            continue

        profile = profile_match.iloc[0]
        pref_langs, pref_genres = normalize_prefs(profile)
        support_history, ground_truth = user_splits[uid]
        valid_history = [
            movie_id for movie_id in support_history if movie_id in cbf_indices.index
        ]

        candidates = build_candidates(
            valid_history,
            support_history,
            ground_truth,
            cosine_sim,
            cbf_meta,
            cbf_indices,
            movies_meta,
        )
        if not candidates:
            continue

        model_scores = {model: [] for model in MODEL_ORDER}
        weights = adaptive_weights(len(valid_history))

        for mid in candidates:
            matrix_idx = get_matrix_idx(cbf_indices, mid)
            if matrix_idx is None:
                continue

            movie_row = cbf_meta.loc[matrix_idx]
            cf_pred = svd.predict(str(uid), mid).est / 5.0

            cbf_score = 0.0
            if valid_history:
                history_idxs = [get_matrix_idx(cbf_indices, h) for h in valid_history]
                history_idxs = [idx for idx in history_idxs if idx is not None]
                if history_idxs:
                    cbf_score = float(
                        np.mean(
                            [cosine_sim[h_idx][matrix_idx] for h_idx in history_idxs]
                        )
                    )

            pref_score = preference_score(movie_row, pref_langs, pref_genres)

            model_scores["CF"].append((mid, cf_pred))
            model_scores["CBF"].append((mid, cbf_score))
            model_scores["NonLocal_Hybrid"].append(
                (mid, 0.625 * cf_pred + 0.375 * cbf_score)
            )
            model_scores["Localized_Hybrid"].append(
                (
                    mid,
                    weights["cf"] * cf_pred
                    + weights["cbf"] * cbf_score
                    + weights["pref"] * pref_score,
                )
            )

        for model_name, scores in model_scores.items():
            top_k = [
                movie_id
                for movie_id, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:K]
            ]
            metrics = calculate_metrics(
                top_k, ground_truth, cbf_meta, pref_langs, pref_genres
            )
            results.append(
                {
                    "UserId": uid,
                    "Model": model_name,
                    "Archetype": profile.get("cohort_group", "Unknown"),
                    "Support_History_Count": len(support_history),
                    "Heldout_Positive_Count": len(ground_truth),
                    "Candidate_Count": len(candidates),
                    "TopK_MovieIds": "|".join(str(movie_id) for movie_id in top_k),
                    **metrics,
                }
            )

    if not results:
        print("No valid evaluation results generated. Check data alignment.")
        return

    results_df = pd.DataFrame(results)
    user_level_path = f"{RESULTS_DIR}/thesis_evaluation_user_level.csv"
    results_df.to_csv(user_level_path, index=False)

    summary_table = (
        results_df.groupby("Model")
        .agg(
            {
                "Precision@10": ["mean", "std"],
                "Recall@10": ["mean", "std"],
                "NDCG@10": ["mean", "std"],
                "Filter_Bubble_Score": ["mean", "std"],
                "Language_Diversity": "mean",
                "Genre_Diversity": "mean",
                "Novelty@10": ["mean", "std"],
            }
        )
        .round(4)
    )
    summary_table.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in list(summary_table.columns)
    ]
    summary_table = summary_table.reset_index()

    print("\nTABLE 1: MODEL PERFORMANCE")
    print(summary_table)
    summary_table.to_csv(f"{RESULTS_DIR}/thesis_summary_by_model.csv", index=False)

    significance_rows = []
    print("\nSTATISTICAL SIGNIFICANCE TESTS")
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        paired = results_df[results_df["Model"].isin([model_a, model_b])].pivot(
            index="UserId", columns="Model", values="NDCG@10"
        )
        paired = paired.dropna()
        scores_a = pd.to_numeric(paired[model_a], errors="coerce").to_numpy(dtype=float)
        scores_b = pd.to_numeric(paired[model_b], errors="coerce").to_numpy(dtype=float)

        row = {
            "Model_A": model_a,
            "Model_B": model_b,
            "Metric": "NDCG@10",
            "N": len(paired),
            "Mean_A": round(float(scores_a.mean()), 4) if len(scores_a) else 0,
            "Mean_B": round(float(scores_b.mean()), 4) if len(scores_b) else 0,
            "T_Statistic": "",
            "P_Value": "",
            "Cohens_D": "",
        }

        diff = scores_a - scores_b
        if len(scores_a) > 1 and len(scores_b) > 1 and np.var(diff) > 0:
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            effect_size = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
            row.update(
                {
                    "T_Statistic": round(float(t_stat), 4),
                    "P_Value": round(float(p_value), 6),
                    "Cohens_D": round(float(effect_size), 4),
                }
            )
            print(
                f"{model_a} vs {model_b}: t={t_stat:.3f}, p={p_value:.4f}, Cohen's d={effect_size:.3f}"
            )
        else:
            print(f"{model_a} vs {model_b}: not enough variance for paired t-test")

        significance_rows.append(row)

    pd.DataFrame(significance_rows).to_csv(
        f"{RESULTS_DIR}/thesis_significance_tests.csv", index=False
    )

    print("\n95% CONFIDENCE INTERVALS")
    for model in MODEL_ORDER:
        model_scores = results_df[results_df["Model"] == model]["NDCG@10"]
        scores = pd.to_numeric(model_scores, errors="coerce").to_numpy(dtype=float)
        mean = scores.mean()
        if len(scores) > 1 and stats.sem(scores) > 0:
            ci = stats.t.interval(
                0.95, len(scores) - 1, loc=mean, scale=stats.sem(scores)
            )
            print(f"{model}: NDCG@10 = {mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print(f"{model}: NDCG@10 = {mean:.4f} [N/A]")

    print("\nFAIRNESS ANALYSIS: BIAS BY ARCHETYPE")
    archetype_perf = results_df.groupby("Archetype")[
        ["Precision@10", "Recall@10", "NDCG@10", "Filter_Bubble_Score"]
    ].mean()
    print(archetype_perf)

    for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
        disparity = archetype_perf[metric].max() - archetype_perf[metric].min()
        print(f"\n{metric} Disparity (Fairness Gap): {disparity:.4f}")
        print(
            f"  Best: {archetype_perf[metric].idxmax()} ({archetype_perf[metric].max():.4f})"
        )
        print(
            f"  Worst: {archetype_perf[metric].idxmin()} ({archetype_perf[metric].min():.4f})"
        )

    print("\nFILTER BUBBLE ANALYSIS (RQ2 FOCUS)")
    bubble_by_arch = (
        results_df.groupby(["Model", "Archetype"])["Filter_Bubble_Score"]
        .mean()
        .unstack()
    )
    print(bubble_by_arch)
    print("\nInterpretation: Higher = more preference-locked recommendations.")

    summary = (
        results_df.groupby(["Model", "Archetype"]).mean(numeric_only=True).reset_index()
    )
    summary.to_csv(f"{RESULTS_DIR}/thesis_evaluation_metrics.csv", index=False)

    print("\nEVALUATION COMPLETE!")
    print(f"User-level rows: {user_level_path}")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
