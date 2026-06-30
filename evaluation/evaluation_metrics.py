import pandas as pd
import numpy as np
import pickle
import os
from surprise import Dataset, Reader, SVD
import warnings
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

REAL_RATINGS = "data/processed/ratings_final.csv"
SYNTH_RATINGS = "data/processed/synthetic_interactions.csv"
SYNTH_PROFILES = "data/processed/synthetic_user_profiles.csv"
CBF_MATRIX = "data/processed/cbf_matrix.pkl"
CBF_META = "data/processed/cbf_metadata.pkl"
MOVIES_FINAL = "data/processed/movies_final.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

GENRE_MAPPING = {
    "Sci-Fi": "ScienceFiction",
    "Science Fiction": "ScienceFiction",
    "Anime": "Animation",
    "Children's": "Family",
}
LANGUAGE_MAPPING = {
    "HINDI": "HI",
    "ENGLISH": "EN",
    "NEPALI": "NE",
    "JAPANESE": "JA",
    "KOREAN": "KO",
}


def normalize_prefs(profile):
    langs_raw = str(profile.get("preferred_languages", "")).split("|")
    genres_raw = str(profile.get("preferred_genres", "")).split("|")

    langs = set(
        [
            LANGUAGE_MAPPING.get(l.strip().upper(), l.strip().upper())
            for l in langs_raw  # noqa: E741
            if l.strip()
        ]
    )
    genres = set(
        [
            GENRE_MAPPING.get(g.strip(), g.strip().replace(" ", ""))
            for g in genres_raw
            if g.strip()
        ]
    )
    return langs, genres


print("Loading Data & Performing Safe 80/20 Train/Test Split...")
real_df = pd.read_csv(REAL_RATINGS)
synth_df = pd.read_csv(SYNTH_RATINGS)
profiles_df = pd.read_csv(SYNTH_PROFILES)
movies_meta = pd.read_csv(MOVIES_FINAL)

synth_shuffled = synth_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(synth_shuffled) * 0.8)
synth_train = synth_shuffled.iloc[:split_idx].copy()
synth_test = synth_shuffled.iloc[split_idx:].copy()

synth_train["userId"] = synth_train["userId"].astype(str)
synth_test["userId"] = synth_test["userId"].astype(str)
real_df["userId"] = real_df["userId"].astype(str)

combined_train = pd.concat(
    [
        real_df[["userId", "movieId", "rating"]],
        synth_train[["userId", "movieId", "rating"]],
    ],
    ignore_index=True,
)

print(f"Training SVD on {len(combined_train):,} combined interactions...")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(combined_train, reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=150, n_epochs=30, lr_all=0.003, reg_all=0.01, random_state=42)
svd.fit(trainset)

print("Loading CBF artifacts...")
with open(CBF_MATRIX, "rb") as f:
    cosine_sim = pickle.load(f)
with open(CBF_META, "rb") as f:
    cbf_meta = pickle.load(f)
cbf_indices = pd.Series(cbf_meta.index, index=cbf_meta["movieId"]).drop_duplicates()

print("Running Offline Evaluation on Synthetic Cohort...")

test_positive = synth_test[synth_test["rating"] >= 4.0]
valid_test_users = (
    test_positive.groupby("userId").filter(lambda x: len(x) >= 2)["userId"].unique()
)
eval_users = np.random.choice(
    valid_test_users, size=min(150, len(valid_test_users)), replace=False
)

results = []
K = 10

for uid in eval_users:
    profile_match = profiles_df[profiles_df["user_id"].astype(str) == str(uid)]
    if profile_match.empty:
        continue
    profile = profile_match.iloc[0]

    pref_langs, pref_genres = normalize_prefs(profile)
    is_cold = bool(profile.get("is_cold_start", False))

    ground_truth = synth_test[
        (synth_test["userId"] == str(uid)) & (synth_test["rating"] >= 4.0)
    ]["movieId"].tolist()
    if not ground_truth:
        continue

    train_history = synth_train[synth_train["userId"] == str(uid)]["movieId"].tolist()
    valid_history = [m for m in train_history if m in cbf_indices.index]

    candidates = set()

    if valid_history:
        safe_idxs = []
        for m in valid_history:
            idx_val = cbf_indices.get(m)
            if idx_val is not None:
                if isinstance(idx_val, pd.Series):
                    safe_idxs.append(int(idx_val.iloc[0]))
                else:
                    safe_idxs.append(int(idx_val))

        if safe_idxs:
            unique_idxs = np.unique(safe_idxs)

            sim_matrix_slice = cosine_sim[unique_idxs, :]
            sim_scores = np.mean(sim_matrix_slice, axis=0)

            top_k = min(300, len(sim_scores))
            top_cbf_idx = np.argsort(sim_scores)[-top_k:]
            candidates.update(cbf_meta.iloc[top_cbf_idx]["movieId"].tolist())

    candidates.update(movies_meta.nlargest(300, "vote_count")["movieId"].tolist())
    candidates = list(candidates - set(train_history))

    if not candidates:
        continue

    model_scores = {"CF": [], "CBF": [], "NonLocal_Hybrid": [], "Localized_Hybrid": []}

    for mid in candidates:
        cf_pred = svd.predict(str(uid), mid).est / 5.0

        cbf_score = 0.0
        if valid_history and mid in cbf_indices.index:
            mid_idx = cbf_indices[mid]
            cbf_score = float(
                np.mean([cosine_sim[cbf_indices[h]][mid_idx] for h in valid_history])
            )

        pref_score = 0.0
        if mid in cbf_indices.index:
            m_row = cbf_meta.loc[cbf_indices[mid]]
            m_lang = str(m_row["language"]).upper()
            m_genres = set(str(m_row["clean_genres"]).split())
            if m_lang in pref_langs:
                pref_score += 0.6
            if not m_genres.isdisjoint(pref_genres):
                pref_score += 0.4

        model_scores["CF"].append((mid, cf_pred))
        model_scores["CBF"].append((mid, cbf_score))

        model_scores["NonLocal_Hybrid"].append(
            (mid, 0.625 * cf_pred + 0.375 * cbf_score)
        )

        if is_cold or not valid_history:
            final_score = 0.6 * cbf_score + 0.4 * pref_score
        else:
            final_score = 0.5 * cf_pred + 0.3 * cbf_score + 0.2 * pref_score
        model_scores["Localized_Hybrid"].append((mid, final_score))

    # METRICS CALCULATION
    for model_name, scores in model_scores.items():
        top_k = [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)[:K]]

        # Accuracy Metrics
        hits = len(set(top_k).intersection(set(ground_truth)))
        precision = hits / K
        recall = hits / len(ground_truth) if ground_truth else 0

        # NDCG@K
        relevance = [1 if m in ground_truth else 0 for m in top_k]
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        ideal_rels = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)])
        ndcg = dcg / idcg if idcg > 0 else 0

        # Diversity / Bias Metrics
        top_k_meta = cbf_meta[cbf_meta["movieId"].isin(top_k)]
        unique_langs = top_k_meta["language"].nunique() if len(top_k_meta) > 0 else 0
        all_genres = " ".join(top_k_meta["clean_genres"].fillna("").astype(str)).split()
        unique_genres = len(set(all_genres))

        # Filter Bubble Score: % of Top-K matching explicit preferences
        bubble_matches = 0
        for _, row in top_k_meta.iterrows():
            r_lang = str(row["language"]).upper()
            r_genres = (
                set(str(row["clean_genres"]).split())
                if pd.notna(row["clean_genres"])
                else set()
            )
            if r_lang in pref_langs or not r_genres.isdisjoint(pref_genres):
                bubble_matches += 1
        bubble_score = bubble_matches / K if K > 0 else 0

        # Novelty Calculation
        novelty_score = 0
        for movie_id in top_k:
            if movie_id not in cbf_indices.index:
                novelty_score += 1
            else:
                m_idx = cbf_indices[movie_id]
                if isinstance(m_idx, pd.Series):
                    m_idx = int(m_idx.iloc[0])
                else:
                    m_idx = int(m_idx)

                m_row = cbf_meta.loc[m_idx]
                m_lang = str(m_row.get("language", "")).upper()
                m_genres = set(str(m_row.get("clean_genres", "")).split())

                if m_lang not in pref_langs and m_genres.isdisjoint(pref_genres):
                    novelty_score += 1
        novelty = novelty_score / K if K > 0 else 0

        results.append(
            {
                "Model": model_name,
                "Archetype": profile.get("cohort_group", "Unknown"),
                "Precision@10": round(precision, 4),
                "Recall@10": round(recall, 4),
                "NDCG@10": round(ndcg, 4),
                "Novelty@10": round(novelty, 4),
                "Language_Diversity": unique_langs,
                "Genre_Diversity": unique_genres,
                "Filter_Bubble_Score": round(bubble_score, 4),
            }
        )

if not results:
    print("No valid evaluation results generated. Check data alignment.")
else:
    results_df = pd.DataFrame(results)

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

    print("\nTABLE 1: MODEL PERFORMANCE")
    print(summary_table)
    summary_table.to_csv(f"{RESULTS_DIR}/thesis_summary_by_model.csv")

    print("\nSTATISTICAL SIGNIFICANCE TESTS")
    models = results_df["Model"].unique()
    for model_a, model_b in combinations(models, 2):
        scores_a = results_df[results_df["Model"] == model_a]["NDCG@10"].values
        scores_b = results_df[results_df["Model"] == model_b]["NDCG@10"].values

        if (
            len(scores_a) > 1
            and len(scores_b) > 1
            and np.var(scores_a) > 0
            and np.var(scores_b) > 0
        ):
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            pooled_std = np.sqrt((scores_a.std() ** 2 + scores_b.std() ** 2) / 2)
            effect_size = (
                (scores_a.mean() - scores_b.mean()) / pooled_std
                if pooled_std > 0
                else 0
            )
            print(
                f"{model_a} vs {model_b}: t={t_stat:.3f}, p={p_value:.4f}, Cohen's d={effect_size:.3f}"
            )

    print("\n95% CONFIDENCE INTERVALS")
    for model in models:
        scores = results_df[results_df["Model"] == model]["NDCG@10"].values
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
        if metric in archetype_perf:
            disparity = archetype_perf[metric].max() - archetype_perf[metric].min()
            print(f"\n{metric} Disparity (Fairness Gap): {disparity:.4f}")
            print(
                f"  Best: {archetype_perf[metric].idxmax()} ({archetype_perf[metric].max():.4f})"
            )
            print(
                f"  Worst: {archetype_perf[metric].idxmin()} ({archetype_perf[metric].min():.4f})"
            )
            if archetype_perf[metric].max() > 0:
                print(
                    f"{(disparity / archetype_perf[metric].max() * 100):.1f}% performance gap"
                )

    print("\nFILTER BUBBLE ANALYSIS (RQ2 FOCUS)")
    bubble_by_arch = (
        results_df.groupby(["Model", "Archetype"])["Filter_Bubble_Score"]
        .mean()
        .unstack()
    )
    print(bubble_by_arch)
    print("\n Interpretation: Higher = more trapped in preferences (worse)")
    print(" Localized_Hybrid should show LOWER bubble scores = better diversity")

    summary = (
        results_df.groupby(["Model", "Archetype"]).mean(numeric_only=True).reset_index()
    )
    summary.to_csv(f"{RESULTS_DIR}/thesis_evaluation_metrics.csv", index=False)

    print("\nEVALUATION COMPLETE!")
    print(f"Results saved to {RESULTS_DIR}/")
