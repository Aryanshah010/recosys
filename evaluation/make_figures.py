"""Generate the RQ1 / RQ2 result figures."""

from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
PROCESSED_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_ORDER = [
    "Popularity",
    "CF_ColdStart",
    "CBF",
    "Affinity_Only",
    "NonLocal_Hybrid",
    "Localized_Hybrid",
]

PALETTE = {
    "Popularity": "#9aa0a6",
    "CF_ColdStart": "#6f9bd1",
    "CBF": "#63a088",
    "Affinity_Only": "#d9a441",
    "NonLocal_Hybrid": "#8d6ab8",
    "Localized_Hybrid": "#c8553d",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _read(path: str) -> pd.DataFrame | None:
    full = os.path.join(RESULTS_DIR, path)
    if not os.path.exists(full):
        logger.warning("Missing %s; skipping.", full)
        return None
    return pd.read_csv(full)


def _read_processed(name: str, **kwargs) -> pd.DataFrame | None:
    full = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(full):
        logger.warning("Missing %s; skipping.", full)
        return None
    return pd.read_csv(full, **kwargs)


def fig_rating_distribution(idx: int) -> None:
    """Descriptive: how the cleaned MovieLens ratings are distributed."""
    df = _read_processed("ratings_final.csv", usecols=["rating"])
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(9, 4.6))
    counts = df["rating"].value_counts().sort_index()
    ax.bar(counts.index, counts.to_numpy(dtype=float), width=0.4, color="#6f9bd1")

    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.set_title(
        f"Distribution of the {len(df):,} cleaned MovieLens ratings",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, f"{idx:02d}_rating_distribution.png")


def fig_genre_popularity(idx: int) -> None:
    """Descriptive: catalogue composition by canonical genre."""
    df = _read_processed("movies_final.csv", usecols=["clean_genres"])
    if df is None:
        return

    counts = df["clean_genres"].str.split("|").explode().value_counts().head(15)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(counts.index[::-1], counts.to_numpy(dtype=float)[::-1], color="#63a088")

    ax.set_xlabel("Number of titles")
    ax.set_title(
        "Catalogue composition — 15 most common genres",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, f"{idx:02d}_genre_popularity.png")


def fig_language_distribution(idx: int) -> None:
    """Descriptive: the catalogue's language skew, which motivates the study."""
    df = _read_processed("movies_final.csv", usecols=["language"])
    if df is None:
        return

    counts = df["language"].value_counts()
    top = counts.head(12)
    n_nepali = int(counts.get("Nepali", 0))

    fig, ax = plt.subplots(figsize=(9, 5.0))
    colors = ["#c8553d" if lang == "English" else "#9aa0a6" for lang in top.index]
    ax.barh(top.index[::-1], top.to_numpy(dtype=float)[::-1], color=colors[::-1])

    for i, v in enumerate(top.to_numpy(dtype=float)[::-1]):
        ax.text(v, i, f"  {v / len(df) * 100:.1f}%", va="center", fontsize=8)

    ax.set_xlabel("Number of titles")
    ax.set_title(
        f"Catalogue language skew — {counts.get('English', 0) / len(df) * 100:.0f}% "
        f"English, {n_nepali} Nepali of {len(df):,}",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, f"{idx:02d}_language_distribution.png")


def fig_model_performance(track: str, idx: int) -> None:
    df = _read(f"rq1_model_performance_{track}.csv")
    if df is None:
        return

    metrics = ["HR@10", "NDCG@10", "MRR@10", "Recall@10"]
    available = [m for m in metrics if f"{m}_mean" in df.columns]
    df = df.set_index("Model").reindex(MODEL_ORDER).dropna(how="all")

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4.2))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        vals = df[f"{metric}_mean"]
        colors = [PALETTE.get(m, "#888") for m in vals.index]
        ax.barh(range(len(vals)), vals.to_numpy(), color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("mean")
        for i, v in enumerate(vals.to_numpy()):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    fig.suptitle(
        f"RQ1 — ranking performance ({track} cohort)", fontsize=13, fontweight="bold"
    )
    _save(fig, f"{idx:02d}_rq1_performance_{track}.png")


def fig_diversity(track: str, idx: int) -> None:
    df = _read(f"rq2_diversity_by_model_{track}.csv")
    if df is None:
        return

    df = df.set_index("Model").reindex(MODEL_ORDER).dropna(how="all")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, title in zip(
        axes,
        ["Language_Diversity", "Genre_Diversity", "Novelty@10"],
        ["Language diversity", "Genre diversity", "Novelty@10 (1 − bubble)"],
    ):
        if col not in df.columns:
            continue
        vals = df[col]
        ax.barh(
            range(len(vals)),
            vals.to_numpy(),
            color=[PALETTE.get(m, "#888") for m in vals.index],
        )
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i, v in enumerate(vals.to_numpy()):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)

    fig.suptitle(
        f"RQ2 — exposure diversity ({track} cohort). Lower novelty = tighter filter bubble.",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, f"{idx:02d}_rq2_diversity_{track}.png")


def fig_bubble_by_archetype(track: str, idx: int) -> None:
    df = _read(f"rq2_filter_bubble_by_archetype_{track}.csv")
    if df is None:
        return

    df = df.set_index("Model").reindex(MODEL_ORDER).dropna(how="all")
    archetypes = [c for c in df.columns]

    x = np.arange(len(archetypes))
    width = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (model, row) in enumerate(df.iterrows()):
        ax.bar(
            x + i * width,
            row[archetypes].to_numpy(dtype=float),
            width,
            label=model,
            color=PALETTE.get(model, "#888"),  # type: ignore
        )

    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(archetypes)
    ax.set_ylabel("Filter Bubble Score")
    ax.set_title(
        f"RQ2 — filter bubble by user segment ({track} cohort)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=3)
    _save(fig, f"{idx:02d}_rq2_bubble_by_archetype_{track}.png")


def fig_tradeoff_curve(track: str, idx: int) -> None:
    """The accuracy / diversity trade-off — the key RQ2 figure."""
    df = _read(f"sensitivity_lambda_summary_{track}.csv")
    if df is None:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(df["Lambda"], df["NDCG@10"], "o-", color="#c8553d", lw=2, label="NDCG@10")
    ax1.set_xlabel("λ — weight on the cohort-affinity term")
    ax1.set_ylabel("NDCG@10", color="#c8553d")
    ax1.tick_params(axis="y", labelcolor="#c8553d")

    ax2.plot(
        df["Lambda"], df["Novelty@10"], "s--", color="#3d6fc8", lw=2, label="Novelty@10"
    )
    ax2.set_ylabel("Novelty@10 (1 − filter bubble)", color="#3d6fc8")
    ax2.tick_params(axis="y", labelcolor="#3d6fc8")
    ax2.grid(False)

    ax1.axvline(0.20, color="#666", ls=":", lw=1.5)
    ax1.annotate(
        "configured λ = 0.20",
        xy=(0.20, ax1.get_ylim()[1]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color="#666",
    )

    fig.suptitle(
        f"Accuracy gained against diversity lost ({track} cohort)",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, f"{idx:02d}_tradeoff_curve_{track}.png")


def fig_benchmark_representation(idx: int) -> None:
    """How unlike the target cohort the standard benchmark population is."""
    df = _read("benchmark_population_profile.csv")
    if df is None:
        return

    from src.cohort_spec import ARCHETYPE_BY_NAME

    df = df.copy()
    df["Target_Share_%"] = df["Archetype"].map(
        lambda a: (
            ARCHETYPE_BY_NAME[a]["target_share"] * 100  # pyright: ignore[reportArgumentType]
            if a in ARCHETYPE_BY_NAME
            else np.nan
        )
    )

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(
        x - width / 2,
        df["Benchmark_Share_%"],
        width,
        label="MovieLens benchmark",
        color="#9aa0a6",
    )
    ax.bar(
        x + width / 2,
        df["Target_Share_%"],
        width,
        label="Target cohort (design)",
        color="#c8553d",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Archetype"])
    ax.set_ylabel("% of population")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(
        "The research gap: benchmark population vs target cohort\n"
        "(log scale — minority archetypes are <0.5% of the benchmark)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()

    for i, (b, t) in enumerate(zip(df["Benchmark_Share_%"], df["Target_Share_%"])):
        ax.text(i - width / 2, b, f"{b:.2f}%", ha="center", va="bottom", fontsize=7)
        if np.isfinite(t):
            ax.text(i + width / 2, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=7)

    _save(fig, f"{idx:02d}_benchmark_representation_gap.png")


def fig_track_comparison(idx: int) -> None:
    """Real versus synthetic track, side by side."""
    path = os.path.join(RESULTS_DIR, "evaluation_user_level.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    if "Track" not in df.columns or df["Track"].nunique() < 2:
        return

    summary = df.groupby(["Track", "Model"])["NDCG@10"].mean().unstack("Track")
    summary = summary.reindex(MODEL_ORDER).dropna(how="all")

    x = np.arange(len(summary))
    tracks = list(summary.columns)
    width = 0.8 / len(tracks)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, track in enumerate(tracks):
        ax.bar(
            x + i * width,
            summary[track].to_numpy(dtype=float),
            width,
            label=f"{track} cohort",
        )

    ax.set_xticks(x + width * (len(tracks) - 1) / 2)
    ax.set_xticklabels(summary.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_title(
        "Real vs synthetic evaluation track — does the result survive real ratings?",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    _save(fig, f"{idx:02d}_track_comparison.png")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    idx = 1
    fig_rating_distribution(idx)
    idx += 1
    fig_genre_popularity(idx)
    idx += 1
    fig_language_distribution(idx)
    idx += 1

    fig_benchmark_representation(idx)
    idx += 1
    fig_track_comparison(idx)
    idx += 1

    for track in ("real", "synthetic"):
        fig_model_performance(track, idx)
        idx += 1
        fig_diversity(track, idx)
        idx += 1
        fig_bubble_by_archetype(track, idx)
        idx += 1
        fig_tradeoff_curve(track, idx)
        idx += 1

    logger.info("Figure generation complete.")


if __name__ == "__main__":
    main()
