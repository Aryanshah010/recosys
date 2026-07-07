from __future__ import annotations

import logging
import os
from typing import cast

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MOVIES_FINAL_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
CBF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "cbf_matrix.pkl")
CBF_METADATA_PATH = os.path.join(PROCESSED_DIR, "cbf_metadata.pkl")

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"
TFIDF_MIN_DF = 2

BAYESIAN_MIN_VOTES_QUANTILE = 0.25

GENRE_REPEAT_COUNT = 2

REQUIRED_INPUT_COLUMNS = [
    "movieId",
    "title",
    "clean_genres",
    "language",
    "overview",
    "release_year",
]


def check_file_exists(path: str) -> None:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Run clean_data.py first to produce movies_final.csv."
        )


def load_movies_final(path: str = MOVIES_FINAL_PATH) -> pd.DataFrame:

    check_file_exists(path)
    logger.info("Loading movie catalog from %s...", path)
    movies_df = pd.read_csv(path)

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in movies_df.columns]
    if missing:
        raise ValueError(
            f"movies_final.csv is missing required columns: {missing}. "
            "This script only consumes the schema produced by clean_data.py."
        )

    if movies_df["movieId"].isna().any():
        raise ValueError(
            "movies_final.csv contains null movieId values. This should "
            "have been caught by clean_data.py's validation."
        )

    if not movies_df["movieId"].is_unique:
        raise ValueError(
            "movies_final.csv contains duplicate movieId values. This "
            "indicates a bug in clean_data.py -- fix it there rather than "
            "deduplicating here, since Section 13 forbids this script from "
            "performing cleaning."
        )

    logger.info("Loaded %d movies.", len(movies_df))
    return movies_df


def build_content_soup(movies_df: pd.DataFrame) -> pd.Series:

    genres = (
        movies_df["clean_genres"]
        .fillna("")
        .astype(str)
        .str.replace("|", " ", regex=False)
    )
    language = movies_df["language"].fillna("").astype(str)
    overview = movies_df["overview"].fillna("").astype(str).str.lower()

    genre_block = ((genres + " ") * GENRE_REPEAT_COUNT).str.strip()

    soup = (genre_block + " " + language + " " + overview).str.strip()

    empty_count = int(soup.str.len().eq(0).sum())
    if empty_count > 0:
        logger.warning(
            "%d movies produced an empty content soup (should not happen "
            "if clean_genres/language default to 'Other' per Sections 9-10 "
            "-- investigate upstream data if this fires).",
            empty_count,
        )

    return soup


def compute_quality_scores(movies_df: pd.DataFrame) -> np.ndarray:

    vote_count = movies_df["vote_count"].fillna(0).astype(float).to_numpy()
    vote_average = movies_df["vote_average"].fillna(0.0).astype(float).to_numpy()

    has_votes_mask = vote_count > 0
    n_with_votes = int(has_votes_mask.sum())

    if n_with_votes == 0:
        logger.warning(
            "No movies with vote_count > 0 found. quality_scores will be all zeros."
        )
        return np.zeros(len(movies_df), dtype=np.float32)

    C = float(vote_average[has_votes_mask].mean())

    m = float(np.quantile(vote_count[has_votes_mask], BAYESIAN_MIN_VOTES_QUANTILE))

    m = max(m, 1.0)

    logger.info(
        "Bayesian WR params: C=%.3f, m=%.1f (from %.0f%% quantile of %d movies with votes).",
        C,
        m,
        BAYESIAN_MIN_VOTES_QUANTILE * 100,
        n_with_votes,
    )

    wr = (vote_count / (vote_count + m)) * vote_average + (m / (vote_count + m)) * C
    wr[~has_votes_mask] = 0.0

    lo, hi = wr.min(), wr.max()
    if hi - lo < 1e-8:
        logger.warning("quality_scores have no variance; returning zeros.")
        return np.zeros(len(movies_df), dtype=np.float32)

    quality_scores = ((wr - lo) / (hi - lo)).astype(np.float32)

    logger.info(
        "quality_scores computed: min=%.4f, max=%.4f, mean=%.4f (non-zero movies: %d).",
        float(quality_scores.min()),
        float(quality_scores.max()),
        float(quality_scores.mean()),
        int((quality_scores > 0).sum()),
    )
    return quality_scores


def get_matrix_shape(matrix: csr_matrix) -> tuple[int, int]:

    return cast(tuple[int, int], matrix.shape)


def build_tfidf_matrix(soup: pd.Series) -> tuple[csr_matrix, TfidfVectorizer]:

    logger.info(
        "Fitting TF-IDF vectorizer (ngram_range=%s, stop_words=%s, min_df=%d)...",
        TFIDF_NGRAM_RANGE,
        TFIDF_STOP_WORDS,
        TFIDF_MIN_DF,
    )
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
        min_df=TFIDF_MIN_DF,
        dtype=np.float32,
    )
    matrix = csr_matrix(vectorizer.fit_transform(soup))

    logger.info(
        "Vocabulary size: %d",
        len(vectorizer.vocabulary_),
    )
    memory_mb = (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (
        1024**2
    )

    logger.info(
        "Sparse matrix size: %.2f MB",
        memory_mb,
    )

    n_rows, n_cols = get_matrix_shape(matrix)
    logger.info(
        "TF-IDF matrix built: %d movies x %d features.",
        n_rows,
        n_cols,
    )
    return matrix, vectorizer


def validate_matrix(matrix: csr_matrix, movies_df: pd.DataFrame) -> None:

    n_rows, n_cols = get_matrix_shape(matrix)

    if n_rows != len(movies_df):
        raise ValueError(
            f"TF-IDF matrix has {n_rows} rows but catalog has "
            f"{len(movies_df)} movies -- row alignment with movie_ids "
            "in metadata would be broken."
        )

    if n_cols == 0:
        raise ValueError(
            "TF-IDF produced zero features. Check min_df is not too "
            "restrictive for this catalog size."
        )

    if matrix.nnz == 0:
        raise ValueError("TF-IDF matrix is entirely zero-valued.")


def build_metadata(
    movies_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    quality_scores: np.ndarray,
) -> dict[str, object]:

    return {
        "movie_ids": movies_df["movieId"].to_numpy(),
        "movie_index": {
            movie_id: idx for idx, movie_id in enumerate(movies_df["movieId"])
        },
        "titles": movies_df["title"].tolist(),
        "clean_genres": movies_df["clean_genres"].tolist(),
        "language": movies_df["language"].tolist(),
        "release_year": movies_df["release_year"].tolist(),
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "n_movies": len(movies_df),
        "n_features": len(vectorizer.get_feature_names_out()),
        "quality_scores": quality_scores,
    }


def save_artifacts(matrix: csr_matrix, metadata: dict[str, object]) -> None:

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(matrix, CBF_MATRIX_PATH, compress=3)
    logger.info("Wrote %s.", CBF_MATRIX_PATH)

    joblib.dump(metadata, CBF_METADATA_PATH, compress=3)
    logger.info("Wrote %s.", CBF_METADATA_PATH)


def main() -> None:

    logger.info("Starting build_cbf_matrix.py pipeline...")

    movies_df = load_movies_final()
    soup = build_content_soup(movies_df)
    matrix, vectorizer = build_tfidf_matrix(soup)
    validate_matrix(matrix, movies_df)
    quality_scores = compute_quality_scores(movies_df)
    metadata = build_metadata(movies_df, vectorizer, quality_scores)
    save_artifacts(matrix, metadata)

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
