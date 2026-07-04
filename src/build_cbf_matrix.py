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
    movies_df: pd.DataFrame, vectorizer: TfidfVectorizer
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
    metadata = build_metadata(movies_df, vectorizer)
    save_artifacts(matrix, metadata)

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
