from __future__ import annotations

import ast
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from mappings import map_genre, map_language, UNKNOWN_GENRE_LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ML_DIR = "data/raw/ml-32m"
TMDB_DIR = "data/raw/tmdb"
PROCESSED_DIR = "data/processed"

MIN_USER_ACTIVITY = 20
MIN_MOVIE_POPULARITY = 20
RATINGS_CHUNK_SIZE = 5_000_000

REQUIRED_MOVIE_COLUMNS = [
    "movieId",
    "tmdbId",
    "title",
    "ml_genres",
    "clean_genres",
    "language",
    "overview",
    "release_year",
    "vote_average",
    "vote_count",
    "popularity",
]


def check_file_exists(path: str) -> None:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Check that the dataset has been downloaded and placed in the "
            "expected data/raw directory structure."
        )


def parse_tmdb_genre_names(genre_str: str) -> list[str]:

    try:
        genres = ast.literal_eval(genre_str)
        return [g["name"] for g in genres if "name" in g]
    except ValueError, SyntaxError, TypeError:
        return []


def normalize_ml_genres(genres_str: str) -> list[str]:

    if not isinstance(genres_str, str) or not genres_str.strip():
        return []
    return [g.strip() for g in genres_str.split("|") if g.strip()]


def canonicalize_genre_list(raw_genres: list[str]) -> str:

    if not raw_genres:
        return UNKNOWN_GENRE_LABEL

    canonical: set[str] = {map_genre(raw) for raw in raw_genres}

    return "|".join(sorted(canonical)) if canonical else UNKNOWN_GENRE_LABEL


def clean_overview_text(overview: object) -> str:

    if not isinstance(overview, str) or not overview.strip():
        return ""

    text = overview.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    return text.strip()


def extract_release_year(date_str: object) -> Optional[int]:

    if not isinstance(date_str, str) or len(date_str) < 4:
        return None
    match = re.match(r"^(\d{4})", date_str.strip())
    if not match:
        return None
    return int(match.group(1))


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("Loading MovieLens movies and links...")
    movies_path = os.path.join(ML_DIR, "movies.csv")
    links_path = os.path.join(ML_DIR, "links.csv")
    check_file_exists(movies_path)
    check_file_exists(links_path)
    ml_movies = pd.read_csv(movies_path)
    ml_links = pd.read_csv(links_path)
    logger.info("Loaded %d MovieLens movies.", len(ml_movies))
    return ml_movies, ml_links


def load_tmdb_metadata() -> pd.DataFrame:

    logger.info("Loading TMDB metadata...")
    tmdb_path = os.path.join(TMDB_DIR, "movies_metadata.csv")
    check_file_exists(tmdb_path)
    tmdb = pd.read_csv(tmdb_path, low_memory=False)

    tmdb["tmdbId"] = pd.to_numeric(tmdb["id"], errors="coerce")
    tmdb = tmdb.dropna(subset=["tmdbId"])
    tmdb["tmdbId"] = tmdb["tmdbId"].astype(int)

    tmdb["vote_average"] = pd.to_numeric(tmdb["vote_average"], errors="coerce")
    tmdb["vote_count"] = pd.to_numeric(tmdb["vote_count"], errors="coerce")
    tmdb["popularity"] = pd.to_numeric(tmdb["popularity"], errors="coerce")

    logger.info("Loaded %d TMDB records with valid tmdbId.", len(tmdb))

    before = len(tmdb)
    tmdb = tmdb.sort_values(
        ["vote_count", "vote_average"], ascending=[False, False]
    ).drop_duplicates(subset="tmdbId", keep="first")
    logger.info(
        "Removed %d duplicate tmdbId rows from TMDB metadata.", before - len(tmdb)
    )

    return tmdb


def build_unified_catalog() -> pd.DataFrame:

    logger.info("STEP 1: Building unified catalog (MovieLens + TMDB)")

    ml_movies, ml_links = load_movielens()
    tmdb = load_tmdb_metadata()

    ml_with_links = pd.merge(
        ml_movies, ml_links[["movieId", "tmdbId"]], on="movieId", how="left"
    )

    tmdb_subset = tmdb[
        [
            "tmdbId",
            "genres",
            "original_language",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
        ]
    ].copy()

    catalog = pd.merge(ml_with_links, tmdb_subset, on="tmdbId", how="left")

    catalog["ml_genre_list"] = (
        catalog["genres_x"].apply(normalize_ml_genres)
        if "genres_x" in catalog.columns
        else catalog["genres"].apply(normalize_ml_genres)
    )

    if "genres_y" in catalog.columns:
        tmdb_genre_col = catalog["genres_y"]
    else:
        tmdb_genre_col = pd.Series([None] * len(catalog))

    catalog["tmdb_genre_list"] = tmdb_genre_col.apply(
        lambda g: parse_tmdb_genre_names(g) if isinstance(g, str) else []
    )

    catalog["ml_genres"] = catalog["ml_genre_list"].apply(canonicalize_genre_list)

    def resolve_clean_genres(row: pd.Series) -> str:
        if row["tmdb_genre_list"]:
            return canonicalize_genre_list(row["tmdb_genre_list"])
        return canonicalize_genre_list(row["ml_genre_list"])

    catalog["clean_genres"] = catalog.apply(resolve_clean_genres, axis=1)
    catalog["language"] = catalog["original_language"].apply(map_language)
    catalog["overview"] = catalog["overview"].apply(clean_overview_text)

    catalog["release_year"] = catalog["release_date"].apply(extract_release_year)
    missing_year = catalog["release_year"].isna()
    if missing_year.any():
        title_years = catalog.loc[missing_year, "title"].str.extract(
            r"\((\d{4})\)\s*$"
        )[0]
        catalog.loc[missing_year, "release_year"] = pd.to_numeric(
            title_years, errors="coerce"
        )
    catalog["release_year"] = catalog["release_year"].astype("Int64")

    catalog["vote_average"] = catalog["vote_average"].fillna(0.0)
    catalog["vote_count"] = catalog["vote_count"].fillna(0).astype("Int64")
    catalog["popularity"] = catalog["popularity"].fillna(0.0)
    catalog["movieId"] = catalog["movieId"].astype("Int64")
    catalog["tmdbId"] = catalog["tmdbId"].astype("Int64")

    catalog = catalog[
        [
            "movieId",
            "tmdbId",
            "title",
            "ml_genres",
            "clean_genres",
            "language",
            "overview",
            "release_year",
            "vote_average",
            "vote_count",
            "popularity",
        ]
    ].copy()

    before_tmdb = len(catalog)
    has_tmdb = catalog["tmdbId"].notna()
    catalog = pd.concat(
        [
            catalog[has_tmdb]
            .sort_values(["vote_count", "vote_average"], ascending=[False, False])
            .drop_duplicates(subset=["tmdbId"], keep="first"),
            catalog[~has_tmdb],
        ],
        ignore_index=True,
    )
    logger.info(
        "Removed %d duplicate tmdbId rows (post-join).", before_tmdb - len(catalog)
    )

    before = len(catalog)
    duplicate_mask = catalog.duplicated(subset=["movieId"], keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(catalog.loc[duplicate_mask, "movieId"].unique().tolist())
        logger.warning(
            "Found %d movieId(s) with duplicate rows before dedup: %s",
            len(duplicate_ids),
            duplicate_ids,
        )
    catalog = catalog.drop_duplicates(subset=["movieId"])
    logger.info("Removed %d duplicate movieId rows.", before - len(catalog))

    logger.info("Unified catalog built with %d movies.", len(catalog))
    return catalog


def clean_ratings_and_remove_orphans(
    catalog_df: pd.DataFrame,
    min_user_activity: int = MIN_USER_ACTIVITY,
    min_movie_popularity: int = MIN_MOVIE_POPULARITY,
) -> pd.DataFrame:

    logger.info("STEP 2: Cleaning ratings and removing orphans")

    ratings_path = os.path.join(ML_DIR, "ratings.csv")
    check_file_exists(ratings_path)

    chunks = []
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        chunksize=RATINGS_CHUNK_SIZE,
    ):
        chunks.append(chunk)
    ratings_df = pd.concat(chunks, ignore_index=True)
    logger.info("Original ratings count: %d", len(ratings_df))

    before = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates(subset=["userId", "movieId"])
    logger.info("Removed %d duplicate ratings.", before - len(ratings_df))

    valid_movie_ids = set(catalog_df["movieId"].unique())
    ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)]
    logger.info(
        "Dropped orphan ratings (movies without a catalog entry). Remaining: %d",
        len(ratings_df),
    )

    logger.info(
        "Iteratively filtering users >= %d ratings and movies >= %d ratings...",
        min_user_activity,
        min_movie_popularity,
    )
    iteration = 0
    while True:
        iteration += 1
        previous_size = len(ratings_df)

        user_counts = ratings_df["userId"].value_counts()
        movie_counts = ratings_df["movieId"].value_counts()

        active_users = user_counts[user_counts >= min_user_activity].index
        popular_movies = movie_counts[movie_counts >= min_movie_popularity].index

        ratings_df = ratings_df[
            ratings_df["userId"].isin(active_users)
            & ratings_df["movieId"].isin(popular_movies)
        ]

        if len(ratings_df) == previous_size:
            break

    logger.info(
        "Filtering stabilized after %d iterations. Final ratings: %d",
        iteration,
        len(ratings_df),
    )
    return ratings_df


def validate_movies_final(catalog_df: pd.DataFrame) -> None:

    missing = [c for c in REQUIRED_MOVIE_COLUMNS if c not in catalog_df.columns]
    if missing:
        raise ValueError(f"movies_final.csv is missing required columns: {missing}")

    if not catalog_df["movieId"].is_unique:
        raise ValueError("movies_final.csv contains duplicate movieId values.")

    if catalog_df["movieId"].isna().any():
        raise ValueError("movies_final.csv contains null movieId values.")

    if catalog_df["clean_genres"].eq("").any():
        raise ValueError("movies_final.csv contains empty clean_genres values.")

    if catalog_df["language"].eq("").any():
        raise ValueError("movies_final.csv contains empty language values.")

    assert str(catalog_df["movieId"].dtype) == "Int64"
    assert str(catalog_df["tmdbId"].dtype) == "Int64"
    assert str(catalog_df["release_year"].dtype) == "Int64"
    assert str(catalog_df["vote_count"].dtype) == "Int64"


def validate_ratings_final(ratings_df: pd.DataFrame) -> None:

    if ratings_df["userId"].isna().any():
        raise ValueError("ratings_final.csv contains null userId values.")

    if ratings_df["movieId"].isna().any():
        raise ValueError("ratings_final.csv contains null movieId values.")

    if not ratings_df["rating"].between(0.5, 5.0).all():
        raise ValueError(
            "ratings_final.csv contains rating values outside the valid "
            "MovieLens range [0.5, 5.0], which indicates corrupted data."
        )


def main() -> None:

    logger.info("Starting clean_data.py pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    catalog_df = build_unified_catalog()
    validate_movies_final(catalog_df)

    ratings_df = clean_ratings_and_remove_orphans(catalog_df)
    validate_ratings_final(ratings_df)

    movies_path = os.path.join(PROCESSED_DIR, "movies_final.csv")
    ratings_path = os.path.join(PROCESSED_DIR, "ratings_final.csv")

    catalog_df.to_csv(movies_path, index=False)
    logger.info("Wrote %s (%d rows).", movies_path, len(catalog_df))

    ratings_df.to_csv(ratings_path, index=False)
    logger.info("Wrote %s (%d rows).", ratings_path, len(ratings_df))

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
