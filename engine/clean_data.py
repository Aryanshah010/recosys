import os
import pandas as pd
import numpy as np

# Directory structure
RAW_DIR = "data/raw/ml-32m"
PROCESSED_DIR = "data/processed"


def clean_movie_data():
    """
    Clean and align movie metadata with IMDb and TMDb IDs.
    Saves the result as movies_clean.csv.
    """
    print("Step 1: Processing movie metadata...")

    movies_path = os.path.join(RAW_DIR, "movies.csv")
    links_path = os.path.join(RAW_DIR, "links.csv")

    # Validate files
    for path in [movies_path, links_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # Load datasets
    movies_df = pd.read_csv(movies_path)
    links_df = pd.read_csv(links_path)

    # Validate required columns
    required_movie_cols = {"movieId", "title", "genres"}
    required_link_cols = {"movieId", "imdbId", "tmdbId"}

    if not required_movie_cols.issubset(movies_df.columns):
        raise ValueError("movies.csv is missing required columns.")

    if not required_link_cols.issubset(links_df.columns):
        raise ValueError("links.csv is missing required columns.")

    # Remove records without TMDb IDs
    links_df = links_df.dropna(subset=["tmdbId"]).copy()

    links_df["imdbId"] = links_df["imdbId"].astype(int)
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)

    # Merge datasets
    movies_clean = pd.merge(
        movies_df,
        links_df[["movieId", "imdbId", "tmdbId"]],
        on="movieId",
        how="inner"
    )

    # Normalize genre formatting
    movies_clean["genres"] = (
        movies_clean["genres"]
        .fillna("")
        .str.replace("|", ", ", regex=False)
    )

    # Save output
    output_path = os.path.join(PROCESSED_DIR, "movies_clean.csv")
    movies_clean.to_csv(output_path, index=False)

    print(
        f"Saved movie metadata: {output_path} "
        f"({len(movies_clean):,} movies)"
    )

    return movies_clean


def clean_and_downsample_ratings(
    min_user_activity=50,
    min_movie_popularity=100
):
    """
    Filter ratings to keep:
    - Users with at least min_user_activity ratings
    - Movies with at least min_movie_popularity ratings

    Saves the result as ratings_clean.csv.
    """

    print("\nStep 2: Processing ratings dataset...")

    ratings_path = os.path.join(RAW_DIR, "ratings.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    print("Loading ratings data...")

    chunks = []

    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
        },
        chunksize=5_000_000
    ):
        chunks.append(chunk)

    ratings_df = pd.concat(chunks, ignore_index=True)

    print(f"Original ratings count: {len(ratings_df):,}")

    print(
        f"🧹 Filtering users >= {min_user_activity} ratings "
        f"and movies >= {min_movie_popularity} ratings..."
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

    print(f"Filtering stabilized after {iteration} iterations.")
    print(f"Final ratings count: {len(ratings_df):,}")

    output_path = os.path.join(PROCESSED_DIR, "ratings_clean.csv")
    ratings_df.to_csv(output_path, index=False)

    print(f"Saved ratings dataset: {output_path}")

    return ratings_df


def main():
    """
    Execute the full preprocessing pipeline.
    """
    print("Starting Dataset Alignment Pipeline...\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    clean_movie_data()

    clean_and_downsample_ratings(
        min_user_activity=80,
        min_movie_popularity=150
    )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()