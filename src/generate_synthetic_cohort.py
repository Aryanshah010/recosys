from __future__ import annotations

import os
import random
from typing import TypedDict

import numpy as np
import pandas as pd

CATALOG_PATH = "data/processed/movies_final.csv"
REAL_RATINGS_PATH = "data/processed/ratings_final.csv"
PROCESSED_DIR = "data/processed"


class Archetype(TypedDict):
    name: str
    count: int
    langs: list[str]
    genres: list[str]


print("Loading validated movie catalog and real ratings...")
movies = pd.read_csv(CATALOG_PATH)
real_ratings = pd.read_csv(REAL_RATINGS_PATH)

movies["clean_genres"] = movies["clean_genres"].fillna("").astype(str)
movies["language"] = movies["language"].fillna("").astype(str).str.upper()

rating_distribution = real_ratings["rating"].value_counts(normalize=True).sort_index()
rating_values: np.ndarray = np.asarray(rating_distribution.index, dtype=np.float64)
rating_probs: np.ndarray = np.asarray(rating_distribution.values, dtype=np.float64)


archetypes: list[Archetype] = [
    {
        "name": "A_Techie",
        "count": 50,
        "langs": ["EN", "JA"],
        "genres": ["ScienceFiction", "Action", "Thriller", "Documentary"],
    },
    {
        "name": "B_Mainstream",
        "count": 50,
        "langs": ["HI", "EN"],
        "genres": ["Action", "Drama", "Romance", "Comedy"],
    },
    {
        "name": "C_AnimeFan",
        "count": 50,
        "langs": ["JA", "KO", "EN"],
        "genres": ["Animation", "Fantasy", "ScienceFiction"],
    },
    {
        "name": "D_Localist",
        "count": 50,
        "langs": ["NE", "HI", "EN"],
        "genres": ["Drama", "Romance", "Musical"],
    },
]

synthetic_users: list[dict] = []
synthetic_ratings: list[dict] = []
user_id_counter = 1000000

print("Generating Synthetic Cohort & Validated Interactions...")
for arch in archetypes:
    langs: list[str] = arch["langs"]
    genres: list[str] = arch["genres"]
    mask_lang = movies["language"].isin(langs)
    mask_genre = movies["clean_genres"].apply(lambda x: any(g in x for g in genres))
    preferred_movies = movies[mask_lang | mask_genre]

    if len(preferred_movies) < 50:
        preferred_movies = movies.head(500)

    for _ in range(arch["count"]):
        uid = user_id_counter
        user_id_counter += 1

        is_cold_start = random.random() < 0.15

        synthetic_users.append(
            {
                "user_id": uid,
                "country": "Nepal",
                "gender": "Male",
                "occupation": "IT Student",
                "cohort_group": arch["name"],
                "preferred_languages": "|".join(arch["langs"]),
                "preferred_genres": "|".join(arch["genres"]),
                "is_cold_start": is_cold_start,
            }
        )

        if not is_cold_start:
            rating_std = np.random.uniform(0.3, 1.2)

            num_pos = random.randint(15, 30)
            watched_pos = preferred_movies.sample(n=min(num_pos, len(preferred_movies)))
            for _, movie in watched_pos.iterrows():
                base_rating = float(np.random.choice(rating_values, p=rating_probs))
                final_rating = min(5.0, max(3.5, base_rating + (1.5 - rating_std)))
                final_rating = round(final_rating * 2) / 2

                synthetic_ratings.append(
                    {
                        "userId": uid,
                        "movieId": movie["movieId"],
                        "rating": final_rating,
                        "is_synthetic": True,
                    }
                )

            num_neg = random.randint(5, 10)
            watched_neg = movies.sample(n=num_neg)
            for _, movie in watched_neg.iterrows():
                if movie["movieId"] not in watched_pos["movieId"].values:
                    base_rating = float(np.random.choice(rating_values, p=rating_probs))
                    final_rating = min(3.0, max(0.5, base_rating - rating_std))
                    final_rating = round(final_rating * 2) / 2

                    synthetic_ratings.append(
                        {
                            "userId": uid,
                            "movieId": movie["movieId"],
                            "rating": final_rating,
                            "is_synthetic": True,
                        }
                    )

os.makedirs(PROCESSED_DIR, exist_ok=True)
pd.DataFrame(synthetic_users).to_csv(
    f"{PROCESSED_DIR}/synthetic_user_profiles.csv", index=False
)
pd.DataFrame(synthetic_ratings).to_csv(
    f"{PROCESSED_DIR}/synthetic_interactions.csv", index=False
)

print(
    f"Success! Generated {len(synthetic_users)} users and {len(synthetic_ratings)} valid interactions."
)
