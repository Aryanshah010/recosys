import os
import ast
import pandas as pd
import numpy as np

# --- DIRECTORY STRUCTURE ---
ML_DIR = "data/raw/ml-32m"
TMDB_DIR = "data/raw/tmdb"
PROCESSED_DIR = "data/processed"

def parse_tmdb_genres(genre_str):
    """
    Kaggle TMDb genres are stringified JSON: "[{'id': 16, 'name': 'Animation'}]"
    We parse this and clean it to match CBF requirements (e.g., 'ScienceFiction').
    """
    try:
        genres = ast.literal_eval(genre_str)
        # Remove spaces and hyphens to prevent TF-IDF tokenization issues
        return "|".join([g['name'].replace(' ', '').replace('-', '') for g in genres])
    except:
        return "Unknown"

def build_unified_catalog():
    """
    Step 1: Fuse MovieLens and TMDb. 
    Filters out unmatched records (Thesis Constraint).
    """
    print("--- STEP 1: Building Unified Catalog (MovieLens + TMDb) ---")
    
    # Load MovieLens
    ml_movies = pd.read_csv(os.path.join(ML_DIR, "movies.csv"))
    ml_links = pd.read_csv(os.path.join(ML_DIR, "links.csv"))
    
    # Load Kaggle TMDb
    kaggle_meta = pd.read_csv(os.path.join(TMDB_DIR, "movies_metadata.csv"), low_memory=False)
    
    # 1. Clean Kaggle TMDb IDs (The 'id' column is notoriously messy with TV show strings)
    kaggle_meta['tmdbId'] = pd.to_numeric(kaggle_meta['id'], errors='coerce')
    kaggle_meta = kaggle_meta.dropna(subset=['tmdbId'])
    kaggle_meta['tmdbId'] = kaggle_meta['tmdbId'].astype(int)
    
    # 2. Extract required CBF features from TMDb
    kaggle_meta['clean_genres'] = kaggle_meta['genres'].apply(parse_tmdb_genres)
    kaggle_meta['language'] = kaggle_meta['original_language'].fillna('Unknown').str.upper()
    kaggle_meta['vote_count'] = pd.to_numeric(kaggle_meta['vote_count'], errors='coerce').fillna(0)
    
    # Keep only what we need from TMDb
    tmdb_subset = kaggle_meta[['tmdbId', 'clean_genres', 'language', 'vote_count']].copy()
    
    # 3. Merge MovieLens with Links
    ml_with_links = pd.merge(ml_movies, ml_links[['movieId', 'tmdbId']], on='movieId', how='inner')
    
    # 4. THE CRITICAL THESIS MERGE (Inner Join filters unmatched records)
    # If a MovieLens movie doesn't have a TMDb match, it is dropped here.
    unified_catalog = pd.merge(ml_with_links, tmdb_subset, on='tmdbId', how='inner')
    
    print(f"✅ Catalog aligned successfully. {len(unified_catalog):,} movies survived the TMDb mapping filter.")
    return unified_catalog

def clean_ratings_and_remove_orphans(catalog_df, min_user_activity=20, min_movie_popularity=20):
    """
    Step 2: Clean ratings, remove orphans, and apply iterative coreness filtering.
    """
    print("\n--- STEP 2: Cleaning Ratings & Removing Orphans ---")
    
    ratings_path = os.path.join(ML_DIR, "ratings.csv")
    
    # Load ratings in chunks to save RAM
    chunks = []
    for chunk in pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"], 
                             dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32}, 
                             chunksize=5_000_000):
        chunks.append(chunk)
    ratings_df = pd.concat(chunks, ignore_index=True)
    
    print(f"Original ratings count: {len(ratings_df):,}")
    
    # 1. REMOVE ORPHAN REFERENCES (Thesis Constraint)
    # Keep only ratings for movies that actually exist in our unified TMDb catalog
    valid_movie_ids = set(catalog_df['movieId'].unique())
    ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]
    print(f"🗑️ Dropped orphan ratings (movies without TMDb metadata). Remaining: {len(ratings_df):,}")
    
    # 2. Iterative Coreness Filtering (Your existing logic)
    print(f"🧹 Filtering users >= {min_user_activity} and movies >= {min_movie_popularity}...")
    iteration = 0
    while True:
        iteration += 1
        previous_size = len(ratings_df)
        
        user_counts = ratings_df["userId"].value_counts()
        movie_counts = ratings_df["movieId"].value_counts()
        
        active_users = user_counts[user_counts >= min_user_activity].index
        popular_movies = movie_counts[movie_counts >= min_movie_popularity].index
        
        ratings_df = ratings_df[
            ratings_df["userId"].isin(active_users) & ratings_df["movieId"].isin(popular_movies)
        ]
        
        if len(ratings_df) == previous_size:
            break
            
    print(f"✅ Filtering stabilized after {iteration} iterations. Final ratings: {len(ratings_df):,}")
    return ratings_df

def main():
    print("🚀 Starting Master Dataset Alignment Pipeline...\n")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Execute Step 1
    catalog_df = build_unified_catalog()
    catalog_path = os.path.join(PROCESSED_DIR, "movies_final.csv")
    catalog_df.to_csv(catalog_path, index=False)
    
    # Execute Step 2
    ratings_df = clean_ratings_and_remove_orphans(
        catalog_df, 
        min_user_activity=20,   # Adjusted for better CF matrix density
        min_movie_popularity=20 # Adjusted for better CF matrix density
    )
    ratings_path = os.path.join(PROCESSED_DIR, "ratings_final.csv")
    ratings_df.to_csv(ratings_path, index=False)
    
    # Save a lightweight version specifically for the CBF Engine
    cbf_items = catalog_df[['movieId', 'title', 'clean_genres', 'language', 'vote_count']].copy()
    cbf_items['soup'] = cbf_items['clean_genres'] + ' ' + cbf_items['language']
    cbf_items.to_csv(os.path.join(PROCESSED_DIR, "cbf_items.csv"), index=False)

    print("\n🎉 Pipeline execution complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()