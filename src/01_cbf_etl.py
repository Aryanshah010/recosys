import pandas as pd
import ast
import os

ML_MOVIES_PATH = 'data/raw/ml-32m/movies.csv'
ML_LINKS_PATH = 'data/raw/ml-32m/links.csv'
KAGGLE_META_PATH = 'data/raw/tmdb/movies_metadata.csv'
OUTPUT_PATH = 'data/processed/cbf_items.csv'

print("1. Loading raw datasets...")
ml_movies = pd.read_csv(ML_MOVIES_PATH)
ml_links = pd.read_csv(ML_LINKS_PATH)

kaggle_meta = pd.read_csv(KAGGLE_META_PATH, low_memory=False) 

print("2. Cleaning TMDb Metadata...")
# Convert TMDb ID to numeric, coerce errors to NaN (filters out TV shows/bad data)
kaggle_meta['tmdbId'] = pd.to_numeric(kaggle_meta['id'], errors='coerce')
kaggle_meta = kaggle_meta.dropna(subset=['tmdbId'])
kaggle_meta['tmdbId'] = kaggle_meta['tmdbId'].astype(int)

# Parse stringified JSON genres: "[{'id': 16, 'name': 'Animation'}]" -> "Animation"
def parse_genres(x):
    try:
        genres = ast.literal_eval(x)
        return ' '.join([g['name'].replace('-', '').replace(' ', '') for g in genres])
    except:
        return 'Unknown'

kaggle_meta['clean_genres'] = kaggle_meta['genres'].apply(parse_genres)
kaggle_meta['language'] = kaggle_meta['original_language'].fillna('Unknown').str.upper()

# Extract TMDb vote counts for Popularity Bias modeling later
kaggle_meta['vote_count'] = pd.to_numeric(kaggle_meta['vote_count'], errors='coerce').fillna(0)

print("3. Merging and Filtering Unmatched Records...")
# Merge Kaggle metadata with MovieLens links
tmdb_mapped = pd.merge(
    ml_links[['movieId', 'tmdbId']], 
    kaggle_meta[['tmdbId', 'clean_genres', 'language', 'vote_count', 'title']], 
    on='tmdbId', 
    how='inner' # Thesis Constraint: Filter incomplete mappings
)

# Merge with official MovieLens movies.csv to get official titles and ML genres as fallback
cbf_items = pd.merge(ml_movies, tmdb_mapped, on='movieId', how='inner')

# Create the final "Soup" feature for TF-IDF
cbf_items['soup'] = cbf_items['clean_genres'] + ' ' + cbf_items['language']

# Save to processed
os.makedirs('data/processed', exist_ok=True)
cbf_items[['movieId', 'title_x', 'clean_genres', 'language', 'vote_count', 'soup']].to_csv(OUTPUT_PATH, index=False)
print(f"Success! Saved {len(cbf_items)} matched items to {OUTPUT_PATH}")