import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

INPUT_PATH = 'data/processed/cbf_items.csv'
MATRIX_OUT = 'data/processed/cbf_matrix.pkl'
META_OUT = 'data/processed/cbf_metadata.pkl'

print("Loading processed items...")
df = pd.read_csv(INPUT_PATH)

original_count = len(df)
df = df.drop_duplicates(subset=['movieId'], keep='first').reset_index(drop=True)
duplicates_removed = original_count - len(df)

if duplicates_removed > 0:
    print(f"Removed {duplicates_removed} duplicate movieId entries.")
    print(f"   Kept first occurrence for each movieId.")
else:
    print("No duplicate movieIds found.")

print(f"Building TF-IDF Matrix on {len(df)} unique movies...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])

print("Computing Cosine Similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cbf_indices = pd.Series(df.index, index=df['movieId'])

assert cbf_indices.is_unique, "FATAL: cbf_indices still contains duplicates!"
print(f"Index verified: {len(cbf_indices)} unique movieId → index mappings")

print("Saving artifacts...")
with open(MATRIX_OUT, 'wb') as f:
    pickle.dump(cosine_sim, f)
with open(META_OUT, 'wb') as f:
    pickle.dump(df, f)

print("CBF Engine artifacts saved successfully.")