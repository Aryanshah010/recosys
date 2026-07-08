import numpy as np
import pandas as pd
from surprise import dump

SVD_MODEL_PATH = "data/processed/svd_model.pkl"
SYNTH_RATINGS = "data/processed/synthetic_ratings.csv"
MOVIES_FINAL = "data/processed/movies_final.csv"

def main():
    print("Loading data...")
    _, svd = dump.load(SVD_MODEL_PATH)
    trainset = svd.trainset
    
    ratings = pd.read_csv(SYNTH_RATINGS)
    movies = pd.read_csv(MOVIES_FINAL)
    
    # 1. Evaluate user 900000 (kdrama) p_hat
    user_id = 900000
    user_ratings = ratings[(ratings["userId"] == user_id) & (ratings["split"] == "train")]
    print(f"\nUser {user_id} liked {len(user_ratings)} movies in train split:")
    
    liked_mids = user_ratings["movieId"].tolist()
    liked_ratings = user_ratings["rating"].tolist()
    for mid, rating in zip(liked_mids[:10], liked_ratings[:10]):
        title_matches = movies[movies["movieId"] == mid]["title"].values
        title = title_matches[0] if len(title_matches) > 0 else "Unknown"
        print(f"  {mid}: {title} - {rating}")
    
    qi_liked = []
    for mid in liked_mids:
        try:
            inner_iid = trainset.to_inner_iid(mid)
            qi_liked.append(svd.qi[inner_iid])
        except ValueError:
            pass
    
    print(f"\nFound {len(qi_liked)} out of {len(liked_mids)} movies in SVD trainset")
    if qi_liked:
        p_hat = np.mean(qi_liked, axis=0)
        print(f"p_hat norm: {np.linalg.norm(p_hat)}")
    else:
        print("p_hat is zero")
        
    # 2. Check localized candidates for Korean
    print("\n--- Korean Language Candidates ---")
    lang_counts = movies["language"].value_counts()
    print("Language distribution (top 5):")
    print(lang_counts.head())
    
    lang = "Korean"
    lang_rows = movies[movies["language"] == lang]
    print(f"\nFound {len(lang_rows)} movies with language='{lang}'")
    if not lang_rows.empty:
        top_lang = lang_rows.nlargest(5, "vote_count")
        print("Top 5 by vote_count:")
        for _, row in top_lang.iterrows():
            print(f"  {row['movieId']} ({row['title']}) - votes: {row.get('vote_count', 'N/A')}")

if __name__ == "__main__":
    main()
