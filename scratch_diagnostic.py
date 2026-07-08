import pandas as pd
import numpy as np

SYNTH_RATINGS = "data/processed/synthetic_ratings.csv"
SYNTH_USERS = "data/processed/synthetic_users.csv"
MOVIES_FINAL = "data/processed/movies_final.csv"

def main():
    print("Loading data...")
    ratings = pd.read_csv(SYNTH_RATINGS)
    users = pd.read_csv(SYNTH_USERS)
    movies = pd.read_csv(MOVIES_FINAL)
    
    # Create a mapping from movieId to language
    movie_lang = dict(zip(movies["movieId"], movies["language"]))
    
    # Map primary language to archetype
    # Assuming user languages column is a string like "Korean|English", we take the first
    archetype_to_lang = {
        "kdrama": "Korean",
        "bollywood": "Hindi",
        "anime": "Japanese",
        "hollywood": "English"
    }
    
    results = []
    
    for _, user_row in users.iterrows():
        user_id = user_row["userId"]
        archetype = user_row["archetype"]
        
        # If archetype is NaN or not in our map, skip or fallback
        if pd.isna(archetype) or archetype not in archetype_to_lang:
            continue
            
        expected_lang = archetype_to_lang[archetype]
        
        user_ratings = ratings[ratings["userId"] == user_id]
        if user_ratings.empty:
            continue
            
        # Get languages of rated movies
        rated_movie_ids = user_ratings["movieId"].tolist()
        rated_langs = [movie_lang.get(mid, "Unknown") for mid in rated_movie_ids]
        
        # Calculate percentage matching expected primary language
        matching_count = sum(1 for lang in rated_langs if lang == expected_lang)
        pct_matching = matching_count / len(rated_langs) if rated_langs else 0
        
        results.append({
            "userId": user_id,
            "archetype": archetype,
            "expected_lang": expected_lang,
            "pct_matching": pct_matching
        })
        
    df_results = pd.DataFrame(results)
    
    print("\nSystematic Match Verification:")
    summary = df_results.groupby("archetype")["pct_matching"].agg(["mean", "min", "max", "count"])
    print(summary)
    
    # Let's print a few specific users
    print("\nSample Users:")
    print(df_results.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
