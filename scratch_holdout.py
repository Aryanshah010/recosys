import pandas as pd
SYNTH_RATINGS = "data/processed/synthetic_ratings.csv"
MOVIES_FINAL = "data/processed/movies_final.csv"

ratings = pd.read_csv(SYNTH_RATINGS)
movies = pd.read_csv(MOVIES_FINAL)
user_id = 900000

train = ratings[(ratings["userId"] == user_id) & (ratings["split"] == "train")]
holdout = ratings[(ratings["userId"] == user_id) & (ratings["split"] == "holdout")]

print("TRAIN MOVIES:")
for mid, rating in zip(train["movieId"].head(10), train["rating"].head(10)):
    title_matches = movies[movies["movieId"] == mid]["title"].values
    title = title_matches[0] if len(title_matches) > 0 else "Unknown"
    lang_matches = movies[movies["movieId"] == mid]["language"].values
    lang = lang_matches[0] if len(lang_matches) > 0 else "Unknown"
    print(f"  {title} ({lang}) - {rating}")

print("\nHOLDOUT MOVIES:")
for mid, rating in zip(holdout["movieId"].head(10), holdout["rating"].head(10)):
    title_matches = movies[movies["movieId"] == mid]["title"].values
    title = title_matches[0] if len(title_matches) > 0 else "Unknown"
    lang_matches = movies[movies["movieId"] == mid]["language"].values
    lang = lang_matches[0] if len(lang_matches) > 0 else "Unknown"
    print(f"  {title} ({lang}) - {rating}")
