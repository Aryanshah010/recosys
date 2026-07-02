import os
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise import dump

PROCESSED_DIR = "data/processed"
RATINGS_PATH = os.path.join(PROCESSED_DIR, "ratings_final.csv")
MODEL_PATH = os.path.join(PROCESSED_DIR, "svd_model.pkl")


def train_svd_model():
    print(f"Loading real MovieLens ratings only from: {RATINGS_PATH}")

    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            "Ratings file not found. Please run 'uv run engine/clean_data.py' first."
        )

    df_ratings = pd.read_csv(RATINGS_PATH)

    print(f"Dataset loaded. Total ratings: {len(df_ratings)}")

    reader = Reader(rating_scale=(0.5, 5.0))

    data = Dataset.load_from_df(df_ratings[["userId", "movieId", "rating"]], reader)

    print("Building the training matrix...")
    trainset = data.build_full_trainset()

    print("Training SVD on real ratings only; no synthetic cohort rows are used.")
    algo = SVD(n_factors=150, n_epochs=30, lr_all=0.003, reg_all=0.01, random_state=42)

    algo.fit(trainset)
    print("Training complete!")

    dump.dump(MODEL_PATH, algo=algo)
    print(f"Model successfully saved to: {MODEL_PATH}")


def get_cf_prediction(user_id, movie_id):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_svd_model() first.")

    _, algo = dump.load(MODEL_PATH)

    prediction = algo.predict(uid=user_id, iid=movie_id)
    return prediction.est


if __name__ == "__main__":
    print("Starting Collaborative Filtering Training Pipeline...")
    train_svd_model()

    test_user = 1
    test_movie = 1  # Toy Story
    try:
        est_score = get_cf_prediction(test_user, test_movie)
        print(
            f"\nQuick Test - Predicted rating for User {test_user} on Movie {test_movie}: {est_score:.2f} / 5.0"
        )
    except Exception as e:
        print(f"Skipping test inference (IDs might not exist in downsampled data): {e}")
