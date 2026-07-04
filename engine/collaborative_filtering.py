from __future__ import annotations

import logging
import os

import pandas as pd
from surprise import SVD, Dataset, Reader, dump
from surprise.prediction_algorithms.predictions import Prediction
from surprise.trainset import Trainset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
RATINGS_PATH = os.path.join(PROCESSED_DIR, "ratings_final.csv")
MODEL_PATH = os.path.join(PROCESSED_DIR, "svd_model.pkl")

REQUIRED_RATING_COLUMNS = ["userId", "movieId", "rating"]
RATING_SCALE = (0.5, 5.0)

N_FACTORS = 100
N_EPOCHS = 20
LR_ALL = 0.005
REG_ALL = 0.02
RANDOM_STATE = 42

RATINGS_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32"}


def check_file_exists(path: str) -> None:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Run clean_data.py first to produce ratings_final.csv."
        )


def load_ratings(path: str = RATINGS_PATH) -> pd.DataFrame:

    check_file_exists(path)
    logger.info("Loading ratings from %s...", path)
    ratings_df = pd.read_csv(path)

    missing = [c for c in REQUIRED_RATING_COLUMNS if c not in ratings_df.columns]
    if missing:
        raise ValueError(
            f"ratings_final.csv is missing required columns: {missing}. "
            "This script only consumes the schema produced by clean_data.py."
        )

    if ratings_df[REQUIRED_RATING_COLUMNS].isna().any().any():
        raise ValueError(
            "ratings_final.csv contains null values in userId, movieId, "
            "or rating. This should have been caught by clean_data.py's "
            "validation."
        )

    if not ratings_df["rating"].between(*RATING_SCALE).all():
        raise ValueError(
            f"ratings_final.csv contains rating values outside the valid "
            f"range {RATING_SCALE}, which indicates corrupted data."
        )

    ratings_df = ratings_df.astype(RATINGS_DTYPES)

    logger.info("Loaded %d ratings.", len(ratings_df))
    return ratings_df


def build_surprise_dataset(ratings_df: pd.DataFrame) -> Dataset:

    reader = Reader(rating_scale=RATING_SCALE)
    return Dataset.load_from_df(ratings_df[REQUIRED_RATING_COLUMNS], reader)


def train_svd_model(data: Dataset) -> SVD:

    logger.info(
        "Training SVD with n_factors=%d, n_epochs=%d, lr_all=%s, reg_all=%s, "
        "random_state=%d",
        N_FACTORS,
        N_EPOCHS,
        LR_ALL,
        REG_ALL,
        RANDOM_STATE,
    )

    trainset: Trainset = data.build_full_trainset()  # type: ignore
    algo = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        lr_all=LR_ALL,
        reg_all=REG_ALL,
        random_state=RANDOM_STATE,
    )
    algo.fit(trainset)

    logger.info(
        "Training complete: %d users, %d items, %d ratings.",
        trainset.n_users,
        trainset.n_items,
        trainset.n_ratings,
    )
    return algo


def validate_model(algo: SVD) -> None:

    trainset = getattr(algo, "trainset", None)
    if trainset is None:
        raise ValueError("Trained SVD model has no attached trainset.")

    if trainset.n_users == 0 or trainset.n_items == 0:
        raise ValueError(
            f"Trained SVD model has {trainset.n_users} users and "
            f"{trainset.n_items} items -- training data was effectively empty."
        )


def save_model(algo: SVD, path: str = MODEL_PATH) -> None:

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dump.dump(path, algo=algo)
    logger.info("Wrote %s.", path)


def load_model(path: str = MODEL_PATH) -> SVD:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_svd_model() first."
        )
    _, algo = dump.load(path)
    return algo


def predict_rating(algo: SVD, user_id: int, movie_id: int) -> Prediction:

    return algo.predict(uid=user_id, iid=movie_id)


def is_known(algo: SVD, user_id: int, movie_id: int) -> tuple[bool, bool]:

    try:
        algo.trainset.to_inner_uid(user_id)
        user_known = True
    except ValueError:
        user_known = False

    try:
        algo.trainset.to_inner_iid(movie_id)
        item_known = True
    except ValueError:
        item_known = False

    return user_known, item_known


def main() -> None:

    logger.info("Starting collaborative_filtering.py pipeline...")

    ratings_df = load_ratings()
    data = build_surprise_dataset(ratings_df)
    algo = train_svd_model(data)
    validate_model(algo)
    save_model(algo)

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
