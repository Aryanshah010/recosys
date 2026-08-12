import argparse
import logging
import sys

import pandas as pd

from api.db import SessionLocal, ensure_schema
from api.models import SyntheticUser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

USERS_CSV = "data/processed/synthetic_users.csv"


def seed_synthetic_users(force: bool = False) -> int:
    ensure_schema()

    try:
        df = pd.read_csv(USERS_CSV)
    except FileNotFoundError:
        logger.error("%s not found. Run `python main.py` first.", USERS_CSV)
        return 0

    db = SessionLocal()
    try:
        existing = db.query(SyntheticUser).count()
        if existing > 0 and not force:
            logger.info("synthetic_users already holds %d rows; skipping.", existing)
            return existing

        if existing > 0:
            db.query(SyntheticUser).delete()
            db.commit()
            logger.info("Cleared %d existing rows for re-seed.", existing)

        db.bulk_save_objects(
            [
                SyntheticUser(
                    id=int(row.userId),  # type: ignore
                    age=int(row.age),  # type: ignore
                    gender=str(row.gender),
                    education=str(row.education),
                    archetype=str(row.archetype),
                    preferred_genres=str(row.preferred_genres),
                    preferred_language=str(row.preferred_language),
                )
                for row in df.itertuples()
            ]
        )
        db.commit()
        logger.info("Seeded %d cohort users into SQLite.", len(df))
        return len(df)
    except Exception as exc:
        logger.error("Error seeding synthetic_users: %s", exc)
        db.rollback()
        return 0
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the demo SQLite database from the processed cohort CSV."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear and re-seed a table that already contains data.",
    )
    args = parser.parse_args()

    if seed_synthetic_users(force=args.force) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
