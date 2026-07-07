import pandas as pd
from api.db import engine, SessionLocal, Base
from api.models import Movie


def init_db():
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database schema created successfully.")


def seed_movies():
    print("Seeding movies from movies_final.csv...")
    csv_path = "data/processed/movies_final.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("movies_final.csv not found. Run the master pipeline first.")
        return

    has_tmdb = df["tmdbId"].notna()
    df = pd.concat(
        [
            df[has_tmdb].drop_duplicates(subset=["tmdbId"], keep="first"),
            df[~has_tmdb],
        ],
        ignore_index=True,
    )

    db = SessionLocal()
    try:
        if db.query(Movie).count() > 0:
            print("Movies table already populated. Skipping seed.")
            return

        movies_to_add = []
        for _, row in df.iterrows():
            tmdb_val = row["tmdbId"]
            movie = Movie(
                id=int(row["movieId"]),
                tmdb_id=int(tmdb_val) if pd.notna(tmdb_val) else None,
                title=str(row["title"]),
                genres=str(row["clean_genres"]),
                original_language=str(row["language"]).lower(),
            )
            movies_to_add.append(movie)

        db.bulk_save_objects(movies_to_add)
        db.commit()
        print(f"Successfully seeded {len(movies_to_add)} movies into SQLite.")
    except Exception as e:
        print(f"Error seeding movies: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    seed_movies()
