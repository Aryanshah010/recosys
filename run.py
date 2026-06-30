import pandas as pd
from api.db import engine, SessionLocal, Base
from api.models import  Movie

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
    
    original_count = len(df)
    df = df.drop_duplicates(subset=['movieId'], keep='first')
    df = df.drop_duplicates(subset=['tmdbId'], keep='first')
    
    duplicates_removed = original_count - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate entries (by movieId or tmdbId) from CSV.")

    db = SessionLocal()
    try:
        if db.query(Movie).count() > 0:
            print("Movies table already populated. Skipping seed.")
            return

        movies_to_add = []
        for _, row in df.iterrows():
            tmdb_val = row.get('tmdbId')
            if pd.notna(tmdb_val):
                tmdb_val = int(tmdb_val)
            else:
                tmdb_val = None
                
            movie = Movie(
                id=int(row['movieId']),
                tmdb_id=tmdb_val,
                title=str(row.get('title_x', row.get('title', 'Unknown'))),
                genres=str(row.get('clean_genres', row.get('genres', ''))),
                original_language=str(row.get('language', 'en')).lower()
            )
            movies_to_add.append(movie)
            
        db.bulk_save_objects(movies_to_add)
        db.commit()
        print(f"Successfully seeded {len(movies_to_add)} unique movies into SQLite.")
    except Exception as e:
        print(f"Error seeding movies: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    seed_movies()