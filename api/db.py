from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./recommender.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_schema() -> None:
    """Create new tables and add only missing columns to the demo SQLite file.

    This lightweight migration keeps the original thesis database usable without
    introducing a migration framework that would be unnecessary for the project.
    """
    from api import models  # Import models only after Base exists.

    models.Base.metadata.create_all(bind=engine)
    additions = {
        "movies": {
            "overview": "TEXT",
            "release_year": "INTEGER",
            "vote_average": "FLOAT",
            "popularity": "FLOAT",
            "poster_path": "VARCHAR",
        },
        "recommendation_logs": {"session_id": "INTEGER"},
    }
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, columns in additions.items():
            present = {column["name"] for column in inspector.get_columns(table)}
            for name, definition in columns.items():
                if name not in present:
                    connection.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
