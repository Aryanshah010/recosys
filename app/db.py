import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Define database path inside the root directory
DATABASE_URL = "sqlite:///./recommender.db"

# Create the SQLAlchemy engine
# connect_args={"check_same_thread": False"} is required specifically for SQLite
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Session factory for handling queries
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class that models will inherit from
Base = declarative_base()

def get_db():
    """
    Dependency generator to yield database sessions per request
    and ensure they are safely closed afterward.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()