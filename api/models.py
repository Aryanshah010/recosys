from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String

from api.db import Base


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    tmdb_id = Column(Integer, nullable=True, unique=True, index=True)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=False)
    original_language = Column(String, nullable=False)


class SyntheticUser(Base):
    __tablename__ = "synthetic_users"

    id = Column(Integer, primary_key=True, index=True)  # == synthetic userId
    age = Column(Integer)
    gender = Column(String)
    education = Column(String)
    archetype = Column(String, index=True)
    preferred_genres = Column(String)
    preferred_language = Column(String)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True)
    model_name = Column(String, index=True)
    movie_id = Column(Integer)
    rank = Column(Integer)
    score = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
