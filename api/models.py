from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tmdb_id: Mapped[int | None] = mapped_column(unique=True, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    genres: Mapped[str] = mapped_column(String, nullable=False)
    original_language: Mapped[str] = mapped_column(String, nullable=False)
    overview: Mapped[str | None] = mapped_column(Text)
    release_year: Mapped[int | None] = mapped_column(Integer)
    vote_average: Mapped[float | None] = mapped_column(Float)
    popularity: Mapped[float | None] = mapped_column(Float)
    poster_path: Mapped[str | None] = mapped_column(String)


class SyntheticUser(Base):
    __tablename__ = "synthetic_users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String)
    education: Mapped[str | None] = mapped_column(String)
    archetype: Mapped[str | None] = mapped_column(String, index=True)
    preferred_genres: Mapped[str | None] = mapped_column(String)
    preferred_language: Mapped[str | None] = mapped_column(String)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(index=True)
    model_name: Mapped[str | None] = mapped_column(String, index=True)
    movie_id: Mapped[int | None] = mapped_column(Integer)
    rank: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    session_id: Mapped[int | None] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class UserRating(Base):
    """A rating entered during the viva demonstration."""

    __tablename__ = "user_ratings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(index=True)
    movie_id: Mapped[int] = mapped_column(index=True)
    rating: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class RecommendationSession(Base):
    """One saved generation request, including its selected localization weights."""

    __tablename__ = "recommendation_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(index=True)
    trigger: Mapped[str] = mapped_column(String, default="generate")
    weights_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class EvaluationLog(Base):
    """Live Top-10 metrics captured with each recommendation session."""

    __tablename__ = "evaluation_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(index=True)
    user_id: Mapped[int] = mapped_column(index=True)
    model_name: Mapped[str] = mapped_column(String, index=True)
    metrics_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
