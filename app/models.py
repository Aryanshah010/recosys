from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Onboarding Preferences Layer (Addresses Cold-Start Problem)
    # Comma-separated strings or JSON arrays containing genres selected during registration
    # e.g., "Action,Sci-Fi,Thriller"
    preferred_genres = Column(Text, nullable=True) 
    preferred_languages = Column(String(100), nullable=True, default="en")

    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")


class Movie(Base):
    __tablename__ = "movies"

    # Important: 'id' maps directly to MovieLens' 'movieId' to maintain a strict 
    # identifier bridge with your ratings_clean.csv training set.
    id = Column(Integer, primary_key=True, index=True, autoincrement=False)
    tmdb_id = Column(Integer, unique=True, nullable=True, index=True)
    title = Column(String(255), nullable=False)
    genres = Column(String(255), nullable=False)  # Stored as raw string text: "Action|Adventure|Sci-Fi"
    release_year = Column(Integer, nullable=True)
    
    # Metadata for lazy loading & localized weighting logic
    poster_path = Column(String(255), nullable=True)
    overview = Column(Text, nullable=True)
    original_language = Column(String(10), nullable=True, default="en")

    # Relationships
    interactions = relationship("Interaction", back_populates="movie", cascade="all, delete-orphan")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    
    # surprise compatibility: numeric mapping is mandatory for Collaborative Filtering.
    # If user clicks "Like", assign 5.0. If "Dislike", assign 1.0. 
    rating = Column(Float, nullable=False, default=5.0)
    
    # Metric segmentation: 'like', 'watchlist', or 'click'
    interaction_type = Column(String(20), nullable=False, default="like")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="interactions")
    movie = relationship("Movie", back_populates="interactions")


class EvaluationMetric(Base):
    """
    Thesis Analytics Table: Populates your /admin dashboard comparison grid.
    Allows you to query data directly for thesis documentation tables and charts.
    """
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)  # 'CF_Only', 'Content_Only', 'Non_Localized_Hybrid', 'Localized_Hybrid'
    k_value = Column(Integer, nullable=False, default=10)  # The 'K' in Precision@K (e.g., 5, 10, 20)
    precision_at_k = Column(Float, nullable=False)
    recall_at_k = Column(Float, nullable=False)
    ndcg = Column(Float, nullable=False)
    rmse = Column(Float, nullable=True)  # Matrix factorization performance metric
    calculated_at = Column(DateTime, default=datetime.utcnow)