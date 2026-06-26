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

    preferred_genres = Column(Text, nullable=True) 
    preferred_languages = Column(String(100), nullable=True, default="en")

    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")


class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True, autoincrement=False)
    tmdb_id = Column(Integer, unique=True, nullable=True, index=True)
    title = Column(String(255), nullable=False)
    genres = Column(String(255), nullable=False)  
    release_year = Column(Integer, nullable=True)
    
    poster_path = Column(String(255), nullable=True)
    overview = Column(Text, nullable=True)
    original_language = Column(String(10), nullable=True, default="en")
    interactions = relationship("Interaction", back_populates="movie", cascade="all, delete-orphan")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)

    rating = Column(Float, nullable=False, default=5.0)
    
    interaction_type = Column(String(20), nullable=False, default="like")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")
    movie = relationship("Movie", back_populates="interactions")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)  
    k_value = Column(Integer, nullable=False, default=10)  
    precision_at_k = Column(Float, nullable=False)
    recall_at_k = Column(Float, nullable=False)
    ndcg = Column(Float, nullable=False)
    rmse = Column(Float, nullable=True)  
    calculated_at = Column(DateTime, default=datetime.utcnow)