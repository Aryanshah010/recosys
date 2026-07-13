"""FastAPI routes for the interactive thesis demonstration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.db import get_db
from api.models import (
    EvaluationLog,
    RecommendationLog,
    RecommendationSession,
    SyntheticUser,
    UserRating,
)
from api.recommender_service import MODEL_LABELS, get_service

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")
RESULTS_DIR = Path("results")
MODELS = ["cf", "cbf", "hybrid", "localized"]


class GenerationRequest(BaseModel):
    user_id: int
    collaborative: float = Field(default=0.50, ge=0, le=0.9)
    genre: float = Field(default=0.18, ge=0, le=0.9)
    language: float = Field(default=0.12, ge=0, le=0.9)


class RatingRequest(GenerationRequest):
    rating: float = Field(ge=1, le=5)


def _manual_ratings(db: Session, user_id: int) -> list[dict]:
    """Use the latest persisted value per movie while retaining all interactions."""
    ratings = (
        db.query(UserRating)
        .filter(UserRating.user_id == user_id)
        .order_by(UserRating.created_at, UserRating.id)
        .all()
    )
    latest = {row.movie_id: row for row in ratings}
    return [{"movie_id": row.movie_id, "rating": row.rating} for row in latest.values()]


def _previous_recommendations(db: Session, user_id: int) -> dict[str, list[dict]]:
    session = (
        db.query(RecommendationSession)
        .filter(RecommendationSession.user_id == user_id)
        .order_by(RecommendationSession.id.desc())
        .first()
    )
    if session is None:
        return {model: [] for model in MODELS}
    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session.id)
        .order_by(RecommendationLog.model_name, RecommendationLog.rank)
        .all()
    )
    service = get_service()
    previous = {model: [] for model in MODELS}
    for item in logs:
        if item.model_name is None or item.movie_id is None or item.rank is None:
            continue
        model_name = item.model_name
        movie_id = item.movie_id
        if model_name not in previous:
            continue
        previous[model_name].append(
            {
                "movieId": movie_id,
                "rank": item.rank,
                "title": (service.movie(movie_id) or {}).get("title", str(movie_id)),
            }
        )
    return previous


def _change_summary(before: list[dict], after: list[dict]) -> dict:
    old_ranks = {item["movieId"]: item["rank"] for item in before}
    new_ranks = {item["movieId"]: item["rank"] for item in after}
    return {
        "new": [item["title"] for item in after if item["movieId"] not in old_ranks],
        "removed": [
            item.get("title", str(item["movieId"]))
            for item in before
            if item["movieId"] not in new_ranks
        ],
        "moved": [
            {
                "title": item["title"],
                "from": old_ranks[item["movieId"]],
                "to": item["rank"],
            }
            for item in after
            if item["movieId"] in old_ranks
            and old_ranks[item["movieId"]] != item["rank"]
        ],
    }


def _generate(db: Session, payload: GenerationRequest, trigger: str) -> dict:
    svc = get_service()
    if svc.get_user_profile(payload.user_id) is None:
        raise HTTPException(404, "Synthetic user not found.")
    weights = svc.clean_weights(payload.model_dump())
    before = _previous_recommendations(db, payload.user_id)
    manual = _manual_ratings(db, payload.user_id)
    session = RecommendationSession(
        user_id=payload.user_id, trigger=trigger, weights_json=json.dumps(weights)
    )
    db.add(session)
    db.flush()
    results, metrics = {}, {}
    for model in MODELS:
        recs, applied = svc.recommend(
            payload.user_id, model, manual, weights if model == "localized" else None
        )
        live = svc.live_metrics(payload.user_id, recs)
        results[model] = {
            "label": MODEL_LABELS[model],
            "recommendations": recs,
            "weights": applied,
        }
        metrics[model] = live
        for item in recs:
            db.add(
                RecommendationLog(
                    user_id=payload.user_id,
                    session_id=session.id,
                    model_name=model,
                    movie_id=item["movieId"],
                    rank=item["rank"],
                    score=item["score"],
                )
            )
        db.add(
            EvaluationLog(
                session_id=session.id,
                user_id=payload.user_id,
                model_name=model,
                metrics_json=json.dumps(live),
            )
        )
    db.commit()
    return {
        "session_id": session.id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": weights,
        "results": results,
        "metrics": metrics,
        "changes": {
            model: _change_summary(before[model], results[model]["recommendations"])
            for model in MODELS
        },
        "pipeline": [
            "Saving rating" if trigger == "rating" else "Reading saved ratings",
            "Updating SQLite",
            "Updating user profile",
            "Generating user vector",
            "Running Collaborative Filtering",
            "Running Content-Based Filtering",
            "Running SVD predictions",
            "Running Localized Hybrid",
            "Ranking movies",
            "Computing live metrics",
            "Finished",
        ],
    }


@router.get("/", response_class=HTMLResponse)
def landing(request: Request, db: Session = Depends(get_db)):
    users = db.query(SyntheticUser).order_by(SyntheticUser.id).limit(36).all()
    return templates.TemplateResponse(
        request,
        "landing.html",
        {"users": users, "top_movies": get_service().top_movies()},
    )


@router.get("/profile/{user_id}", response_class=HTMLResponse)
def profile(request: Request, user_id: int, db: Session = Depends(get_db)):
    value = get_service().get_user_profile(user_id)
    if value is None:
        raise HTTPException(404, "Synthetic user not found.")
    value["manual_rating_count"] = len(_manual_ratings(db, user_id))
    return templates.TemplateResponse(request, "profile.html", {"profile": value})


@router.get("/dashboard/{user_id}", response_class=HTMLResponse)
def dashboard(request: Request, user_id: int):
    profile_value = get_service().get_user_profile(user_id)
    if profile_value is None:
        raise HTTPException(404, "Synthetic user not found.")
    return templates.TemplateResponse(
        request, "dashboard.html", {"profile": profile_value}
    )


@router.get("/movie/{movie_id}", response_class=HTMLResponse)
def movie_detail(
    request: Request,
    movie_id: int,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
):
    movie = get_service().movie(movie_id)
    profile_value = get_service().get_user_profile(user_id)
    if movie is None or profile_value is None:
        raise HTTPException(404, "Movie or synthetic user not found.")
    current = next(
        (item for item in _manual_ratings(db, user_id) if item["movie_id"] == movie_id),
        None,
    )
    return templates.TemplateResponse(
        request,
        "movie_detail.html",
        {
            "movie": movie,
            "profile": profile_value,
            "current_rating": current["rating"] if current else None,
        },
    )


@router.get("/history/{user_id}", response_class=HTMLResponse)
def history(request: Request, user_id: int, db: Session = Depends(get_db)):
    sessions = (
        db.query(RecommendationSession)
        .filter(RecommendationSession.user_id == user_id)
        .order_by(RecommendationSession.id.desc())
        .limit(12)
        .all()
    )
    return templates.TemplateResponse(
        request, "history.html", {"user_id": user_id, "sessions": sessions}
    )


@router.post("/api/recommendations/generate")
def generate(payload: GenerationRequest, db: Session = Depends(get_db)):
    return _generate(db, payload, "generate")


@router.post("/api/movies/{movie_id}/rate")
def rate_movie(movie_id: int, payload: RatingRequest, db: Session = Depends(get_db)):
    if get_service().movie(movie_id) is None:
        raise HTTPException(404, "Movie not found.")
    db.add(
        UserRating(user_id=payload.user_id, movie_id=movie_id, rating=payload.rating)
    )
    db.commit()
    return _generate(db, payload, "rating")


@router.get("/api/movies/{movie_id}")
def movie_api(movie_id: int):
    movie = get_service().movie(movie_id)
    if movie is None:
        raise HTTPException(404, "Movie not found.")
    return movie


@router.get("/api/recommend/{user_id}")
def recommend_api(
    user_id: int,
    model: str = Query(..., pattern="^(cf|cbf|hybrid|localized)$"),
    db: Session = Depends(get_db),
):
    recs, _ = get_service().recommend(user_id, model, _manual_ratings(db, user_id))
    return {"userId": user_id, "model": model, "results": recs}


def _display_model_names(rows: list[dict]) -> list[dict]:
    for row in rows:
        for key in ("Model", "Model_A", "Model_B"):
            if row.get(key) == "MF_ColdStart":
                row[key] = "CF_ColdStart"
    return rows


@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request):
    perf = RESULTS_DIR / "rq1_model_performance.csv"
    sig = RESULTS_DIR / "rq1_significance_tests.csv"
    return templates.TemplateResponse(
        request,
        "metrics.html",
        {
            "performance": _display_model_names(pd.read_csv(perf).to_dict("records"))
            if perf.exists()
            else [],
            "significance": _display_model_names(pd.read_csv(sig).to_dict("records"))
            if sig.exists()
            else [],
        },
    )


@router.get("/bias", response_class=HTMLResponse)
def bias_page(request: Request):
    div, fb = (
        RESULTS_DIR / "rq2_diversity_by_model.csv",
        RESULTS_DIR / "rq2_filter_bubble_by_archetype.csv",
    )
    return templates.TemplateResponse(
        request,
        "bias.html",
        {
            "diversity": _display_model_names(pd.read_csv(div).to_dict("records"))
            if div.exists()
            else [],
            "filter_bubble": _display_model_names(pd.read_csv(fb).to_dict("records"))
            if fb.exists()
            else [],
        },
    )
