from __future__ import annotations

import json
import time
from datetime import UTC, datetime
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
from api.recommender_service import (
    MODEL_DESCRIPTIONS,
    MODEL_LABELS,
    MODEL_ORDER,
    MODEL_WEIGHTS,
    get_service,
)
from src.cohort_spec import ARCHETYPE_SPECS

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TRACKS = ("real", "synthetic")
TRACK_LABELS = {
    "real": "Real proxy cohort",
    "synthetic": "Synthetic cohort",
}
USERS_PER_ARCHETYPE = 4
NOT_FOUND = "Cohort user not found."


class GenerationRequest(BaseModel):
    user_id: int


class RatingRequest(GenerationRequest):
    rating: float = Field(ge=1, le=5)


def _read_results(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _resolve_track(track: str) -> str:
    return track if track in TRACKS else "real"


def _figure(stem: str) -> str | None:
    
    matches = sorted(FIGURES_DIR.glob(f"*_{stem}.png")) if FIGURES_DIR.exists() else []
    return f"/figures/{matches[0].name}" if matches else None


def _figures(*stems: str) -> list[str]:
    return [src for src in (_figure(s) for s in stems) if src]


def cohort_profile(user_id: int) -> dict:
    """Resolve a cohort user or 404. Used as a dependency by every user page."""
    profile = get_service().get_user_profile(user_id)
    if profile is None:
        raise HTTPException(404, NOT_FOUND)
    return profile


def _manual_ratings(db: Session, user_id: int) -> list[dict]:
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
    previous: dict[str, list[dict]] = {model: [] for model in MODEL_ORDER}
    if session is None:
        return previous

    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session.id)
        .order_by(RecommendationLog.model_name, RecommendationLog.rank)
        .all()
    )
    service = get_service()
    for item in logs:
        if item.model_name not in previous:
            continue
        previous[item.model_name].append(
            {
                "movieId": item.movie_id,
                "rank": item.rank,
                "title": service.title(item.movie_id),
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
        raise HTTPException(404, "Cohort user not found.")

    stages: list[dict] = []

    def stage(name: str, fn):
        started = time.perf_counter()
        value = fn()
        stages.append(
            {"stage": name, "ms": round((time.perf_counter() - started) * 1000, 1)}
        )
        return value

    before = stage(
        "Loading previous session",
        lambda: _previous_recommendations(db, payload.user_id),
    )
    manual = stage(
        "Reading persisted ratings", lambda: _manual_ratings(db, payload.user_id)
    )

    session = RecommendationSession(user_id=payload.user_id, trigger=trigger)
    db.add(session)
    db.flush()
    session_id = session.id

    results, metrics = {}, {}
    for model in MODEL_ORDER:
        recs = stage(
            f"Scoring {MODEL_LABELS[model]}",
            lambda model=model: svc.recommend(payload.user_id, model, manual),
        )
        live = svc.live_metrics(payload.user_id, recs)
        results[model] = {
            "label": MODEL_LABELS[model],
            "description": MODEL_DESCRIPTIONS[model],
            "weights": MODEL_WEIGHTS[model],
            "recommendations": recs,
        }
        metrics[model] = live

        db.add_all(
            RecommendationLog(
                session_id=session_id,
                user_id=payload.user_id,
                model_name=model,
                movie_id=item["movieId"],
                rank=item["rank"],
                score=item["score"],
            )
            for item in recs
        )
        db.add(
            EvaluationLog(
                session_id=session_id,
                user_id=payload.user_id,
                model_name=model,
                metrics_json=json.dumps(live),
            )
        )

    stage("Persisting session", db.commit)

    return {
        "session_id": session_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "trigger": trigger,
        "results": results,
        "metrics": metrics,
        "changes": {
            model: _change_summary(before[model], results[model]["recommendations"])
            for model in MODEL_ORDER
        },
        "pipeline": stages,
    }


@router.get("/", response_class=HTMLResponse)
def landing(request: Request, db: Session = Depends(get_db)):
    groups = [
        {
            "archetype": spec["name"],
            "share": round(spec["target_share"] * 100),
            "language": spec["language"],
            "users": db.query(SyntheticUser)
            .filter(SyntheticUser.archetype == spec["name"])
            .order_by(SyntheticUser.id)
            .limit(USERS_PER_ARCHETYPE)
            .all(),
        }
        for spec in sorted(
            ARCHETYPE_SPECS, key=lambda s: s["target_share"], reverse=True
        )
    ]
    return templates.TemplateResponse(request, "landing.html", {"groups": groups})


@router.get("/profile/{user_id}", response_class=HTMLResponse)
def profile(
    request: Request,
    user_id: int,
    value: dict = Depends(cohort_profile),
    db: Session = Depends(get_db),
):
    value["manual_rating_count"] = len(_manual_ratings(db, user_id))
    sessions = (
        db.query(RecommendationSession)
        .filter(RecommendationSession.user_id == user_id)
        .order_by(RecommendationSession.id.desc())
        .limit(8)
        .all()
    )
    return templates.TemplateResponse(
        request, "profile.html", {"profile": value, "sessions": sessions}
    )


@router.get("/dashboard/{user_id}", response_class=HTMLResponse)
def dashboard(request: Request, value: dict = Depends(cohort_profile)):
    return templates.TemplateResponse(request, "dashboard.html", {"profile": value})


@router.get("/movie/{movie_id}", response_class=HTMLResponse)
def movie_detail(
    request: Request,
    movie_id: int,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
):
    svc = get_service()
    profile_value = svc.get_user_profile(user_id)
    if profile_value is None:
        raise HTTPException(404, NOT_FOUND)
    svc.warm_posters([movie_id])
    movie = svc.movie(movie_id)
    if movie is None:
        raise HTTPException(404, "Movie not found.")
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


def _track_context(track: str) -> dict:
    return {
        "track": track,
        "track_label": TRACK_LABELS[track],
        "tracks": TRACKS,
        "track_labels": TRACK_LABELS,
    }


@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request, track: str = Query("real")):
    track = _resolve_track(track)
    return templates.TemplateResponse(
        request,
        "metrics.html",
        _track_context(track)
        | {
            "performance": _read_results(f"rq1_model_performance_{track}.csv"),
            "headline": _read_results(f"rq1_headline_effect_{track}.csv"),
            "significance": _read_results(f"rq1_significance_tests_{track}.csv"),
            "intervals": _read_results(f"rq1_confidence_intervals_{track}.csv"),
            "figure": _figure(f"rq1_performance_{track}"),
        },
    )


@router.get("/bias", response_class=HTMLResponse)
def bias_page(request: Request, track: str = Query("real")):
    track = _resolve_track(track)
    return templates.TemplateResponse(
        request,
        "bias.html",
        _track_context(track)
        | {
            "diversity": _read_results(f"rq2_diversity_by_model_{track}.csv"),
            "filter_bubble": _read_results(
                f"rq2_filter_bubble_by_archetype_{track}.csv"
            ),
            "by_archetype": _read_results(f"rq2_performance_by_archetype_{track}.csv"),
            "sweep": _read_results(f"sensitivity_lambda_summary_{track}.csv"),
            "figures": _figures(
                f"rq2_diversity_{track}",
                f"rq2_bubble_by_archetype_{track}",
                f"tradeoff_curve_{track}",
            ),
        },
    )


@router.get("/cohort", response_class=HTMLResponse)
def cohort_page(request: Request):
    affinity = _read_results("sensitivity_affinity_tables.csv")
    benchmark = _read_results("benchmark_population_profile.csv")

    # Derived, not narrated. These shares were previously hardcoded in the
    # template and would silently go stale on every pipeline re-run.
    shares = {
        str(row["Archetype"]): float(row["Benchmark_Share_%"])
        for row in benchmark
        if "Benchmark_Share_%" in row
    }
    minority = [
        spec["name"]
        for spec in ARCHETYPE_SPECS
        if spec["language"] not in (None, "English")
    ]

    return templates.TemplateResponse(
        request,
        "cohort.html",
        {
            "benchmark": benchmark,
            "sampling": _read_results("real_cohort_sampling_audit.csv"),
            "languages": [r for r in affinity if r.get("Type") == "language"],
            "genres": [r for r in affinity if r.get("Type") == "genre"],
            "catalogue_size": get_service().catalogue_size,
            "english_share": round(shares["hollywood"], 2) if shares else None,
            "cohort_share": (
                round(sum(shares.get(name, 0.0) for name in minority), 2)
                if shares
                else None
            ),
            "figures": _figures("benchmark_representation_gap", "track_comparison"),
        },
    )
