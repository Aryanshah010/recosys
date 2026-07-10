from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.db import SessionLocal
from api.models import RecommendationLog, SyntheticUser
from api.recommender_service import MODEL_LABELS, get_service

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

RESULTS_DIR = Path("results")
MODELS = ["cf", "cbf", "hybrid", "localized"]
RESULT_MODEL_LABELS = {
    "MF_ColdStart": "CF_ColdStart",  # results generated before the rename
    "CF_ColdStart": "CF_ColdStart",
}


def _display_model_names(rows: list[dict]) -> list[dict]:
    """Present stable website terminology without mutating CSV artifacts."""
    for row in rows:
        if "Model" in row:
            row["Model"] = RESULT_MODEL_LABELS.get(row["Model"], row["Model"])
        for key in ("Model_A", "Model_B"):
            if key in row:
                row[key] = RESULT_MODEL_LABELS.get(row[key], row[key])
    return rows


@router.get("/", response_class=HTMLResponse)
def landing(request: Request):
    db = SessionLocal()
    try:
        users = db.query(SyntheticUser).order_by(SyntheticUser.id).all()
    finally:
        db.close()
    return templates.TemplateResponse(request, "landing.html", {"users": users})


@router.get("/profile/{user_id}", response_class=HTMLResponse)
def profile(request: Request, user_id: int):
    svc = get_service()
    p = svc.get_user_profile(user_id)
    if p is None:
        raise HTTPException(404, "Synthetic user not found.")
    return templates.TemplateResponse(request, "profile.html", {"profile": p})


@router.get("/dashboard/{user_id}", response_class=HTMLResponse)
def dashboard(request: Request, user_id: int):
    svc = get_service()
    p = svc.get_user_profile(user_id)
    if p is None:
        raise HTTPException(404, "Synthetic user not found.")

    db = SessionLocal()
    columns = []
    try:
        for model in MODELS:
            recs = svc.recommend(user_id, model, k=10)
            for r in recs:
                db.add(
                    RecommendationLog(
                        user_id=user_id,
                        model_name=model,
                        movie_id=r["movieId"],
                        rank=r["rank"],
                        score=r["score"],
                    )
                )
            metrics = svc.get_user_metrics(user_id, model)
            columns.append(
                {
                    "key": model,
                    "label": MODEL_LABELS[model],
                    "recs": recs,
                    "metrics": metrics,
                }
            )
        db.commit()
    finally:
        db.close()

    return templates.TemplateResponse(
        request, "dashboard.html", {"profile": p, "columns": columns}
    )


@router.get("/api/recommend/{user_id}")
def recommend_api(
    user_id: int, model: str = Query(..., pattern="^(cf|cbf|hybrid|localized)$")
):
    svc = get_service()
    if svc.get_user_profile(user_id) is None:
        raise HTTPException(404, "Synthetic user not found.")
    return {
        "userId": user_id,
        "model": model,
        "results": svc.recommend(user_id, model, k=10),
    }


@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request):
    perf_path = RESULTS_DIR / "rq1_model_performance.csv"
    sig_path = RESULTS_DIR / "rq1_significance_tests.csv"
    performance = _display_model_names(
        pd.read_csv(perf_path).to_dict("records") if perf_path.exists() else []
    )
    significance = _display_model_names(
        pd.read_csv(sig_path).to_dict("records") if sig_path.exists() else []
    )
    return templates.TemplateResponse(
        request,
        "metrics.html",
        {"performance": performance, "significance": significance},
    )


@router.get("/bias", response_class=HTMLResponse)
def bias_page(request: Request):
    div_path = RESULTS_DIR / "rq2_diversity_by_model.csv"
    fb_path = RESULTS_DIR / "rq2_filter_bubble_by_archetype.csv"
    diversity = _display_model_names(
        pd.read_csv(div_path).to_dict("records") if div_path.exists() else []
    )
    filter_bubble = _display_model_names(
        pd.read_csv(fb_path).to_dict("records") if fb_path.exists() else []
    )
    return templates.TemplateResponse(
        request, "bias.html", {"diversity": diversity, "filter_bubble": filter_bubble}
    )
