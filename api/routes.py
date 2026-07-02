from pathlib import Path
import csv
import pandas as pd
from fastapi import (
    APIRouter,
    Request,
    Depends,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .db import get_db
from .models import User, Interaction, Movie
from engine.hybrid_fusion import HybridFusionEngine

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "api" / "templates"
SYNTH_PROFILES_PATH = BASE_DIR / "data" / "processed" / "synthetic_user_profiles.csv"
SYNTH_INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "synthetic_interactions.csv"
EVAL_RESULTS_PATH = BASE_DIR / "results" / "thesis_evaluation_metrics.csv"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

print("Loading Hybrid Fusion Engine into memory...")
rec_engine = HybridFusionEngine()
print("Hybrid Fusion Engine Ready.")

synth_df = None
synth_interactions_df = None
sample_users = []

if SYNTH_PROFILES_PATH.exists():
    print("Loading synthetic user profiles...")
    synth_df = pd.read_csv(SYNTH_PROFILES_PATH)

    if "user_id" in synth_df.columns:
        synth_df["user_id"] = synth_df["user_id"].astype(str)

    sample_users = synth_df.head(30).to_dict("records")
    print(f"Loaded {len(synth_df)} synthetic users")

else:
    print(f"Synthetic profile file not found: {SYNTH_PROFILES_PATH}")

if SYNTH_INTERACTIONS_PATH.exists():
    print("Loading synthetic evaluation interactions...")
    synth_interactions_df = pd.read_csv(SYNTH_INTERACTIONS_PATH)
    if "userId" in synth_interactions_df.columns:
        synth_interactions_df["userId"] = synth_interactions_df["userId"].astype(str)
    print(f"Loaded {len(synth_interactions_df)} synthetic interactions")
else:
    print(f"Synthetic interaction file not found: {SYNTH_INTERACTIONS_PATH}")


def artifact_message():
    if rec_engine.is_ready:
        return None
    missing = ", ".join(rec_engine.missing_artifacts) or "processed artifacts"
    return f"Run the master pipeline first: uv run python main.py. Missing: {missing}"


def get_synthetic_profile(user_id):
    if synth_df is None:
        return None
    profile = synth_df[synth_df["user_id"].astype(str) == str(user_id)]
    if profile.empty:
        return None
    return profile.iloc[0]


def get_synthetic_support_history(user_id):
    if synth_interactions_df is None:
        return []

    rows = synth_interactions_df[synth_interactions_df["userId"].astype(str) == str(user_id)]
    if rows.empty:
        return []

    rows = rows.sort_values(["rating", "movieId"], ascending=[False, True])
    positives = rows[rows["rating"] >= 4.0]
    if len(positives) >= 2:
        holdout_count = max(1, int(round(len(positives) * 0.3)))
        holdout_indexes = set(positives.tail(holdout_count).index)
        rows = rows[~rows.index.isin(holdout_indexes)]

    return rows["movieId"].astype(int).tolist()


@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"users": sample_users, "artifact_error": artifact_message()},
    )


@router.post("/login")
async def login(
    user_id: str = Form(...),
    db: Session = Depends(get_db),
):
    user_id = user_id

    user = db.query(User).filter(User.username == user_id).first()

    if user is None:
        if synth_df is None:
            raise HTTPException(
                status_code=500,
                detail="Synthetic profiles not loaded",
            )

        profile = synth_df[synth_df["user_id"] == user_id]

        if profile.empty:
            raise HTTPException(
                status_code=404,
                detail="Synthetic user not found",
            )

        profile = profile.iloc[0]

        user = User(
            username=user_id,
            password_hash="synthetic_hash",
            preferred_genres=str(profile.get("preferred_genres", "")),
            preferred_languages=str(profile.get("preferred_languages", "")),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

    return RedirectResponse(
        url=f"/dashboard?user_id={user.id}",
        status_code=303,
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"artifact_error": artifact_message()},
    )


@router.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    db: Session = Depends(get_db),
):
    form = await request.form()
    languages = form.getlist("languages")
    genres = form.getlist("genres")

    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        return RedirectResponse(
            url=f"/dashboard?user_id={existing.id}",
            status_code=303,
        )

    preferred_languages = (
        "|".join([x for x in languages if isinstance(x, str)]) if languages else "EN"
    )
    preferred_genres = (
        "|".join([x for x in genres if isinstance(x, str)])
        if genres
        else "Action|Drama"
    )

    user = User(
        username=username,
        password_hash="user_registered",
        preferred_languages=preferred_languages,
        preferred_genres=preferred_genres,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(
        url=f"/dashboard?user_id={user.id}",
        status_code=303,
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    interactions = db.query(Interaction).filter(Interaction.user_id == user.id).all()

    history = [interaction.movie_id for interaction in interactions]

    user_profile = {
        "preferred_languages": user.preferred_languages,
        "preferred_genres": user.preferred_genres,
    }

    try:
        recs = rec_engine.recommend(
            user_id=str(user.username),
            user_profile=user_profile,
            user_history=history,
            k=10,
        )

    except Exception as e:
        print(f"Recommendation error: {e}")
        recs = []

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "user": user,
            "recommendations": recs,
            "history_count": len(history),
            "artifact_error": artifact_message(),
        },
    )


@router.post("/api/interact")
async def log_interaction(
    user_id: int = Form(...),
    movie_id: int = Form(...),
    action: str = Form(...),
    db: Session = Depends(get_db),
):
    rating_map = {
        "like": 5.0,
        "watchlist": 4.0,
        "click": 3.0,
        "dislike": 1.0,
    }

    rating = rating_map.get(action, 3.0)

    existing = (
        db.query(Interaction)
        .filter(
            Interaction.user_id == user_id,
            Interaction.movie_id == movie_id,
        )
        .first()
    )

    if existing is None:
        interaction = Interaction(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            interaction_type=action,
        )
        db.add(interaction)
        db.commit()

    return {
        "status": "success",
        "message": (f"Logged {action} for movie {movie_id}"),
    }


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"artifact_error": artifact_message()},
    )


@router.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"artifact_error": artifact_message()},
    )


@router.get("/comparison", response_class=HTMLResponse)
async def comparison_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="comparison.html",
        context={
            "users": sample_users,
            "artifact_error": artifact_message(),
            "has_synthetic_data": synth_df is not None,
        },
    )


@router.get("/architecture", response_class=HTMLResponse)
async def architecture_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="architecture.html",
        context={"artifact_error": artifact_message()},
    )


@router.get("/api/admin/stats")
async def admin_stats(db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    total_interactions = db.query(Interaction).count()
    total_movies = db.query(Movie).count()
    total_likes = (
        db.query(Interaction).filter(Interaction.interaction_type == "like").count()
    )

    return JSONResponse(
        {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "total_movies": total_movies,
            "total_likes": total_likes,
        }
    )


@router.get("/api/admin/metrics")
async def admin_metrics():
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run the master pipeline first.",
        )

    raw_rows = []
    with open(EVAL_RESULTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(
                {
                    "Model": row.get("Model", ""),
                    "Archetype": row.get("Archetype", ""),
                    "Precision@10": float(row.get("Precision@10", 0)),
                    "Recall@10": float(row.get("Recall@10", 0)),
                    "NDCG@10": float(row.get("NDCG@10", 0)),
                    "Filter_Bubble_Score": float(row.get("Filter_Bubble_Score", 0)),
                    "Language_Diversity": float(row.get("Language_Diversity", 0)),
                    "Genre_Diversity": float(row.get("Genre_Diversity", 0)),
                }
            )

    return JSONResponse(
        {
            "raw": raw_rows,
            "summary": raw_rows,
        }
    )


@router.post("/api/comparison/all-models", response_class=JSONResponse)
async def comparison_all_models(user_id: str = Form(...)):
    if not rec_engine.is_ready:
        raise HTTPException(status_code=503, detail=artifact_message())

    profile = get_synthetic_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Synthetic user not found")

    user_profile = {
        "preferred_languages": str(profile.get("preferred_languages", "")),
        "preferred_genres": str(profile.get("preferred_genres", "")),
    }
    support_history = get_synthetic_support_history(user_id)

    try:
        models = rec_engine.compare_models(
            user_id=user_id,
            user_profile=user_profile,
            user_history=support_history,
            k=10,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(
        {
            "status": "success",
            "user_id": user_id,
            "archetype": str(profile.get("cohort_group", "Unknown")),
            "preferences": user_profile,
            "support_history_count": len(support_history),
            "models": models,
        }
    )
