from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .db import get_db
from .models import User, Interaction
import pandas as pd
import os
import sys

# Import the ML Engine from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.hybrid_fusion import HybridFusionEngine

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

# 🧠 Initialize the ML Engine once on startup (loads .pkl files into RAM)
print("🔄 Loading Hybrid Fusion Engine into memory...")
rec_engine = HybridFusionEngine()

# 📂 Load Synthetic Cohort for the Login Dropdown
SYNTH_PROFILES_PATH = "data/processed/synthetic_user_profiles.csv"
sample_users = []
if os.path.exists(SYNTH_PROFILES_PATH):
    synth_df = pd.read_csv(SYNTH_PROFILES_PATH)
    sample_users = synth_df.head(30).to_dict('records') # Show 30 users in dropdown
else:
    sample_users = []

# ==========================================
# ROUTES
# ==========================================

@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """Renders the login page with the synthetic cohort dropdown."""
    context = {
        "request": request,
        "users": sample_users
    }
    return templates.TemplateResponse("login.html", context)

@router.post("/login")
async def login(user_id: str = Form(...), db: Session = Depends(get_db)):
    """Handles login. Creates the user in SQLite if they don't exist yet."""
    user = db.query(User).filter(User.username == str(user_id)).first()
    
    if not user:
        # Fetch profile from synthetic CSV
        profile = synth_df[synth_df['user_id'].astype(str) == str(user_id)]
        if profile.empty:
            raise HTTPException(status_code=404, detail="Synthetic user not found in cohort")
        profile = profile.iloc[0]
        
        # Save to SQLite
        user = User(
            username=str(user_id),
            password_hash="synthetic_hash", # Dummy hash for prototype
            preferred_genres=str(profile.get('preferred_genres', '')),
            preferred_languages=str(profile.get('preferred_languages', ''))
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
    return RedirectResponse(url=f"/dashboard?user_id={user.id}", status_code=303)

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user_id: int, db: Session = Depends(get_db)):
    """Generates recommendations using the Hybrid Engine and SQLite history."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found in database")
        
    # 1. Fetch Interaction History from SQLite (The Feedback Loop)
    interactions = db.query(Interaction).filter(Interaction.user_id == user.id).all()
    history = [i.movie_id for i in interactions]
    
    # 2. Prepare Profile for the ML Engine
    user_profile = {
        'preferred_languages': user.preferred_languages,
        'preferred_genres': user.preferred_genres
    }
    
    # 3. Generate Recommendations
    recs = rec_engine.recommend(
        user_id=str(user.username), # SVD expects the original string ID (e.g., '1000269')
        user_profile=user_profile,
        user_history=history,
        k=10
    )
    
    context = {
        "request": request,
        "user": user,
        "recommendations": recs,
        "history_count": len(history)
    }
    return templates.TemplateResponse("dashboard.html", context)

@router.post("/api/interact")
async def log_interaction(
    user_id: int = Form(...), 
    movie_id: int = Form(...), 
    action: str = Form(...), 
    db: Session = Depends(get_db)
):
    """
    CRITICAL THESIS COMPONENT: The User Interaction Loop.
    Logs clicks/likes to SQLite to feed the CF model and update cold-start status.
    """
    # Map UI actions to numeric ratings for Surprise/CF compatibility
    rating_map = {"like": 5.0, "watchlist": 4.0, "click": 3.0, "dislike": 1.0}
    rating = rating_map.get(action, 3.0)
    
    # Prevent duplicate logs
    existing = db.query(Interaction).filter(
        Interaction.user_id == user_id,
        Interaction.movie_id == movie_id,
        Interaction.interaction_type == action
    ).first()
    
    if not existing:
        new_interaction = Interaction(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            interaction_type=action
        )
        db.add(new_interaction)
        db.commit()
        
    return {"status": "success", "message": f"Logged {action} for movie {movie_id}"}
