from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(title="Localized Hybrid Movie Recommender — Thesis Prototype")
app.mount("/static", StaticFiles(directory="api/static"), name="static")
app.include_router(router)
