from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.db import ensure_schema
from api.recommender_service import get_service
from api.routes import router

app = FastAPI(title="Localized Hybrid Movie Recommender — Thesis Prototype")
app.mount("/static", StaticFiles(directory="api/static"), name="static")
app.include_router(router)


@app.on_event("startup")
def initialise_demo_database() -> None:
    ensure_schema()
    # Force the heavy SVD/CBF/cohort load to happen now, once, in the
    # terminal — rather than silently on whichever page a user opens first.
    get_service()
