from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.db import SessionLocal, ensure_schema
from api.models import SyntheticUser
from api.recommender_service import get_service
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("results/figures")


def _seed_if_empty() -> None:
    db = SessionLocal()
    try:
        if db.query(SyntheticUser).count() > 0:
            return
    finally:
        db.close()

    from seed_db import seed_synthetic_users

    logger.info("Cohort table is empty; seeding from the processed CSV.")
    seed_synthetic_users()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_schema()
    _seed_if_empty()
    get_service()
    yield


app = FastAPI(
    title="recosys - Localized Hybrid Movie Recommender",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="api/static"), name="static")
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
app.include_router(router)
