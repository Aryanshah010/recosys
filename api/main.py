from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import router
import os

app = FastAPI(title="RecoSys - BSc Thesis Prototype")

os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

app.include_router(router)
