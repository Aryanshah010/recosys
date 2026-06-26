from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import router
import os

app = FastAPI(title="RecoSys - BSc Thesis Prototype")

# Create static folder if it doesn't exist (for CSS/JS/Images)
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Include API and Page routes
app.include_router(router)