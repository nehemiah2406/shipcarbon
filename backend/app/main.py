"""ShipCarbon API.

Run locally from the project root:
    uvicorn backend.app.main:app --reload

Then open http://127.0.0.1:8000 for the website, or http://127.0.0.1:8000/docs for the API documentation.
"""

import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo.database import Database

from backend.app.db import ensure_indexes, get_db, is_reachable
from backend.app.routes import predict, stats
from ml.features import ROOT

LOCAL_FRONTEND = "http://localhost:5173"  # Vite dev server
FRONTEND_DIR = ROOT / "frontend"  # the website (plain HTML, CSS and JavaScript), served at /


def allowed_origins() -> list[str]:
    """The local website plus any comma-separated origins in FRONTEND_ORIGINS (e.g. the Vercel URL)."""
    extra = [o.strip().rstrip("/") for o in os.getenv("FRONTEND_ORIGINS", "").split(",") if o.strip()]
    return [LOCAL_FRONTEND] + [o for o in extra if o != LOCAL_FRONTEND]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On start-up: load the model, make sure the indexes exist and fill the statistics cache in the background.
    # None of this stops the server if it fails; /api/health shows what is missing.
    if predict.model_loaded():
        predict.run_prediction(predict.EXAMPLE)  # the first prediction is slower; do it now instead of for a visitor
    db = app.dependency_overrides.get(get_db, get_db)()
    if is_reachable(db):
        ensure_indexes(db)
        if os.getenv("WARM_CACHE", "1") == "1":
            threading.Thread(target=stats.warm_cache, args=(db,), daemon=True).start()
    yield


app = FastAPI(
    title="ShipCarbon API",
    description="CO₂ from large ships calling at EU/EEA ports (EU MRV data, EMSA THETIS-MRV), and a model that "
                "predicts a ship's annual fuel consumption.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])
app.include_router(stats.router)
app.include_router(predict.router)


@app.get("/api/health", tags=["status"], summary="Is the API working? Database and model status")
def health(db: Database = Depends(get_db)):
    database, model = is_reachable(db), predict.model_loaded()
    return {
        "status": "ok" if database and model else "degraded",
        "database": database,
        "model": model,
        "data_versions": stats.data_versions(),
    }


# Mounted last so /docs, /openapi.json and every /api route are matched first.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="website")
