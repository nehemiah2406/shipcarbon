"""POST /api/predict: annual fuel and CO2 for one ship, from the Phase 4 model (ml/predict.py)."""

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from threadpoolctl import threadpool_limits  # installed with scikit-learn

from ml.features import MAX_HOURS_PER_YEAR
from ml.predict import load_bundle, predict_one
from backend.app.routes.stats import merge_one

router = APIRouter(prefix="/api", tags=["prediction"])

EXAMPLE = {"ship_type": "Container ship", "time_at_sea_h": 6000, "tech_eff_type": "EEXI", "tech_eff_value": 9.5,
           "flag": "Liberia", "ice_class": False, "year": 2024}


# All predictions run on one worker thread that uses a single CPU thread. The model has 1,000 trees and, by
# default, splits each tree over every CPU core: for one ship that costs ~100 ms, plus several seconds the first time
# each new request thread starts its own CPU-thread pool. One dedicated thread: ~35 ms, with no slow first call.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict",
                             initializer=lambda: threadpool_limits(1, user_api="openmp"))


def run_prediction(ship: dict) -> dict:
    return _worker.submit(predict_one, ship).result()


def model_loaded() -> bool:
    try:
        load_bundle()
        return True
    except Exception:  # missing or unreadable model file; /api/health reports it
        return False


class PredictIn(BaseModel):
    """What we know about the ship. Only ship_type and time_at_sea_h are required."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    ship_type: str = Field(description='One of the ship types from GET /api/meta, e.g. "Bulk carrier"')
    time_at_sea_h: float = Field(ge=0, le=MAX_HOURS_PER_YEAR, description="Hours at sea in the year (max 8,784)")
    tech_eff_type: Literal["EEDI", "EEXI", "EIV", "Not Applicable"] | None = Field(
        None, description="Technical efficiency index type")
    tech_eff_value: float | None = Field(None, gt=0, le=500, description="Efficiency index value in gCO₂/t·nm")
    flag: str | None = Field(None, description="Flag state; flags outside the top 15 count as 'Other'")
    ice_class: bool = Field(False, description="Does the ship have an ice class?")
    year: int | None = Field(None, ge=2018, le=2030, description="Reporting year (default: latest in the data)")

    @field_validator("ship_type")
    @classmethod
    def known_ship_type(cls, value: str) -> str:
        if not model_loaded():
            return value  # the endpoint answers 503 instead
        known = load_bundle()["ship_types"]
        if merge_one(value) not in known:
            raise ValueError(f"unknown ship type; expected one of: {', '.join(known)}")
        return value


class PredictOut(BaseModel):
    fuel_t: float = Field(description="Predicted fuel burned in the year, tonnes")
    co2_t: float = Field(description="Predicted CO₂, tonnes (fuel × the ship type's CO₂/fuel factor)")
    smoke: float = Field(description="0–1 size of the smoke plume on the website (log scale of CO₂)")


@router.post("/predict", response_model=PredictOut, summary="Predict a ship's annual fuel and CO₂")
def predict(ship: PredictIn):
    if not model_loaded():
        raise HTTPException(status_code=503, detail="The model file is not available on the server.")
    try:
        return run_prediction(ship.model_dump(exclude_none=True))
    except ValueError as err:  # predict_one's own checks; the Pydantic model normally catches these first
        raise HTTPException(status_code=422, detail=str(err)) from err
