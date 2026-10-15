"""Predict one ship's annual fuel and CO2 from a plain dict. Used by the API in Phase 5a.

Usage:
    python -m ml.predict

Example:
    >>> predict_one({"ship_type": "Container ship", "time_at_sea_h": 6000, "tech_eff_type": "EEXI",
    ...              "tech_eff_value": 9.5, "flag": "Liberia", "ice_class": None, "year": 2024})
    {'fuel_t': ..., 'co2_t': ..., 'co2_factor': ..., 'smoke': ...}
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.features import MAX_HOURS_PER_YEAR, ROOT, make_features, merge_ship_types

MODEL_DIR = ROOT / "backend" / "app" / "model"
REQUIRED = ["ship_type", "time_at_sea_h"]
OPTIONAL = {"tech_eff_type": None, "tech_eff_value": None, "flag": "Other", "ice_class": None, "year": None}


@lru_cache(maxsize=1)
def load_bundle(model_dir: Path = MODEL_DIR) -> dict:
    bundle = joblib.load(model_dir / "fuel_model.joblib")
    bundle["co2_factors"] = json.loads((model_dir / "co2_factor_by_type.json").read_text(encoding="utf-8"))
    return bundle


def smoke_scale(co2_t: float, p05: float, p99: float) -> float:
    """0..1 on a log scale between the 5th and 99th percentile of yearly CO2, for the hero's smoke plume."""
    x = (math.log(max(co2_t, 1e-9)) - math.log(p05)) / (math.log(p99) - math.log(p05))
    return round(min(max(x, 0.0), 1.0), 3)


def to_row(ship: dict, bundle: dict) -> dict:
    """Check the input and fill optional fields. Raises ValueError with a readable message."""
    missing = [k for k in REQUIRED if ship.get(k) is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    unknown = sorted(set(ship) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}")
    row = {**OPTIONAL, **ship}
    row["ship_type"] = merge_ship_types(pd.Series([row["ship_type"]])).iloc[0]
    if row["ship_type"] not in bundle["ship_types"]:
        raise ValueError(f"Unknown ship_type {ship['ship_type']!r}; expected one of {bundle['ship_types']}")
    hours = float(row["time_at_sea_h"])
    if not 0 <= hours <= MAX_HOURS_PER_YEAR:
        raise ValueError(f"time_at_sea_h must be between 0 and {MAX_HOURS_PER_YEAR}, got {hours}")
    if row["year"] is None:
        row["year"] = bundle["years"][1]
    # ice_class may be a bool or a class name such as "IA Super"; False / None mean no ice class.
    if row["ice_class"] is False:
        row["ice_class"] = None
    return row


def predict_one(ship: dict) -> dict:
    bundle = load_bundle()
    row = to_row(ship, bundle)
    X = make_features(pd.DataFrame([row]), bundle["top_flags"])
    fuel = float(bundle["model"].predict(X)[0])
    factors = bundle["co2_factors"]
    factor = factors["by_type"].get(row["ship_type"], factors["default"])
    co2 = fuel * factor
    return {
        "fuel_t": round(fuel, 1),
        "co2_t": round(co2, 1),
        "co2_factor": factor,
        "smoke": smoke_scale(co2, bundle["smoke"]["co2_p05"], bundle["smoke"]["co2_p99"]),
    }


if __name__ == "__main__":
    example = {"ship_type": "Container ship", "time_at_sea_h": 6000, "tech_eff_type": "EEXI",
               "tech_eff_value": 9.5, "flag": "Liberia", "ice_class": None, "year": 2024}
    print(example)
    print(predict_one(example))
