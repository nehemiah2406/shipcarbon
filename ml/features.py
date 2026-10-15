"""Shared data preparation for the fuel model. Used by ml/train.py, ml/predict.py and (later) the API.

Leakage rule: a feature may only describe the ship and how long it sailed. Anything measured from or calculated
from fuel (CO2, CH4, N2O, CO2eq, per-distance or per-transport-work figures, distance) is banned, because
CO2 is roughly fuel x 3.14 and would hand the model the answer.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIP_YEARS_CSV = ROOT / "data" / "processed" / "ship_years.csv"

TARGET = "fuel_t"
CATEGORICAL = ["ship_type", "tech_eff_type", "flag_group"]
NUMERIC = ["tech_eff_value", "time_at_sea_h", "ice_class_yes", "year"]
FEATURES = CATEGORICAL + NUMERIC
N_TOP_FLAGS = 15
MAX_HOURS_PER_YEAR = 8784  # 366 days; 19 ship-years report more (up to 603 million h), which are data errors

# Any column whose name contains one of these words is never allowed as a feature.
BANNED_WORDS = ["fuel", "co2", "ch4", "n2o", "per_nm", "distance", "transport_work"]

# EMSA relabelled most passenger ships in 2025; treat both labels as one type (same as the EDA).
SHIP_TYPE_MERGE = {"Passenger ship (Cruise Passenger ship)": "Passenger ship"}


def check_no_leakage(features: list[str]) -> None:
    banned = [f for f in features if any(word in f.lower() for word in BANNED_WORDS)]
    if banned:
        raise ValueError(f"Leakage: these features are computed from fuel or are the target: {banned}")


def merge_ship_types(ship_type: pd.Series) -> pd.Series:
    return ship_type.replace(SHIP_TYPE_MERGE)


def load_training_data(path: Path = SHIP_YEARS_CSV) -> pd.DataFrame:
    """Ship-years from full reports with fuel > 0."""
    df = pd.read_csv(path, dtype={"imo": str}, keep_default_na=False, na_values=[""])
    df = df[(df["source"] == "full") & (df[TARGET] > 0)].copy()
    df["ship_type"] = merge_ship_types(df["ship_type"])
    return df.reset_index(drop=True)


def top_flags(df: pd.DataFrame, n: int = N_TOP_FLAGS) -> list[str]:
    """The n flags with the most ship-years (Unknown excluded); every other flag becomes 'Other'."""
    counts = df.loc[df["flag"] != "Unknown", "flag"].value_counts()
    return counts.head(n).index.tolist()


def make_features(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """Turn ship-year rows (or API inputs with the same column names) into the model's feature table."""
    X = pd.DataFrame(index=df.index)
    X["ship_type"] = merge_ship_types(df["ship_type"].astype(str))
    X["tech_eff_type"] = df["tech_eff_type"].fillna("Missing").astype(str)
    X["flag_group"] = np.where(df["flag"].isin(flags), df["flag"], "Other")
    X["tech_eff_value"] = pd.to_numeric(df["tech_eff_value"], errors="coerce")
    hours = pd.to_numeric(df["time_at_sea_h"], errors="coerce")
    X["time_at_sea_h"] = hours.where(hours.between(0, MAX_HOURS_PER_YEAR))  # impossible values -> unknown
    ice = df["ice_class"]
    X["ice_class_yes"] = (ice.notna() & ~ice.astype(str).str.strip().isin(["", "False", "false", "0", "No", "no"])).astype(int)
    X["year"] = pd.to_numeric(df["year"], errors="coerce")
    check_no_leakage(list(X.columns))
    return X[FEATURES]
