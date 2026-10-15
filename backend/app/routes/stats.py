"""Read-only statistics endpoints, computed by MongoDB aggregation pipelines on the ship_years collection.

ship_years has one row per ship (IMO) per year, so partial reports are never double counted. EMSA's 2025 passenger
label is merged with the old one using the same mapping as the EDA and the model (ml.features.SHIP_TYPE_MERGE).

Every data endpoint takes the optional filters year, ship_type and flag. Totals are in tonnes.

Speed: the data only changes when pipeline/load_mongo.py runs, so answers are cached in memory for CACHE_SECONDS.
A full pass over all ship-years takes 100-400 ms locally; a cached answer takes a few ms. main.py fills the cache
for the most common views (no filter, and each single year) in the background when the server starts.
"""

import json
import time

import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from pymongo.database import Database

from backend.app.db import get_db
from ml.features import ROOT, SHIP_TYPE_MERGE, merge_ship_types
from pipeline.clean import SHIP_GROUPS

router = APIRouter(prefix="/api", tags=["statistics"])

SHIP_GROUP_ORDER = ["Container", "Tanker", "Bulk", "LNG", "Ro-pax", "Other"]
MANIFEST = ROOT / "data" / "raw" / "manifest.json"
CO2EQ_FIRST_YEAR = 2024  # EMSA publishes CH4, N2O and CO2eq only from reporting year 2024
CACHE_SECONDS = 600
_cache: dict[str, tuple[float, list[dict]]] = {}

# MongoDB expression: the ship type with EMSA's 2025 relabelling undone.
MERGED_SHIP_TYPE = {"$switch": {
    "branches": [{"case": {"$eq": ["$ship_type", old]}, "then": new} for old, new in SHIP_TYPE_MERGE.items()],
    "default": "$ship_type",
}}


def merge_one(ship_type: str) -> str:
    return merge_ship_types(pd.Series([ship_type])).iloc[0]


def data_versions() -> dict[str, int]:
    """EMSA file version per reporting year, as downloaded in Phase 1."""
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return {year: info["version"] for year, info in manifest["years"].items()}
    except (OSError, KeyError, ValueError):
        return {}


def filters(
    year: int | None = Query(None, ge=2018, le=2100, description="Reporting year, e.g. 2024"),
    ship_type: str | None = Query(None, description='Ship type, e.g. "Container ship"'),
    flag: str | None = Query(None, description='Flag state, e.g. "Liberia"'),
) -> dict:
    """Turn the query filters into a MongoDB $match document."""
    match: dict = {}
    if year is not None:
        match["year"] = year
    if ship_type:
        merged = merge_one(ship_type)
        labels = [merged] + [old for old, new in SHIP_TYPE_MERGE.items() if new == merged]
        match["ship_type"] = {"$in": labels}
    if flag:
        match["flag"] = flag
    return match


def r1(x) -> float:
    return round(float(x or 0), 1)


def aggregate(db: Database, pipeline: list[dict]) -> list[dict]:
    """Run an aggregation on ship_years, or return the cached answer if it is recent enough."""
    key = json.dumps([db.name, pipeline], sort_keys=True, default=str)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    rows = list(db["ship_years"].aggregate(pipeline))
    if len(_cache) > 2000:  # many different filter combinations: start again rather than grow forever
        _cache.clear()
    _cache[key] = (time.monotonic(), rows)
    return rows


def clear_cache() -> None:
    _cache.clear()


def co2eq_or_none(co2eq, first_year) -> float | None:
    """CO2eq only exists from 2024; a total that includes earlier years would be misleading."""
    return r1(co2eq) if first_year is not None and first_year >= CO2EQ_FIRST_YEAR else None


# ---------- response shapes (shown on /docs) ----------

class Kpis(BaseModel):
    co2_t: float
    co2eq_t: float | None  # null when the selection includes years before 2024
    fuel_t: float
    ships: int
    ship_years: int
    co2_per_ship_t: float  # average per ship per year


class TypeRow(BaseModel):
    ship_type: str
    co2_t: float
    share_pct: float
    ships: int
    ship_years: int
    co2_per_ship_t: float


class YearRow(BaseModel):
    year: int
    co2_t: float
    co2eq_t: float | None  # null before 2024
    fuel_t: float
    ships: int


class Emitter(BaseModel):
    rank: int
    imo: str
    name: str | None
    year: int
    ship_type: str
    flag: str | None
    co2_t: float
    fuel_t: float


class ShipGroup(BaseModel):
    group: str
    ship_types: list[str]


class Meta(BaseModel):
    years: list[int]
    ship_types: list[str]
    flags: list[str]
    ship_groups: list[ShipGroup]


# ---------- endpoints ----------

def warm_cache(db: Database) -> None:
    """Pre-compute the views the website opens with: no filter, each single year and each single ship type."""
    years = sorted(int(y) for y in db["ship_years"].distinct("year"))
    ship_types = sorted({merge_one(t) for t in db["ship_years"].distinct("ship_type") if t})
    views = [(None, None)] + [(y, None) for y in years] + [(None, t) for t in ship_types]
    for match in [filters(year=y, ship_type=t, flag=None) for y, t in views]:
        kpis(match, db), co2_by_type(match, db), co2_by_year(match, db), top_emitters(match, 10, db)


@router.get("/meta", response_model=Meta, summary="Years, ship types, flags and ship groups for the filters")
def meta(db: Database = Depends(get_db)):
    col = db["ship_years"]
    ship_types = sorted({merge_one(t) for t in col.distinct("ship_type") if t})
    flags = sorted(f for f in col.distinct("flag") if f)
    flags = [f for f in flags if f != "Unknown"] + (["Unknown"] if "Unknown" in flags else [])
    groups = {g: [] for g in SHIP_GROUP_ORDER}
    for t in ship_types:
        groups[SHIP_GROUPS.get(t, "Other")].append(t)
    return {
        "years": sorted(int(y) for y in col.distinct("year")),
        "ship_types": ship_types,
        "flags": flags,
        "ship_groups": [{"group": g, "ship_types": ts} for g, ts in groups.items()],
    }


@router.get("/kpis", response_model=Kpis, summary="Headline totals for the selected filters")
def kpis(match: dict = Depends(filters), db: Database = Depends(get_db)):
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$imo", "co2": {"$sum": "$co2_t"}, "co2eq": {"$sum": "$co2eq_t"},
                    "fuel": {"$sum": "$fuel_t"}, "n": {"$sum": 1}, "first_year": {"$min": "$year"}}},
        {"$group": {"_id": None, "co2": {"$sum": "$co2"}, "co2eq": {"$sum": "$co2eq"}, "fuel": {"$sum": "$fuel"},
                    "ships": {"$sum": 1}, "ship_years": {"$sum": "$n"}, "first_year": {"$min": "$first_year"}}},
    ]
    rows = aggregate(db, pipeline)
    row = rows[0] if rows else {}
    ship_years = int(row.get("ship_years", 0))
    return {
        "co2_t": r1(row.get("co2")), "co2eq_t": co2eq_or_none(row.get("co2eq"), row.get("first_year")),
        "fuel_t": r1(row.get("fuel")),
        "ships": int(row.get("ships", 0)), "ship_years": ship_years,
        "co2_per_ship_t": r1(row["co2"] / ship_years) if ship_years else 0.0,
    }


@router.get("/co2-by-type", response_model=list[TypeRow], summary="CO₂ per ship type, largest first")
def co2_by_type(match: dict = Depends(filters), db: Database = Depends(get_db)):
    pipeline = [
        {"$match": match},
        {"$group": {"_id": {"type": MERGED_SHIP_TYPE, "imo": "$imo"}, "co2": {"$sum": "$co2_t"}, "n": {"$sum": 1}}},
        {"$group": {"_id": "$_id.type", "co2": {"$sum": "$co2"}, "ships": {"$sum": 1}, "ship_years": {"$sum": "$n"}}},
        {"$sort": {"co2": -1}},
    ]
    rows = aggregate(db, pipeline)
    total = sum(r["co2"] for r in rows)
    return [{
        "ship_type": r["_id"], "co2_t": r1(r["co2"]), "share_pct": r1(100 * r["co2"] / total) if total else 0.0,
        "ships": r["ships"], "ship_years": r["ship_years"], "co2_per_ship_t": r1(r["co2"] / r["ship_years"]),
    } for r in rows]


@router.get("/co2-by-year", response_model=list[YearRow], summary="CO₂ per reporting year, oldest first")
def co2_by_year(match: dict = Depends(filters), db: Database = Depends(get_db)):
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$year", "co2": {"$sum": "$co2_t"}, "co2eq": {"$sum": "$co2eq_t"},
                    "fuel": {"$sum": "$fuel_t"}, "ships": {"$sum": 1}}},  # one row per ship per year
        {"$sort": {"_id": 1}},
    ]
    return [{"year": r["_id"], "co2_t": r1(r["co2"]), "co2eq_t": co2eq_or_none(r["co2eq"], r["_id"]),
             "fuel_t": r1(r["fuel"]), "ships": r["ships"]} for r in aggregate(db, pipeline)]


@router.get("/top-emitters", response_model=list[Emitter],
            summary="Ship-years with the most CO₂ (without a year filter, all years are ranked together)")
def top_emitters(match: dict = Depends(filters), limit: int = Query(10, ge=1, le=50),
                 db: Database = Depends(get_db)):
    pipeline = [
        {"$match": match},
        {"$sort": {"co2_t": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "imo": 1, "name": 1, "year": 1, "ship_type": MERGED_SHIP_TYPE, "flag": 1,
                      "co2_t": 1, "fuel_t": 1}},
    ]
    return [{**r, "rank": i, "imo": str(r["imo"]), "co2_t": r1(r["co2_t"]), "fuel_t": r1(r["fuel_t"])}
            for i, r in enumerate(aggregate(db, pipeline), start=1)]
