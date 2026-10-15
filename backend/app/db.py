"""MongoDB connection for the API.

Settings come from .env (or real environment variables on Render):
    MONGODB_URI  default mongodb://localhost:27017
    MONGODB_DB   default shipcarbon

One MongoClient is shared by all requests (it keeps its own pool of connections). Routes get the database through
the get_db dependency, so tests can swap in a fake database with app.dependency_overrides.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from ml.features import ROOT

load_dotenv(ROOT / ".env")

# Indexes the API relies on, in addition to those made by pipeline/load_mongo.py.
# The three "covering" indexes start with one filter each and also hold every field the statistics add up, so MongoDB
# can answer from the index instead of reading whole documents (e.g. ship_type=Bulk carrier: ~150 ms -> ~60 ms).
STATS_FIELDS = [("imo", ASCENDING), ("co2_t", ASCENDING), ("co2eq_t", ASCENDING), ("fuel_t", ASCENDING)]
API_INDEXES = [
    [("co2_t", DESCENDING)],                                                                  # top emitters
    [("year", ASCENDING), ("ship_type", ASCENDING), ("flag", ASCENDING)] + STATS_FIELDS,
    [("ship_type", ASCENDING), ("year", ASCENDING), ("flag", ASCENDING)] + STATS_FIELDS,
    [("flag", ASCENDING), ("year", ASCENDING), ("ship_type", ASCENDING)] + STATS_FIELDS,
]


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return MongoClient(uri, serverSelectionTimeoutMS=3000)


def get_db() -> Database:
    return get_client()[os.getenv("MONGODB_DB", "shipcarbon")]


def is_reachable(db: Database) -> bool:
    try:
        db.command("ping")
        return True
    except Exception:  # any connection problem means "not reachable"; /api/health reports it instead of crashing
        return False


def ensure_indexes(db: Database) -> None:
    for keys in API_INDEXES:
        db["ship_years"].create_index(keys)
