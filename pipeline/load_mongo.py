"""Load the cleaned CSV files into MongoDB.

Usage:
    python -m pipeline.load_mongo

The connection comes from .env (MONGODB_URI, MONGODB_DB); default is the local server.
Each document has a fixed _id, and loading uses upserts, so running this twice never creates duplicates.
Documents that are no longer in the CSV (e.g. a report EMSA withdrew) are removed.
"""

import math
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
BATCH_SIZE = 5000

COLLECTIONS = {
    "reports": {
        "file": "emissions.csv",
        "id": lambda r: f"{r['imo']}-{r['year']}-{r['report_kind']}-{r['period_start']}",
        "indexes": [[("year", ASCENDING)], [("ship_type", ASCENDING)], [("flag", ASCENDING)],
                    [("imo", ASCENDING), ("year", ASCENDING)]],
    },
    "ship_years": {
        "file": "ship_years.csv",
        "id": lambda r: f"{r['imo']}-{r['year']}",
        "indexes": [[("year", ASCENDING)], [("ship_type", ASCENDING)], [("flag", ASCENDING)],
                    [("imo", ASCENDING), ("year", ASCENDING)],
                    [("year", ASCENDING), ("co2_t", DESCENDING)]],  # top emitters per year
    },
}


def read_csv(path: Path) -> list[dict]:
    """CSV rows as plain Python dicts. Empty cells become None (null in MongoDB)."""
    # keep_default_na=False: a ship called "NA" stays "NA" instead of becoming empty.
    df = pd.read_csv(path, dtype={"imo": str}, keep_default_na=False, na_values=[""], encoding="utf-8")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in df.to_dict("records")
    ]


def load_collection(db, name: str, spec: dict) -> None:
    # _id goes first: MongoDB stores it first, so an unchanged row then compares as identical (no rewrite).
    rows = [{"_id": spec["id"](row), **row} for row in read_csv(PROCESSED_DIR / spec["file"])]
    ids = [row["_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate _id values in {spec['file']}")

    collection = db[name]
    written = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        result = collection.bulk_write([ReplaceOne({"_id": r["_id"]}, r, upsert=True) for r in batch])
        written += result.upserted_count + result.modified_count
    stale = collection.delete_many({"_id": {"$nin": ids}}).deleted_count
    for keys in spec["indexes"]:
        collection.create_index(keys)

    print(f"  {name}: {len(rows):,} rows in CSV -> {collection.count_documents({}):,} documents "
          f"({written:,} inserted/changed, {stale:,} stale removed, {len(spec['indexes'])} indexes)")


def main() -> None:
    load_dotenv(ROOT / ".env")
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "shipcarbon")
    host = uri.split("@")[-1]  # never print a password that may be in the URI
    print(f"Loading into MongoDB {host} / database '{db_name}'")

    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        db = client[db_name]
        for name, spec in COLLECTIONS.items():
            load_collection(db, name, spec)


if __name__ == "__main__":
    main()
