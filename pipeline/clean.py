"""Clean the raw EMSA files into two tables.

    data/processed/emissions.csv   one row per emission report (full and partial), same row count as the raw files
    data/processed/ship_years.csv  one row per ship (IMO) per year, for top emitters and ML

Usage:
    python -m pipeline.clean

Partial reports: from 2024 a ship that changed company has partial reports for parts of the year.
When the ship also has a full report that year, the full report already covers the partial periods
(checked in Phase 2: identical numbers in 190+ cases, impossible sea hours otherwise). So only rows with
counts_in_total=True are added up: every full report, plus partial reports of ships without a full report.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.columns import NUMERIC, column_map

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PORT_TO_FLAG = ROOT / "data" / "reference" / "port_to_flag.csv"

# Groups that drive the 3D ship buttons on the website. Everything else is "Other".
SHIP_GROUPS = {
    "Container ship": "Container",
    "Container/ro-ro cargo ship": "Container",
    "Oil tanker": "Tanker",
    "Chemical tanker": "Tanker",
    "Bulk carrier": "Bulk",
    "LNG carrier": "LNG",
    "Ro-pax ship": "Ro-pax",
}

TECH_EFF_TYPES = {"EEDI": "EEDI", "EEXI": "EEXI", "EIV": "EIV", "NOT APPLICABLE": "Not Applicable"}
APOSTROPHES = str.maketrans({"´": "'", "`": "'", "’": "'", "‘": "'"})
PARTIAL_PERIOD = r"\((\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})\)"  # "2024 (1/1 - 28/6)" -> day/month

ADDITIVE = [
    "fuel_t", "co2_t", "co2_between_ms_t", "co2_from_ms_t", "co2_to_ms_t", "co2_at_berth_t",
    "ch4_t", "n2o_t", "co2eq_t", "time_at_sea_h", "distance_nm",
]
SHIP_INFO = [
    "name", "ship_type", "ship_group", "tech_eff_type", "tech_eff_value",
    "port_of_registry", "flag", "flag_source", "ice_class",
]
REPORT_COLUMNS = [
    "imo", "name", "year", "report_kind", "period_start", "period_end", "counts_in_total",
    "ship_type", "ship_group", "tech_eff_type", "tech_eff_value",
    "port_of_registry", "flag", "flag_source", "ice_class", "verifier_name",
    "fuel_t", "zero_fuel", "co2_t", "co2_between_ms_t", "co2_from_ms_t", "co2_to_ms_t", "co2_at_berth_t",
    "ch4_t", "n2o_t", "co2eq_t", "time_at_sea_h", "distance_nm", "fuel_per_nm_kg", "co2_per_nm_kg",
]


# ---------- small building blocks (tested in tests/test_clean.py) ----------

def apply_column_map(raw: pd.DataFrame, year: int, label: str = "") -> pd.DataFrame:
    """Rename raw columns to clean names. A column missing from the sheet becomes NaN with a warning."""
    out = pd.DataFrame(index=raw.index)
    for clean, source in column_map(year).items():
        if source is not None and source in raw.columns:
            out[clean] = raw[source]
        else:
            if source is not None:  # None = not published in this era, which is expected
                warnings.warn(f"{label or year}: column {source!r} is missing -> '{clean}' left empty")
            out[clean] = np.nan
    return out


def to_numbers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Turn text such as 'Division by zero!' into NaN."""
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_tech_eff(value) -> tuple[str | None, float]:
    """'EEDI (4.12 gCO₂/t·nm)' -> ('EEDI', 4.12). Not Applicable, zero or missing values -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return None, np.nan
    text = value.strip()
    kind_match = re.match(r"(EEDI|EEXI|EIV|Not Applicable)", text, re.IGNORECASE)
    kind = TECH_EFF_TYPES[kind_match.group(1).upper()] if kind_match else None
    number_match = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    number = float(number_match.group()) if number_match else np.nan
    if kind == "Not Applicable" or not number > 0:
        number = np.nan
    return kind, number


def normalise_port(value) -> str | None:
    """' St. John´s ' -> "ST. JOHN'S"; 'Piraeus 9939' -> 'PIRAEUS' (Greek registry number). Blank -> None."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.translate(APOSTROPHES).split()).upper()
    text = re.sub(r"\s+[0-9]+$", "", text)
    return text or None


def load_port_lookup(path: Path = PORT_TO_FLAG) -> dict[str, str]:
    table = pd.read_csv(path, encoding="utf-8")
    return {normalise_port(p): f for p, f in zip(table["port"], table["flag"])}


def assign_flags(df: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    """Add flag + flag_source. A blank port borrows the same ship's port from the nearest other year."""
    df = df.copy()
    port = df["port_of_registry"].copy()
    borrowed = pd.Series(False, index=df.index)

    blank = port.isna()
    if blank.any():
        known = df.loc[~blank, ["imo", "year", "port_of_registry"]].rename(
            columns={"year": "other_year", "port_of_registry": "other_port"})
        candidates = (
            df.loc[blank, ["imo", "year"]].rename_axis("row").reset_index()
            .merge(known, on="imo")
            .assign(distance=lambda d: (d["year"] - d["other_year"]).abs())
            .sort_values(["row", "distance", "other_year"], ascending=[True, True, False])
            .drop_duplicates("row")
        )
        port.loc[candidates["row"]] = candidates["other_port"].to_numpy()
        borrowed.loc[candidates["row"]] = True

    flag = port.map(lookup)
    df["flag"] = flag.fillna("Unknown")
    df["flag_source"] = np.select(
        [flag.notna() & ~borrowed, flag.notna() & borrowed, port.notna()],
        ["port", "port_other_year", "unmapped_port"],
        default="blank_port",
    )
    return df


def period_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """period_start / period_end as 'YYYY-MM-DD'. Full reports cover the whole year."""
    df = df.copy()
    parts = df["reporting_period"].astype(str).str.extract(PARTIAL_PERIOD).astype(float)
    start = pd.to_datetime(
        dict(year=df["year"], month=parts[1].fillna(1), day=parts[0].fillna(1)), errors="coerce")
    end = pd.to_datetime(
        dict(year=df["year"], month=parts[3].fillna(12), day=parts[2].fillna(31)), errors="coerce")
    df["period_start"] = start.dt.strftime("%Y-%m-%d")
    df["period_end"] = end.dt.strftime("%Y-%m-%d")
    return df


def mark_counts_in_total(df: pd.DataFrame) -> pd.DataFrame:
    """Full reports count; partial reports count only when the ship has no full report that year."""
    df = df.copy()
    full_keys = pd.MultiIndex.from_frame(df.loc[df["report_kind"] == "full", ["imo", "year"]])
    keys = pd.MultiIndex.from_frame(df[["imo", "year"]])
    df["counts_in_total"] = (df["report_kind"] == "full") | ~keys.isin(full_keys)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zero_fuel"] = df["fuel_t"].eq(0)
    df["ship_group"] = df["ship_type"].map(SHIP_GROUPS).fillna("Other")
    per_nm = df["fuel_per_nm_kg"].where(df["fuel_per_nm_kg"] > 0)
    df["distance_nm"] = df["fuel_t"] * 1000 / per_nm  # derived from fuel: never an ML feature
    return df


# ---------- whole-table steps ----------

def read_raw_year(path: Path, year: int) -> pd.DataFrame:
    """Read every sheet of one year's file and map it to clean columns."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")  # harmless Excel styling
        sheets = pd.read_excel(path, sheet_name=None, header=2, dtype=object)
    frames = []
    for sheet, raw in sheets.items():
        raw = raw.dropna(subset=["IMO Number"])
        df = apply_column_map(raw, year, label=f"{year} '{sheet}'")
        df["year"] = year
        df["report_kind"] = "partial" if "partial" in sheet.lower() else "full"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_reports(raw: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    df = to_numbers(raw, NUMERIC)
    df["imo"] = df["imo"].astype(str).str.strip().str.removesuffix(".0")
    df["name"] = df["name"].astype(str).str.strip()
    df[["tech_eff_type", "tech_eff_value"]] = pd.DataFrame(
        [parse_tech_eff(v) for v in df["tech_eff_raw"]], index=df.index)
    df["port_of_registry"] = df["port_raw"].map(normalise_port)
    df = assign_flags(df, lookup)
    df = period_bounds(df)
    df = mark_counts_in_total(df)
    df = add_derived(df)
    df = df[REPORT_COLUMNS].sort_values(["year", "imo", "report_kind", "period_start"], ignore_index=True)
    if df.duplicated(["imo", "year", "report_kind", "period_start"]).any():
        raise ValueError("Two reports share imo + year + kind + period_start; the MongoDB _id would clash.")
    return df


def build_ship_years(reports: pd.DataFrame) -> pd.DataFrame:
    """One row per imo + year from the rows that count in totals (no double counting)."""
    rows = reports[reports["counts_in_total"]].sort_values("period_start")
    groups = rows.groupby(["imo", "year"])
    sums = groups[ADDITIVE].sum(min_count=1)
    latest = rows.drop_duplicates(["imo", "year"], keep="last").set_index(["imo", "year"])
    ship_years = sums.join(latest[SHIP_INFO + ["fuel_per_nm_kg", "co2_per_nm_kg"]])
    ship_years["n_reports"] = groups.size()
    ship_years["source"] = np.where(groups["report_kind"].first() == "full", "full", "partial_only")

    # Several partial reports: recompute per-distance ratios from the summed totals.
    several = ship_years["n_reports"] > 1
    distance = ship_years["distance_nm"].where(ship_years["distance_nm"] > 0)
    ship_years.loc[several, "fuel_per_nm_kg"] = (ship_years["fuel_t"] * 1000 / distance)[several]
    ship_years.loc[several, "co2_per_nm_kg"] = (ship_years["co2_t"] * 1000 / distance)[several]
    ship_years["zero_fuel"] = ship_years["fuel_t"].eq(0)
    return ship_years.reset_index()


def summary(reports: pd.DataFrame, ship_years: pd.DataFrame) -> pd.DataFrame:
    by_year = reports.groupby("year")
    table = pd.DataFrame({
        "reports": by_year.size(),
        "partial": by_year["report_kind"].apply(lambda s: (s == "partial").sum()),
        "co2_all_rows_Mt": by_year["co2_t"].sum() / 1e6,
        "co2_dedup_Mt": reports[reports["counts_in_total"]].groupby("year")["co2_t"].sum() / 1e6,
        "ships": ship_years.groupby("year").size(),
        "flag_known_pct": ship_years.groupby("year")["flag"].apply(lambda s: (s != "Unknown").mean() * 100),
    })
    return table.round(2)


def main() -> None:
    lookup = load_port_lookup()
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit("No raw files found. Run: python -m pipeline.download")

    frames = []
    for path in files:
        year = int(path.stem)
        print(f"Reading {path.name} ...")
        frames.append(read_raw_year(path, year))
    reports = clean_reports(pd.concat(frames, ignore_index=True), lookup)
    ship_years = build_ship_years(reports)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reports.to_csv(PROCESSED_DIR / "emissions.csv", index=False, encoding="utf-8")
    ship_years.to_csv(PROCESSED_DIR / "ship_years.csv", index=False, encoding="utf-8")

    print(f"\nWrote {len(reports):,} reports and {len(ship_years):,} ship-years to {PROCESSED_DIR}")
    print(summary(reports, ship_years).to_string())
    print("\nflag_source (ship-years):", ship_years["flag_source"].value_counts().to_dict())
    unmapped = reports.loc[reports["flag_source"] == "unmapped_port", "port_of_registry"].value_counts()
    print(f"Unmapped ports: {len(unmapped)} distinct; most common: {unmapped.head(8).to_dict()}")


if __name__ == "__main__":
    main()
