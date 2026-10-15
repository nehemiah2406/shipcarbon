"""Tests for pipeline/columns.py and pipeline/clean.py, using tiny hand-made tables."""

import math

import numpy as np
import pandas as pd
import pytest

from pipeline import clean
from pipeline.columns import column_map, era_for_year

LOOKUP = {"MONROVIA": "Liberia", "VALLETTA": "Malta", "ST. JOHN'S": "Antigua and Barbuda"}


# ---------- technical efficiency ----------

@pytest.mark.parametrize("raw, kind, value", [
    ("EEDI (4.12 gCO₂/t·nm)", "EEDI", 4.12),
    ("EEXI (12 gCO₂/t·nm)", "EEXI", 12.0),
    ("EIV (57.84 gCO₂/t·nm)", "EIV", 57.84),
    ("Not Applicable", "Not Applicable", None),
    ("Not Applicable (3.1 gCO₂/t·nm)", "Not Applicable", None),
    ("EEDI (0 gCO₂/t·nm)", "EEDI", None),   # zero is not a real index value
    ("5.5 gCO₂/t·nm", None, 5.5),            # number without a type
    ("EIV", "EIV", None),                    # type without a number
    (None, None, None),
    ("", None, None),
])
def test_parse_tech_eff(raw, kind, value):
    got_kind, got_value = clean.parse_tech_eff(raw)
    assert got_kind == kind
    if value is None:
        assert math.isnan(got_value)
    else:
        assert got_value == pytest.approx(value)


# ---------- ports and flags ----------

@pytest.mark.parametrize("raw, expected", [
    (" Valletta ", "VALLETTA"),
    ("St. John´s", "ST. JOHN'S"),
    ("St. John`s", "ST. JOHN'S"),
    ("Hong   Kong", "HONG KONG"),
    ("Piraeus 9939", "PIRAEUS"),
    ("", None),
    (None, None),
    (np.nan, None),
])
def test_normalise_port(raw, expected):
    assert clean.normalise_port(raw) == expected


def ships(rows):
    return pd.DataFrame(rows, columns=["imo", "year", "port_of_registry"])


def test_flag_lookup_from_port():
    df = clean.assign_flags(ships([("1", 2020, "MONROVIA"), ("2", 2020, "ST. JOHN'S")]), LOOKUP)
    assert df["flag"].tolist() == ["Liberia", "Antigua and Barbuda"]
    assert df["flag_source"].tolist() == ["port", "port"]


def test_unmapped_and_blank_ports_become_unknown():
    df = clean.assign_flags(ships([("1", 2020, "ATLANTIS"), ("2", 2020, None)]), LOOKUP)
    assert df["flag"].tolist() == ["Unknown", "Unknown"]
    assert df["flag_source"].tolist() == ["unmapped_port", "blank_port"]


def test_blank_port_borrows_nearest_year_of_same_ship():
    df = clean.assign_flags(ships([
        ("1", 2018, "MONROVIA"),
        ("1", 2021, None),        # 2022 is nearer than 2018
        ("1", 2022, "VALLETTA"),
        ("2", 2021, None),        # ship 2 has no port in any year
    ]), LOOKUP)
    assert df["flag"].tolist() == ["Liberia", "Malta", "Malta", "Unknown"]
    assert df["flag_source"].tolist() == ["port", "port_other_year", "port", "blank_port"]
    assert pd.isna(df.loc[1, "port_of_registry"])  # the original port stays blank


def test_port_lookup_file_has_no_duplicate_ports():
    table = pd.read_csv(clean.PORT_TO_FLAG, encoding="utf-8")
    normalised = table["port"].map(clean.normalise_port)
    assert not normalised.duplicated().any()
    assert set(table["confidence"]) <= {"sure", "unsure", "checked"}


# ---------- column mapping per era ----------

@pytest.mark.parametrize("year, era, time_column", [
    (2018, "2018-2019", "Annual Total time spent at sea [hours]"),
    (2019, "2018-2019", "Annual Total time spent at sea [hours]"),
    (2020, "2020-2023", "Annual Time spent at sea [hours]"),
    (2023, "2020-2023", "Annual Time spent at sea [hours]"),
    (2024, "2024+", "Time spent at sea [hours]"),
    (2025, "2024+", "Time spent at sea [hours]"),
])
def test_era_and_time_at_sea_column(year, era, time_column):
    assert era_for_year(year) == era
    assert column_map(year)["time_at_sea_h"] == time_column


def raw_sheet(year, **overrides):
    """A one-row raw sheet with every source column of that year's era."""
    data = {source: ["x"] for source in column_map(year).values() if source}
    data.update(overrides)
    return pd.DataFrame(data)


def test_column_mapping_renames_to_clean_names():
    raw = raw_sheet(2021, **{"Annual Time spent at sea [hours]": [1234.5], "IMO Number": ["9000001"]})
    df = clean.apply_column_map(raw, 2021)
    assert df.loc[0, "time_at_sea_h"] == 1234.5
    assert df.loc[0, "imo"] == "9000001"
    assert df["ch4_t"].isna().all()  # CH4 is not published before 2024


def test_company_imo_column_is_not_used_as_ship_imo():
    raw = raw_sheet(2024, **{"IMO Number": ["9000001"], "IMO Number.1": ["5112991"]})
    assert clean.apply_column_map(raw, 2024).loc[0, "imo"] == "9000001"


def test_missing_column_becomes_nan_with_warning():
    raw = raw_sheet(2024).drop(columns=["Total CH₄ emissions [m tonnes]"])
    with pytest.warns(UserWarning, match="Total CH₄ emissions"):
        df = clean.apply_column_map(raw, 2024)
    assert df["ch4_t"].isna().all()


# ---------- numbers ----------

def test_division_by_zero_text_becomes_nan():
    df = pd.DataFrame({"co2_per_nm_kg": ["Division by zero!", 312.5, "18.2", None]})
    out = clean.to_numbers(df, ["co2_per_nm_kg"])
    assert out["co2_per_nm_kg"].isna().tolist() == [True, False, False, True]
    assert out.loc[2, "co2_per_nm_kg"] == 18.2


def test_zero_fuel_flag():
    df = pd.DataFrame({
        "fuel_t": [0.0, 125.3, np.nan],
        "ship_type": ["Bulk carrier", "LNG carrier", "Gas carrier"],
        "fuel_per_nm_kg": [0.0, 50.0, np.nan],
    })
    out = clean.add_derived(df)
    assert out["zero_fuel"].tolist() == [True, False, False]
    assert out["ship_group"].tolist() == ["Bulk", "LNG", "Other"]
    assert out.loc[1, "distance_nm"] == pytest.approx(125.3 * 1000 / 50.0)
    assert out[["distance_nm"]].iloc[[0, 2]].isna().all().all()


# ---------- partial reports ----------

def test_partial_period_dates():
    df = pd.DataFrame({"year": [2024, 2024], "reporting_period": ["2024 (9/1 - 8/7)", 2024.0]})
    out = clean.period_bounds(df)
    assert out["period_start"].tolist() == ["2024-01-09", "2024-01-01"]
    assert out["period_end"].tolist() == ["2024-07-08", "2024-12-31"]


def test_partials_count_only_when_ship_has_no_full_report():
    df = pd.DataFrame({
        "imo": ["1", "1", "2", "2"],
        "year": [2024] * 4,
        "report_kind": ["full", "partial", "partial", "partial"],
    })
    assert clean.mark_counts_in_total(df)["counts_in_total"].tolist() == [True, False, True, True]


def test_ship_years_do_not_double_count_partials():
    reports = pd.DataFrame({
        "imo": ["1", "1", "2", "2"],
        "year": [2024] * 4,
        "report_kind": ["full", "partial", "partial", "partial"],
        "counts_in_total": [True, False, True, True],
        "period_start": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-06-01"],
        "co2_t": [300.0, 300.0, 100.0, 50.0],
        "fuel_t": [100.0, 100.0, 30.0, 20.0],
        "distance_nm": [1000.0, 1000.0, 300.0, 200.0],
        "name": ["A", "A", "OLD NAME", "NEW NAME"],
        "fuel_per_nm_kg": [100.0, 100.0, 100.0, 100.0],
        "co2_per_nm_kg": [300.0, 300.0, 333.0, 250.0],
    })
    for col in clean.ADDITIVE + clean.SHIP_INFO:
        if col not in reports:
            reports[col] = np.nan
    sy = clean.build_ship_years(reports).set_index("imo")
    assert sy.loc["1", "co2_t"] == 300.0                  # full report only, partial not added
    assert sy.loc["2", "co2_t"] == 150.0                  # partials summed
    assert sy.loc["2", "name"] == "NEW NAME"              # latest period wins
    assert sy.loc["2", "co2_per_nm_kg"] == pytest.approx(150.0 * 1000 / 500.0)
    assert sy["source"].tolist() == ["full", "partial_only"]
