"""Map EMSA's raw column names (which changed over the years) to short, clean names.

EMSA used three different header layouts ("eras"):
    2018-2019  62 columns, "Annual Total time spent at sea [hours]"
    2020-2023  62 columns, "Annual Time spent at sea [hours]"
    2024+      113 columns, no "Annual average" prefix, adds CH4 / N2O / CO2eq, Full + Partial sheets

From 2024 the sheet also has a second "IMO Number"/"Name" pair for the *Company*.
pandas renames those to "IMO Number.1"/"Name.1", so the plain names below always mean the ship.
"""

# Same source name in every era.
COMMON = {
    "imo": "IMO Number",
    "name": "Name",
    "ship_type": "Ship type",
    "reporting_period": "Reporting Period",
    "tech_eff_raw": "Technical efficiency",
    "port_raw": "Port of Registry",
    "ice_class": "Ice Class",
    "verifier_name": "Verifier Name",
    "fuel_t": "Total fuel consumption [m tonnes]",
    "co2_t": "Total CO₂ emissions [m tonnes]",
    "co2_between_ms_t": "CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes]",
    "co2_from_ms_t": "CO₂ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]",
    "co2_to_ms_t": "CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes]",
    "co2_at_berth_t": "CO₂ emissions which occurred within ports under a MS jurisdiction at berth [m tonnes]",
}

# Columns whose name differs per era. None = the column does not exist in that era.
ERA_SPECIFIC = {
    "2018-2019": {
        "time_at_sea_h": "Annual Total time spent at sea [hours]",
        "fuel_per_nm_kg": "Annual average Fuel consumption per distance [kg / n mile]",
        "co2_per_nm_kg": "Annual average CO₂ emissions per distance [kg CO₂ / n mile]",
        "ch4_t": None,
        "n2o_t": None,
        "co2eq_t": None,
    },
    "2020-2023": {
        "time_at_sea_h": "Annual Time spent at sea [hours]",
        "fuel_per_nm_kg": "Annual average Fuel consumption per distance [kg / n mile]",
        "co2_per_nm_kg": "Annual average CO₂ emissions per distance [kg CO₂ / n mile]",
        "ch4_t": None,
        "n2o_t": None,
        "co2eq_t": None,
    },
    "2024+": {
        "time_at_sea_h": "Time spent at sea [hours]",
        "fuel_per_nm_kg": "Fuel consumption per distance [kg / n mile]",
        "co2_per_nm_kg": "CO₂ emissions per distance [kg CO₂ / n mile]",
        "ch4_t": "Total CH₄ emissions [m tonnes]",
        "n2o_t": "Total N₂O emissions [m tonnes]",
        "co2eq_t": "Total CO₂eq emissions [m tonnes]",
    },
}

# Clean columns that must be numbers ("Division by zero!" and other text becomes NaN).
NUMERIC = [
    "fuel_t", "co2_t", "co2_between_ms_t", "co2_from_ms_t", "co2_to_ms_t", "co2_at_berth_t",
    "time_at_sea_h", "fuel_per_nm_kg", "co2_per_nm_kg", "ch4_t", "n2o_t", "co2eq_t",
]


def era_for_year(year: int) -> str:
    if year <= 2019:
        return "2018-2019"
    if year <= 2023:
        return "2020-2023"
    return "2024+"


def column_map(year: int) -> dict[str, str | None]:
    """Clean name -> raw source name (or None when the era has no such column)."""
    return {**COMMON, **ERA_SPECIFIC[era_for_year(year)]}
