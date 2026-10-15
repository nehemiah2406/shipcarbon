"""Tests for ml/features.py, ml/train.py helpers and ml/predict.py (uses the saved model in backend/app/model)."""

import json

import numpy as np
import pandas as pd
import pytest

from ml import features, predict
from ml.train import Baseline, scores

FLAGS = ["Liberia", "Malta"]


def ship_years(**overrides) -> pd.DataFrame:
    rows = {
        "ship_type": ["Bulk carrier", "Passenger ship (Cruise Passenger ship)", "Oil tanker"],
        "tech_eff_type": ["EEDI", None, "EEXI"],
        "tech_eff_value": [4.1, None, 3.2],
        "time_at_sea_h": [5000, 9_000_000, 3000],
        "ice_class": [None, "IA Super", "False"],
        "flag": ["Liberia", "Togo", "Unknown"],
        "year": [2020, 2025, 2024],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


# ---------- features ----------

def test_leakage_check_rejects_fuel_derived_columns():
    for bad in ["fuel_t", "co2_t", "co2eq_t", "fuel_per_nm_kg", "distance_nm", "ch4_t"]:
        with pytest.raises(ValueError, match="Leakage"):
            features.check_no_leakage(["ship_type", bad])
    features.check_no_leakage(features.FEATURES)


def test_make_features_columns_and_cleaning():
    X = features.make_features(ship_years(), FLAGS)
    assert list(X.columns) == features.FEATURES
    assert X["ship_type"].tolist() == ["Bulk carrier", "Passenger ship", "Oil tanker"]
    assert X["tech_eff_type"].tolist() == ["EEDI", "Missing", "EEXI"]
    assert X["flag_group"].tolist() == ["Liberia", "Other", "Other"]
    assert np.isnan(X.loc[1, "time_at_sea_h"])  # more hours than a year has -> unknown
    assert X["ice_class_yes"].tolist() == [0, 1, 0]


def test_top_flags_skips_unknown():
    df = pd.DataFrame({"flag": ["Unknown"] * 5 + ["Malta"] * 3 + ["Liberia"] * 2 + ["Togo"]})
    assert features.top_flags(df, n=2) == ["Malta", "Liberia"]


# ---------- baseline and scores ----------

def test_baseline_uses_rate_per_hour_and_falls_back_for_zero_hours():
    X = pd.DataFrame({"ship_type": ["A", "A", "B"], "time_at_sea_h": [100.0, 200.0, 0.0]})
    y = pd.Series([1000.0, 2000.0, 50.0])
    base = Baseline().fit(X, y)
    pred = base.predict(pd.DataFrame({"ship_type": ["A", "B", "C"], "time_at_sea_h": [300.0, 0.0, 10.0]}))
    assert pred[0] == pytest.approx(3000.0)  # 10 t/h x 300 h
    assert pred[1] == pytest.approx(50.0)    # 0 h at sea -> median fuel of type B
    assert pred[2] == pytest.approx(100.0)   # unknown type -> overall rate


def test_scores_perfect_prediction():
    s = scores([10.0, 20.0], [10.0, 20.0])
    assert s["MAE_t"] == 0 and s["MAPE_%"] == 0 and s["R2"] == 1.0


# ---------- saved model ----------

def test_saved_metrics_model_beats_baseline():
    metrics = json.loads((predict.MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    for split in ["GroupKFold by IMO", "Train 2018-23, test 2024-25"]:
        model, base = metrics["validation"][f"{split} | model"], metrics["validation"][f"{split} | baseline"]
        assert model["MAE_t"] < 0.75 * base["MAE_t"]
    features.check_no_leakage(metrics["features"])


def test_predict_one_full_input():
    out = predict.predict_one({"ship_type": "Container ship", "time_at_sea_h": 6000, "tech_eff_type": "EEXI",
                               "tech_eff_value": 9.5, "flag": "Liberia", "ice_class": "IA", "year": 2024})
    assert set(out) == {"fuel_t", "co2_t", "co2_factor", "smoke"}
    assert 100 < out["fuel_t"] < 100_000
    assert out["co2_t"] == pytest.approx(out["fuel_t"] * out["co2_factor"], rel=1e-3)
    assert 0 <= out["smoke"] <= 1


def test_predict_one_minimal_input_and_merged_passenger_label():
    a = predict.predict_one({"ship_type": "Passenger ship", "time_at_sea_h": 5000})
    b = predict.predict_one({"ship_type": "Passenger ship (Cruise Passenger ship)", "time_at_sea_h": 5000})
    assert a == b


def test_more_hours_at_sea_means_more_fuel():
    low = predict.predict_one({"ship_type": "Bulk carrier", "time_at_sea_h": 1000})
    high = predict.predict_one({"ship_type": "Bulk carrier", "time_at_sea_h": 7000})
    assert high["fuel_t"] > 2 * low["fuel_t"]


@pytest.mark.parametrize("ship, message", [
    ({"time_at_sea_h": 100}, "Missing"),
    ({"ship_type": "Submarine", "time_at_sea_h": 100}, "Unknown ship_type"),
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 9000}, "between"),
    ({"ship_type": "Bulk carrier", "time_at_sea_h": 100, "co2_t": 5}, "Unknown fields"),
])
def test_predict_one_rejects_bad_input(ship, message):
    with pytest.raises(ValueError, match=message):
        predict.predict_one(ship)


def test_smoke_scale_is_clipped_log_scale():
    assert predict.smoke_scale(10, 100, 10_000) == 0
    assert predict.smoke_scale(1_000, 100, 10_000) == 0.5
    assert predict.smoke_scale(1e6, 100, 10_000) == 1
