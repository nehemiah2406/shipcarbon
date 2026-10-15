"""Train and evaluate the annual fuel-consumption model.

Usage:
    python -m ml.train

Writes backend/app/model/{fuel_model.joblib, metrics.json, co2_factor_by_type.json}
and docs/img/08_feature_importance.png.
"""

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.features import CATEGORICAL, FEATURES, ROOT, TARGET, check_no_leakage, load_training_data, make_features, top_flags

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after choosing the non-window backend)

MODEL_DIR = ROOT / "backend" / "app" / "model"
IMG_DIR = ROOT / "docs" / "img"
SEED = 42
TIME_SPLIT_LAST_TRAIN_YEAR = 2023

# A small, sensible search (6 settings). Early stopping picks the number of trees.
PARAM_GRID = [
    dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20),
    dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20),
    dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50),
    dict(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=50),
    dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
]


def make_model(**params) -> TransformedTargetRegressor:
    """Encode text columns as numbers, fit gradient-boosted trees on log(fuel), predict back in tonnes."""
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    trees = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),  # the encoded text columns come first
        max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=SEED, **params,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("encode", encode), ("trees", trees)]), func=np.log, inverse_func=np.exp,
    )


class Baseline:
    """Median fuel per hour at sea, per ship type, times the ship's hours at sea."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Baseline":
        d = X.assign(fuel=y.to_numpy())
        sea = d[d["time_at_sea_h"] > 0]
        rate = sea["fuel"] / sea["time_at_sea_h"]
        self.rate_by_type, self.rate_all = rate.groupby(sea["ship_type"]).median(), rate.median()
        self.fuel_by_type, self.fuel_all = d.groupby("ship_type")["fuel"].median(), d["fuel"].median()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rate = X["ship_type"].map(self.rate_by_type).fillna(self.rate_all)
        fallback = X["ship_type"].map(self.fuel_by_type).fillna(self.fuel_all)  # ships with 0 h at sea
        return np.where(X["time_at_sea_h"] > 0, rate * X["time_at_sea_h"], fallback)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    pct_error = np.abs(pred - y) / y
    return {
        "MAE_t": round(float(mean_absolute_error(y, pred)), 1),
        "MAPE_%": round(float(pct_error.mean() * 100), 1),
        "median_%_error": round(float(np.median(pct_error) * 100), 1),
        "R2": round(float(r2_score(y, pred)), 3),
        "R2_log": round(float(r2_score(np.log(y), np.log(np.clip(pred, 1e-3, None)))), 3),
    }


def out_of_fold(make, X, y, groups, n_splits) -> np.ndarray:
    """Predict every row with a model that never saw that ship (GroupKFold by IMO)."""
    pred = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    return pred


def search(X, y, groups) -> tuple[dict, list[dict]]:
    results = []
    for params in PARAM_GRID:
        start = time.time()
        pred = out_of_fold(lambda: make_model(**params), X, y, groups, n_splits=3)
        results.append({**params, "MAE_t": round(float(mean_absolute_error(y, pred)), 1),
                        "seconds": round(time.time() - start, 1)})
        print(f"  {params} -> MAE {results[-1]['MAE_t']:,.0f} t ({results[-1]['seconds']}s)")
    best = min(results, key=lambda r: r["MAE_t"])
    return {k: v for k, v in best.items() if k not in ("MAE_t", "seconds")}, results


def importance_chart(importance: pd.Series, path) -> None:
    BG, LINE, TEXT, MUTED, AMBER = "#0a1220", "#20324a", "#e9eef4", "#9aabbf", "#ffb547"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 10.5,
        "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": LINE, "grid.color": LINE, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
    labels = {"time_at_sea_h": "Time at sea", "ship_type": "Ship type", "tech_eff_value": "Efficiency index value",
              "tech_eff_type": "Efficiency index type", "flag_group": "Flag", "year": "Year",
              "ice_class_yes": "Ice class"}
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh([labels.get(i, i) for i in imp.index], imp, color=AMBER, height=0.62, edgecolor=BG, linewidth=1.5)
    for i, v in enumerate(imp):
        ax.annotate(f"+{v:,.0f} t", (max(v, 0), i), xytext=(4, 0), textcoords="offset points", va="center",
                    color=TEXT, fontsize=9.5)
    ax.set_xlim(0, imp.max() * 1.18)
    ax.set_xlabel("Increase in average error (tonnes of fuel) when the feature is shuffled")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.annotate("What drives the fuel prediction", (0, 1), xycoords="axes fraction", xytext=(0, 38),
                textcoords="offset points", fontsize=15, fontweight="bold", color=TEXT, va="bottom")
    ax.annotate("Permutation importance on unseen years 2024–2025 (model trained on 2018–2023)", (0, 1),
                xycoords="axes fraction", xytext=(0, 16), textcoords="offset points", color=MUTED, va="bottom")
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(box.x0 / fig.get_figwidth(), (box.y0 - 0.3) / fig.get_figheight(),
             "Source: EMSA THETIS-MRV (EU MRV), ShipCarbon model", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    check_no_leakage(FEATURES)
    df = load_training_data()
    flags = top_flags(df)
    X, y, groups = make_features(df, flags), df[TARGET], df["imo"]
    print(f"Training rows: {len(df):,} ship-years, {groups.nunique():,} ships, years {df['year'].min()}-{df['year'].max()}")
    print(f"Features: {FEATURES}")

    past = df["year"] <= TIME_SPLIT_LAST_TRAIN_YEAR
    print("\n1) Small search on 2018-2023 (GroupKFold 3 by IMO):")
    best, search_results = search(X[past], y[past], groups[past])
    print(f"   best settings: {best}")

    print("\n2) GroupKFold 5 by IMO on all years ...")
    cv_model = out_of_fold(lambda: make_model(**best), X, y, groups, n_splits=5)
    cv_base = out_of_fold(Baseline, X, y, groups, n_splits=5)

    print("3) Time check: train 2018-2023, test 2024-2025 ...")
    time_model = make_model(**best).fit(X[past], y[past])
    time_base = Baseline().fit(X[past], y[past])
    t_model, t_base = time_model.predict(X[~past]), time_base.predict(X[~past])

    comparison = pd.DataFrame({
        ("GroupKFold by IMO", "baseline"): scores(y, cv_base),
        ("GroupKFold by IMO", "model"): scores(y, cv_model),
        ("Train 2018-23, test 2024-25", "baseline"): scores(y[~past], t_base),
        ("Train 2018-23, test 2024-25", "model"): scores(y[~past], t_model),
    }).T
    print("\n=== Baseline vs model ===")
    print(comparison.to_string())

    per_type = pd.DataFrame({"ship_type": X["ship_type"], "fuel": y, "model": cv_model, "baseline": cv_base})
    per_type = per_type.groupby("ship_type").apply(lambda d: pd.Series({
        "ship_years": len(d),
        "median_fuel_t": d["fuel"].median(),
        "MAE_model_t": mean_absolute_error(d["fuel"], d["model"]),
        "MAE_baseline_t": mean_absolute_error(d["fuel"], d["baseline"]),
        "median_%_error_model": np.median(np.abs(d["model"] - d["fuel"]) / d["fuel"]) * 100,
        "median_%_error_baseline": np.median(np.abs(d["baseline"] - d["fuel"]) / d["fuel"]) * 100,
    }), include_groups=False).sort_values("ship_years", ascending=False).round(1)
    print("\n=== Per ship type (GroupKFold out-of-fold) ===")
    print(per_type.to_string())

    print("\n4) Permutation importance on 2024-2025 ...")
    perm = permutation_importance(time_model, X[~past], y[~past], scoring="neg_mean_absolute_error",
                                  n_repeats=5, random_state=SEED, n_jobs=1)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    print(importance.round(1).to_string())
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    importance_chart(importance, IMG_DIR / "08_feature_importance.png")

    print("\n5) Final model on all data ...")
    final = make_model(**best).fit(X, y)
    co2_ratio = (df["co2_t"] / df[TARGET]).groupby(df["ship_type"]).median().round(4)
    factors = {"by_type": co2_ratio.to_dict(), "default": round(float((df["co2_t"] / df[TARGET]).median()), 4),
               "note": "median CO2 / fuel per ship type, full-report ship-years with fuel > 0"}
    co2 = df["co2_t"]
    bundle = {
        "model": final,
        "features": FEATURES,
        "top_flags": flags,
        "ship_types": sorted(X["ship_type"].unique()),
        "tech_eff_types": sorted(X["tech_eff_type"].unique()),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "smoke": {"co2_p05": float(co2.quantile(0.05)), "co2_p99": float(co2.quantile(0.99))},
        "sklearn_version": sklearn.__version__,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fuel_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    (MODEL_DIR / "co2_factor_by_type.json").write_text(json.dumps(factors, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads((ROOT / "data" / "raw" / "manifest.json").read_text(encoding="utf-8"))
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "fuel_t (trained on log(fuel_t), reported in tonnes)",
        "rows": len(df), "ships": int(groups.nunique()), "features": FEATURES, "top_flags": flags,
        "best_params": best, "search": search_results,
        "validation": {f"{a} | {b}": row for (a, b), row in comparison.to_dict(orient="index").items()},
        "per_ship_type": per_type.reset_index().to_dict(orient="records"),
        "permutation_importance_mae_t": importance.round(1).to_dict(),
        "data_versions": {y_: v["version"] for y_, v in manifest["years"].items()},
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {model_path.relative_to(ROOT)} ({model_path.stat().st_size / 1e6:.2f} MB), "
          f"metrics.json, co2_factor_by_type.json, 08_feature_importance.png")


if __name__ == "__main__":
    main()
