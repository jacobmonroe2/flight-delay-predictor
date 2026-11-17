"""Train and evaluate the delay model.

Target: arrival 15+ minutes late (the DOT definition of a delayed flight).
Split: whole days are held out, so the model is always scored on days it has
never seen. A random row split would let it memorise individual storm days.

Outputs: models/model.txt (LightGBM booster) and models/meta.json (category
lists, schedule lookups, metrics and the reference tables the app displays).
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL, FEATURES, GROUPS, GROUP_ORDER, build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flights.parquet"
MODELS = ROOT / "models"
SEED = 42

AIRLINE_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
    "NK": "Spirit Airlines", "F9": "Frontier Airlines", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "OO": "SkyWest Airlines", "9E": "Endeavor Air",
    "YX": "Republic Airways", "MQ": "Envoy Air", "OH": "PSA Airlines",
    "YV": "Mesa Airlines", "QX": "Horizon Air", "ZW": "Air Wisconsin",
    "G7": "GoJet Airlines", "C5": "CommuteAir", "PT": "Piedmont Airlines",
    "EV": "ExpressJet", "SY": "Sun Country Airlines", "MX": "Breeze Airways",
    "XP": "Avelo Airlines",
}


def schedule_lookups(df):
    """Traffic levels taken from the schedule alone (never from delay outcomes)."""
    n_days = df["date"].nunique()
    route = df["origin"].astype(str) + "-" + df["dest"].astype(str)
    route_daily = (route.value_counts() / n_days).round(3).to_dict()

    def hourly(col, time_col):
        hour = (df[time_col] // 100) % 24
        tab = (df.groupby([df[col].astype(str), hour], observed=True).size()
               .unstack(fill_value=0).reindex(columns=range(24), fill_value=0) / n_days)
        return {k: [round(float(x), 3) for x in v] for k, v in zip(tab.index, tab.values)}

    return {
        "route_daily": route_daily,
        "origin_hour_deps": hourly("origin", "crs_dep"),
        "dest_hour_arrs": hourly("dest", "crs_arr"),
    }


def score(y, p):
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} flights")

    meta = {"categories": {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}}
    meta.update(schedule_lookups(df))

    X = build_features(df, meta)
    y = df["delayed"].values
    days = df["date"].values
    print(f"Features built in {time.time() - t0:.0f}s")

    # Hold out whole days: 70% train, 15% validation, 15% test.
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(X, y, days))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED)
                          .split(rest, groups=days[rest]))
    va, te = rest[va_rel], rest[te_rel]
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,} flights "
          f"({len(np.unique(days[tr]))}/{len(np.unique(days[va]))}/{len(np.unique(days[te]))} days)")

    results = {}
    base_rate = float(y[tr].mean())

    # Baseline 1: always answer with the overall delay rate.
    results["Overall delay rate"] = score(y[te], np.full(len(te), base_rate))

    # Baseline 2: the airline's historical delay rate on that route (smoothed).
    k = 50
    key = (df["airline"].astype(str) + "|" + df["origin"].astype(str) + "-"
           + df["dest"].astype(str)).values
    g = pd.DataFrame({"key": key[tr], "y": y[tr]}).groupby("key")["y"].agg(["sum", "count"])
    rate = ((g["sum"] + k * base_rate) / (g["count"] + k)).to_dict()
    p_route = pd.Series(key[te]).map(rate).fillna(base_rate).values
    results["Airline + route history"] = score(y[te], p_route)

    # Baseline 3: logistic regression on one-hot features.
    lr_cols = ["airline", "origin", "dest", "month", "day_of_week", "dep_hour_int"]
    Xl = X[["airline", "origin", "dest", "month", "day_of_week"]].astype(str)
    Xl["dep_hour_int"] = X["dep_hour"].astype(int).astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(tr, size=min(len(tr), 1_500_000), replace=False)
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(enc.fit_transform(Xl.iloc[sub][lr_cols]), y[sub])
    results["Logistic regression"] = score(
        y[te], lr.predict_proba(enc.transform(Xl.iloc[te][lr_cols]))[:, 1])
    print(f"Baselines done in {time.time() - t0:.0f}s")

    # The model: gradient-boosted decision trees.
    params = dict(
        objective="binary", learning_rate=0.03, num_leaves=127,
        min_data_in_leaf=1500, feature_fraction=0.6, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=5.0, cat_smooth=50, cat_l2=10,
        max_cat_to_onehot=8, max_bin=255, seed=SEED, verbose=-1,
    )
    dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(X.iloc[va], y[va], reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=3000, valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    p_test = booster.predict(X.iloc[te], num_iteration=booster.best_iteration)
    results["Gradient-boosted trees"] = score(y[te], p_test)
    print(f"Model trained in {time.time() - t0:.0f}s, {booster.best_iteration} trees")
    for name, r in results.items():
        print(f"  {name:26s} AUC {r['auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  "
              f"Brier {r['brier']:.4f}  logloss {r['log_loss']:.4f}")

    # Calibration: does a predicted 30% come true about 30% of the time?
    yt = y[te]
    bins = pd.qcut(p_test, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"b": bins, "p": p_test, "y": yt}).groupby("b")
           .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")))
    calibration = [{"predicted": round(float(r.predicted), 4),
                    "actual": round(float(r.actual), 4), "n": int(r.n)}
                   for r in cal.itertuples()]

    # What a user gets from each risk band.
    order = np.argsort(-p_test)
    top10 = order[: len(order) // 10]
    bot10 = order[-(len(order) // 10):]
    bands = []
    for label, lo, hi in [("Low", 0.0, 0.15), ("Moderate", 0.15, 0.25),
                          ("Elevated", 0.25, 0.35), ("High", 0.35, 1.01)]:
        m = (p_test >= lo) & (p_test < hi)
        bands.append({"band": label, "lo": lo, "hi": min(hi, 1.0),
                      "share": round(float(m.mean()), 4),
                      "actual": round(float(yt[m].mean()), 4) if m.any() else None})

    # Which factors matter overall: mean |SHAP| per factor group on a test sample.
    samp = rng.choice(te, size=min(len(te), 60_000), replace=False)
    contrib = booster.predict(X.iloc[samp], pred_contrib=True,
                              num_iteration=booster.best_iteration)[:, :-1]
    grouped = {}
    for j, f in enumerate(FEATURES):
        grouped.setdefault(GROUPS[f], np.zeros(len(samp)))
        grouped[GROUPS[f]] += contrib[:, j]
    imp = {gname: float(np.abs(v).mean()) for gname, v in grouped.items()}
    total = sum(imp.values())
    importance = [{"factor": gname, "share": round(imp[gname] / total, 4)}
                  for gname in sorted(GROUP_ORDER, key=lambda n: -imp[n])]

    # Reference tables for the app and the slides (descriptive, full dataset).
    def rate_by(col):
        g = df.groupby(col, observed=True)["delayed"].agg(["mean", "size"])
        return {str(k): {"rate": round(float(r["mean"]), 4), "n": int(r["size"])}
                for k, r in g.iterrows()}

    dep_hour_int = (df["crs_dep"] // 100) % 24
    delayed_rows = df[df["delayed"] == 1]
    causes = {c: float(delayed_rows[c + "_delay"].sum())
              for c in ["carrier", "weather", "nas", "security", "late_aircraft"]}
    cause_total = sum(causes.values())

    # Per-route facts the app needs to fill in a query.
    df["route"] = (df["origin"].astype(str) + "-" + df["dest"].astype(str))
    dep_min = (df["crs_dep"] // 100) * 60 + df["crs_dep"] % 100
    arr_min = (df["crs_arr"] // 100) * 60 + df["crs_arr"] % 100
    df["clock_shift"] = (arr_min - dep_min) % 1440   # includes time-zone change
    df["late60"] = df["arr_delay"] > 60              # more than an hour late
    rg = df.groupby("route").agg(distance=("distance", "median"),
                                 clock=("clock_shift", "median"),
                                 rate=("delayed", "mean"), late60=("late60", "mean"),
                                 n=("delayed", "size"))
    ra = df.groupby("route")["airline"].agg(lambda s: sorted(s.astype(str).unique()))
    routes = {r: {"distance": int(v.distance), "shift": int(v.clock),
                  "rate": round(float(v.rate), 4), "late60": round(float(v.late60), 4),
                  "n": int(v.n), "airlines": ra[r]}
              for r, v in rg.iterrows()}

    cities = {}
    for code_col, city_col in (("origin", "origin_city"), ("dest", "dest_city")):
        pairs = df[[code_col, city_col]].drop_duplicates()
        cities.update(dict(zip(pairs[code_col].astype(str), pairs[city_col].astype(str))))

    ap = pd.read_csv(ROOT / "data" / "airports.csv",
                     usecols=["iata_code", "latitude_deg", "longitude_deg", "type"])
    ap = ap[ap["iata_code"].isin(cities) & (ap["type"] != "closed")]
    ap = ap.drop_duplicates("iata_code").set_index("iata_code")
    origin_rates, dest_rates = rate_by("origin"), rate_by("dest")
    airports = {}
    fallback = {"PBI": (26.6832, -80.0956)}   # missing an IATA code in the source file
    for code, city in cities.items():
        if code in ap.index:
            lat, lon = ap.at[code, "latitude_deg"], ap.at[code, "longitude_deg"]
        elif code in fallback:
            lat, lon = fallback[code]
        else:
            continue
        airports[code] = {
            "city": city, "lat": round(float(lat), 4), "lon": round(float(lon), 4),
            "dep_rate": origin_rates.get(code, {}).get("rate"),
            "arr_rate": dest_rates.get(code, {}).get("rate"),
            "flights": origin_rates.get(code, {}).get("n", 0),
        }
    missing = sorted(set(cities) - set(airports))
    if missing:
        print(f"No coordinates for {len(missing)} airports: {missing[:15]}")

    airline_rates = rate_by("airline")
    meta.update({
        "base_rate": round(float(y.mean()), 4),
        "n_flights": int(len(df)),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_airports": int(df["origin"].nunique()),
        "n_routes": int(len(routes)),
        "best_iteration": int(booster.best_iteration),
        "results": results,
        "calibration": calibration,
        "bands": bands,
        "top_decile_rate": round(float(yt[top10].mean()), 4),
        "bottom_decile_rate": round(float(yt[bot10].mean()), 4),
        "importance": importance,
        "percentiles": [round(float(q), 4) for q in np.quantile(p_test, np.linspace(0, 1, 101))],
        "delay_causes": {k: round(v / cause_total, 4) for k, v in causes.items()},
        "avg_delay_minutes_when_late": round(float(delayed_rows["arr_delay"].mean()), 1),
        "airlines": {code: {"name": AIRLINE_NAMES.get(code, code), **r}
                     for code, r in airline_rates.items()},
        "airports": airports,
        "routes": routes,
        "rate_by_hour": {str(k): {"rate": round(float(r["mean"]), 4), "n": int(r["size"])}
                         for k, r in df.groupby(dep_hour_int)["delayed"]
                         .agg(["mean", "size"]).iterrows()},
        "rate_by_dow": rate_by("day_of_week"),
        "rate_by_month": rate_by("month"),
    })

    MODELS.mkdir(exist_ok=True)
    booster.save_model(str(MODELS / "model.txt"), num_iteration=booster.best_iteration)
    (MODELS / "meta.json").write_text(json.dumps(meta))
    print(f"Saved model and metadata to {MODELS}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
