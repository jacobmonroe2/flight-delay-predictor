"""Flask app: serves the site locally and offers the prediction as a Python API.

The public site (GitHub Pages) is static and runs the model in the browser with
docs/predictor.js. This server is for local use and for checking that the
JavaScript predictor matches LightGBM: POST /api/predict returns the reference answer.

Run:  python src/app.py   then open http://localhost:5000
"""
import bisect
import datetime as dt
import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from features import FEATURES, GROUPS, GROUP_ORDER, build_features

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "docs"
META = json.loads((ROOT / "models" / "meta.json").read_text())
MODEL = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False   # keep factors and results in their own order

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def band(p):
    for b in META["bands"]:
        if p < b["hi"]:
            return b["band"]
    return META["bands"][-1]["band"]


def pct(x):
    return f"{x * 100:.0f}%"


def rows_for(airline, origin, dest, date, minutes_list):
    """One feature row per departure time (minutes after midnight)."""
    r = META["routes"][f"{origin}-{dest}"]
    recs = []
    for m in minutes_list:
        arr = (m + r["shift"]) % 1440
        recs.append({
            "airline": airline, "origin": origin, "dest": dest, "date": date,
            "month": date.month, "day_of_week": date.isoweekday(),
            "crs_dep": (m // 60) * 100 + m % 60,
            "crs_arr": (arr // 60) * 100 + arr % 60,
            "distance": r["distance"],
        })
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    return df, build_features(df, META)


def explain(x_row, contrib_row, p, q):
    """Roll per-feature SHAP values up into the factors shown to the user and
    express each as percentage points of risk."""
    sums = {}
    for j, f in enumerate(FEATURES):
        sums[GROUPS[f]] = sums.get(GROUPS[f], 0.0) + float(contrib_row[j])
    logit = math.log(p / (1 - p))
    base = META["base_rate"]
    al = META["airlines"][q["airline"]]
    o, d = META["airports"].get(q["origin"], {}), META["airports"].get(q["dest"], {})
    route = META["routes"][f"{q['origin']}-{q['dest']}"]
    hour = q["minutes"] // 60
    hr = META["rate_by_hour"].get(str(hour), {}).get("rate")
    dow_rate = META["rate_by_dow"][str(q["date"].isoweekday())]["rate"]
    mon = META["rate_by_month"].get(str(q["date"].month), {}).get("rate")
    gap = int(x_row["holiday_gap"])

    season = f"{MONTHS[q['date'].month - 1]} flights run late {pct(mon)} of the time." if mon else ""
    if gap <= 3:
        season += f" This date is within {gap} day{'s' if gap != 1 else ''} of a holiday travel peak."
    detail = {
        "Departure time": (f"Flights leaving in the {hour:02d}:00 hour run late {pct(hr)} of the time; "
                           "delays build up through the day." if hr is not None else ""),
        "Airline": f"{al['name']} arrives late on {pct(al['rate'])} of its flights.",
        "Origin airport": (f"{pct(o['dep_rate'])} of flights leaving {q['origin']} arrive late."
                           if o.get("dep_rate") is not None else ""),
        "Destination airport": (f"{pct(d['arr_rate'])} of flights into {q['dest']} arrive late."
                                if d.get("arr_rate") is not None else ""),
        "Season": season,
        "Day of week": f"{DOW[q['date'].isoweekday() - 1]} flights run late {pct(dow_rate)} of the time.",
        "Route": (f"{route['distance']:,} miles, about {max(1, round(META['route_daily'].get(q['origin'] + '-' + q['dest'], 0)))} "
                  f"flights a day; {pct(route['rate'])} of them arrive late."),
        "Airport congestion": (f"{q['origin']} schedules about {x_row['origin_hour_deps']:.0f} departures that hour; "
                               f"{q['dest']} about {x_row['dest_hour_arrs']:.0f} arrivals when you land."),
    }
    factors = []
    for name in GROUP_ORDER:
        c = sums.get(name, 0.0)
        without = 1 / (1 + math.exp(-(logit - c)))
        factors.append({"factor": name, "points": round((p - without) * 100, 1),
                        "detail": detail[name]})
    factors.sort(key=lambda f: -abs(f["points"]))
    return factors, base


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/<path:name>")
def static_file(name):
    return send_from_directory(WEB, name)


@app.get("/api/meta")
def meta():
    keys = ["base_rate", "n_flights", "n_train", "n_test", "date_min", "date_max",
            "n_airports", "n_routes", "results", "calibration", "bands", "importance",
            "airlines", "airports", "top_decile_rate", "bottom_decile_rate"]
    out = {k: META[k] for k in keys}
    out["routes"] = {k: {"airlines": v["airlines"], "distance": v["distance"]}
                     for k, v in META["routes"].items()}
    return jsonify(out)


@app.post("/api/predict")
def predict():
    q = request.get_json(force=True)
    try:
        airline = str(q["airline"]).upper()
        origin, dest = str(q["origin"]).upper(), str(q["dest"]).upper()
        date = dt.date.fromisoformat(q["date"])
        hh, mm = (int(x) for x in str(q["time"]).split(":")[:2])
        minutes = hh * 60 + mm
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Send airline, origin, dest, date (YYYY-MM-DD) and time (HH:MM)."), 400
    if airline not in META["airlines"]:
        return jsonify(error=f"Unknown airline {airline}."), 400
    route = META["routes"].get(f"{origin}-{dest}")
    if route is None:
        return jsonify(error=f"No {origin} to {dest} flights in the training data."), 400

    hours = list(range(5, 24))
    times = [minutes] + [h * 60 + mm for h in hours]
    df, X = rows_for(airline, origin, dest, date, times)
    probs = MODEL.predict(X)
    contrib = MODEL.predict(X.iloc[[0]], pred_contrib=True)[0]
    p = float(probs[0])
    query = {"airline": airline, "origin": origin, "dest": dest, "date": date, "minutes": minutes}
    factors, base = explain(X.iloc[0], contrib, p, query)
    arr = (minutes + route["shift"]) % 1440

    return jsonify({
        "risk": round(p, 4),
        "band": band(p),
        "base_rate": base,
        "percentile": max(0, min(99, bisect.bisect_left(META["percentiles"], p) - 1)),
        "factors": factors,
        "hourly": [{"hour": h, "risk": round(float(r), 4)} for h, r in zip(hours, probs[1:])],
        "arrival": f"{arr // 60:02d}:{arr % 60:02d}",
        "distance": route["distance"],
        "route_rate": route["rate"],
        "airline_flies_route": airline in route["airlines"],
        "airline_name": META["airlines"][airline]["name"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
