"""Feature engineering shared by training (train.py) and the app (app.py).

Every feature is something a traveller knows when booking: nothing here uses
weather, the aircraft's earlier legs, or anything else only known on the day.
"""
import datetime as dt

import numpy as np
import pandas as pd

CATEGORICAL = ["airline", "origin", "dest"]
NUMERIC = [
    "month", "day_of_week", "dep_hour", "arr_hour", "distance",
    "holiday_gap", "route_daily", "origin_hour_deps", "dest_hour_arrs",
]
FEATURES = CATEGORICAL + NUMERIC

# How individual features roll up into the factors shown to the user.
GROUPS = {
    "airline": "Airline",
    "origin": "Origin airport",
    "dest": "Destination airport",
    "dep_hour": "Departure time",
    "arr_hour": "Departure time",
    "day_of_week": "Day of week",
    "month": "Season",
    "holiday_gap": "Season",
    "distance": "Route",
    "route_daily": "Route",
    "origin_hour_deps": "Airport congestion",
    "dest_hour_arrs": "Airport congestion",
}
GROUP_ORDER = ["Departure time", "Airline", "Origin airport", "Destination airport",
               "Season", "Day of week", "Route", "Airport congestion"]


def _nth_weekday(year, month, weekday, n):
    """n-th given weekday of a month (weekday: Monday = 0); n = -1 for the last."""
    if n > 0:
        d = dt.date(year, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
        return d
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def holidays(year):
    """The US travel-peak holidays for one year."""
    return [
        dt.date(year, 1, 1),
        _nth_weekday(year, 5, 0, -1),    # Memorial Day
        dt.date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),     # Labor Day
        _nth_weekday(year, 11, 3, 4),    # Thanksgiving
        dt.date(year, 12, 25),
    ]


def holiday_gap(dates):
    """Days to the nearest travel-peak holiday, capped at 14."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    years = range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2)
    hol = np.array(sorted(h for y in years for h in holidays(y)), dtype="datetime64[D]")
    uniq = dates.drop_duplicates()
    d = uniq.values.astype("datetime64[D]")
    idx = np.clip(np.searchsorted(hol, d), 1, len(hol) - 1)
    gap = np.minimum(np.abs((d - hol[idx - 1]).astype(int)),
                     np.abs((hol[idx] - d).astype(int)))
    lookup = dict(zip(uniq.values, np.minimum(gap, 14)))
    return dates.map(lookup).astype("int8").values


def hhmm_to_hour(hhmm):
    hhmm = np.asarray(hhmm)
    return ((hhmm // 100) % 24 + (hhmm % 100) / 60).astype("float32")


def build_features(df, meta):
    """df needs: airline, origin, dest, date, month, day_of_week, crs_dep,
    crs_arr, distance. meta carries the category lists and schedule lookups."""
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c].astype(str), categories=meta["categories"][c])
    X["month"] = df["month"].astype("int8")
    X["day_of_week"] = df["day_of_week"].astype("int8")
    X["dep_hour"] = hhmm_to_hour(df["crs_dep"])
    X["arr_hour"] = hhmm_to_hour(df["crs_arr"])
    X["distance"] = df["distance"].astype("float32")
    X["holiday_gap"] = holiday_gap(df["date"])

    route = df["origin"].astype(str) + "-" + df["dest"].astype(str)
    X["route_daily"] = route.map(meta["route_daily"]).fillna(0).astype("float32").values

    dep_h = (np.asarray(df["crs_dep"]) // 100) % 24
    arr_h = (np.asarray(df["crs_arr"]) // 100) % 24
    od = meta["origin_hour_deps"]
    da = meta["dest_hour_arrs"]
    zeros = [0.0] * 24
    X["origin_hour_deps"] = np.fromiter(
        (od.get(o, zeros)[h] for o, h in zip(df["origin"].astype(str), dep_h)),
        dtype="float32", count=len(df))
    X["dest_hour_arrs"] = np.fromiter(
        (da.get(d, zeros)[h] for d, h in zip(df["dest"].astype(str), arr_h)),
        dtype="float32", count=len(df))
    return X[FEATURES]
