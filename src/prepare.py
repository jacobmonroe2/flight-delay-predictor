"""Turn the raw BTS zip files into one clean table of flights.

Reads every data/raw/ontime_*.zip, keeps only the columns the project needs,
drops cancelled and diverted flights (they have no arrival delay), and writes
data/flights.parquet. Also fetches airport coordinates for the map.
"""
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "flights.parquet"
AIRPORTS = ROOT / "data" / "airports.csv"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

COLS = {
    "FlightDate": "date",
    "Month": "month",
    "DayOfWeek": "day_of_week",          # 1 = Monday ... 7 = Sunday
    "Reporting_Airline": "airline",
    "Origin": "origin",
    "OriginCityName": "origin_city",
    "Dest": "dest",
    "DestCityName": "dest_city",
    "CRSDepTime": "crs_dep",             # scheduled, local time, hhmm
    "CRSArrTime": "crs_arr",
    "CRSElapsedTime": "crs_elapsed",
    "Distance": "distance",
    "ArrDelay": "arr_delay",
    "ArrDel15": "delayed",               # 1 if arrival is 15+ minutes late
    "Cancelled": "cancelled",
    "Diverted": "diverted",
    "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
}


def read_month(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=list(COLS), low_memory=False)
    return df.rename(columns=COLS)


def main():
    files = sorted(RAW.glob("ontime_*.zip"))
    if not files:
        raise SystemExit("No raw files found. Run src/download_data.py first.")

    parts, stats = [], {"raw": 0, "cancelled": 0, "diverted": 0}
    for f in files:
        df = read_month(f)
        stats["raw"] += len(df)
        stats["cancelled"] += int(df["cancelled"].sum())
        stats["diverted"] += int(df["diverted"].sum())
        df = df[(df["cancelled"] == 0) & (df["diverted"] == 0)]
        df = df.dropna(subset=["delayed", "crs_dep", "crs_arr", "distance"])
        df = df.drop(columns=["cancelled", "diverted"])
        print(f"{f.name}: {len(df):,} usable flights, "
              f"{df['delayed'].mean():.1%} delayed")
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["delayed"] = df["delayed"].astype("int8")
    for c in ("month", "day_of_week"):
        df[c] = df[c].astype("int8")
    for c in ("crs_dep", "crs_arr"):
        df[c] = df[c].astype("int16")
    for c in ("airline", "origin", "dest", "origin_city", "dest_city"):
        df[c] = df[c].astype("category")
    df.to_parquet(OUT, index=False)

    print(f"\nRaw rows:        {stats['raw']:,}")
    print(f"Cancelled:       {stats['cancelled']:,}")
    print(f"Diverted:        {stats['diverted']:,}")
    print(f"Usable flights:  {len(df):,}")
    print(f"Date range:      {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Delayed overall: {df['delayed'].mean():.2%}")
    print(f"Wrote {OUT}")

    if not AIRPORTS.exists():
        r = requests.get(AIRPORTS_URL, timeout=120)
        r.raise_for_status()
        AIRPORTS.write_bytes(r.content)
        print(f"Wrote {AIRPORTS}")


if __name__ == "__main__":
    main()
