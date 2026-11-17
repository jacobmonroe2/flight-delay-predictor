"""Download monthly On-Time Performance files from the Bureau of Transportation Statistics.

Usage:
    python src/download_data.py                 # most recent 12 available months
    python src/download_data.py 2025-09 2026-08 # explicit inclusive range
"""
import sys
import datetime as dt
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
HEADERS = {"User-Agent": "flight-delay-predictor/1.0 (student project)"}


def month_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(y, m):
    """Download one month. Returns True if the file is on disk afterwards."""
    dest = RAW / f"ontime_{y}_{m:02d}.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  {y}-{m:02d} already downloaded")
        return True
    r = requests.get(URL.format(y=y, m=m), headers=HEADERS, stream=True, timeout=120)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "html" in ctype:
        print(f"  {y}-{m:02d} not available (HTTP {r.status_code}, {ctype})")
        return False
    tmp = dest.with_suffix(".part")
    size = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            size += len(chunk)
    if size < 1_000_000:
        tmp.unlink()
        print(f"  {y}-{m:02d} response too small ({size} bytes), skipped")
        return False
    tmp.rename(dest)
    print(f"  {y}-{m:02d} downloaded ({size / 1e6:.1f} MB)")
    return True


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 3:
        start = tuple(int(x) for x in sys.argv[1].split("-"))
        end = tuple(int(x) for x in sys.argv[2].split("-"))
        for y, m in month_range(start, end):
            fetch(y, m)
        return

    # BTS publishes with a lag of roughly two to three months: walk backwards
    # from today until 12 months have been collected.
    today = dt.date.today()
    y, m = today.year, today.month
    got, tried = 0, 0
    while got < 12 and tried < 20:
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        tried += 1
        if fetch(y, m):
            got += 1
        elif got:
            break  # a gap after the newest month means something is wrong
    print(f"{got} months on disk in {RAW}")


if __name__ == "__main__":
    main()
