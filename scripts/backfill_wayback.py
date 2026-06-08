#!/usr/bin/env python3
"""
Backfill historical snow-report data from the Wayback Machine.

For each resort in URL_HISTORY, queries the Wayback Machine CDX API to find one
archived snapshot per day during SH ski season (May–Oct), fetches the raw content,
parses it with the same parsers used in collectors/resort_snapshot.py, and appends
unique (resort, date) rows to data/raw/snow_reports/{resort_id}.csv.

CDX API is queried per year to avoid timeouts (one year window ≈ 200 snapshots).
Already-collected dates are skipped to allow safe re-runs.

Usage:
    python scripts/backfill_wayback.py
    python scripts/backfill_wayback.py --resort thredbo
    python scripts/backfill_wayback.py --resort thredbo --year 2022
    python scripts/backfill_wayback.py --from-year 2020 --to-year 2024
    python scripts/backfill_wayback.py --dry-run   # parse + print, no write
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.resort_snapshot import (
    CSV_COLS,
    RAW_DIR,
    HEADERS,
    _PARSERS,
    classify_condition,
)

REGIONS = Path("regions.yaml")
CDX_API  = "https://web.archive.org/cdx/search/cdx"
WB_BASE  = "https://web.archive.org/web"
CDX_TIMEOUT   = 90   # seconds — CDX API can be slow
FETCH_TIMEOUT = 45   # seconds for individual archive page fetches
SLEEP_AFTER_FETCH = 1.5  # polite delay between Wayback requests


# ── URL history per resort ─────────────────────────────────────────────────────
# Each entry: (year_from, year_to_inclusive, archive_url, parser_format)
# year_from / year_to refer to the calendar year of the SH ski season.
# When a resort changed URL, add multiple entries; the correct one is selected
# per year automatically.

URL_HISTORY: dict[str, list[tuple[int, int, str, str]]] = {
    "thredbo": [
        (2015, 2025,
         "https://www.thredbo.com.au/mountain-information/snow-report/",
         "livepass_xml"),
    ],
    "perisher": [
        (2015, 2022,
         "https://www.perisher.com.au/mountain-information/snow-report/",
         "perisher_html"),
        (2023, 2025,
         "https://www.perisher.com.au/reports-cams/reports/snow-report",
         "perisher_html"),
    ],
    "falls_creek": [
        # JSON endpoint — filename reused across years; Wayback has it
        (2021, 2021,
         "https://www.fallscreek.com.au/wp-content/uploads/FCSnowReport_2021.json",
         "falls_creek_json"),
        (2022, 2022,
         "https://www.fallscreek.com.au/wp-content/uploads/FCSnowReport_2022.json",
         "falls_creek_json"),
        (2023, 2023,
         "https://www.fallscreek.com.au/wp-content/uploads/FCSnowReport_2023.json",
         "falls_creek_json"),
        (2024, 2024,
         "https://www.fallscreek.com.au/wp-content/uploads/FCSnowReport_2024.json",
         "falls_creek_json"),
        (2025, 2025,
         "https://www.fallscreek.com.au/wp-content/uploads/FCSnowReport_2025.json",
         "falls_creek_json"),
        # Old HTML fallback for pre-JSON era
        (2015, 2020,
         "https://www.fallscreek.com.au/snow-report/",
         "generic_html"),
    ],
    "whakapapa": [
        (2018, 2025,
         "https://www.whakapapa.com/report",
         "whakapapa_lit"),
        # Older URL variant
        (2015, 2017,
         "https://www.whakapapa.com/snow-report",
         "generic_html"),
    ],
    "coronet_peak": [
        (2015, 2025,
         "https://www.nzski.com/coronet-peak/the-mountain/snow-report",
         "nzski_html"),
    ],
    "the_remarkables": [
        (2015, 2025,
         "https://www.nzski.com/remarkables/the-mountain/snow-report",
         "nzski_html"),
    ],
    "mt_hutt": [
        (2015, 2025,
         "https://www.nzski.com/mt-hutt/the-mountain/snow-report",
         "nzski_html"),
    ],
    "cardrona": [
        (2015, 2025,
         "https://www.cardrona.com/snow-report/",
         "generic_html"),
    ],
}


# ── Extra parsers for historical HTML formats ──────────────────────────────────

import re


def _parse_nzski_html(text: str) -> dict:
    """
    Parse NZSki resort snow report HTML (Coronet Peak, Remarkables, Mt Hutt).
    NZSki used a simple table/list layout; the JS-rendered version came later.
    """
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-z#\d]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    def _num(pattern: str) -> float:
        m = re.search(pattern, clean, re.I)
        return float(m.group(1)) if m else 0.0

    snow_24h = _num(r"(?:new\s+snow|fresh\s+snow|snowfall\s+24\s*h[rs]?|24\s*h[rs]?\s+snow)[:\s]*([\d.]+)\s*cm")
    snow_7d  = _num(r"(?:7\s*day|week[ly]?\s+snow)[:\s]*([\d.]+)\s*cm")
    base     = _num(r"(?:base\s+depth|snow\s+depth|natural\s+depth|base)[:\s]*([\d.]+)\s*cm")

    temp_m  = re.search(r"(?:temp(?:erature)?|°C)[:\s]*([\-\d.]+)\s*°?C", clean, re.I)
    wind_m  = re.search(r"([\d.]+)\s*km/?h\s*([NSEW]{1,3})", clean, re.I)

    conditions_m = re.search(
        r"(?:conditions?|snow\s+summary|overview|daily\s+report)[:\s]+(.{20,400}?)"
        r"(?:lifts?\s+open|roads?|webcam|forecast|\Z)",
        clean, re.I,
    )
    conditions = conditions_m.group(1).strip() if conditions_m else ""

    return {
        "snow_24h_cm":    snow_24h,
        "snow_7d_cm":     snow_7d,
        "base_cm":        base,
        "temp_now_c":     float(temp_m.group(1)) if temp_m else 0.0,
        "wind_speed_kmh": float(wind_m.group(1)) if wind_m else 0.0,
        "wind_direction": wind_m.group(2) if wind_m else "",
        "conditions_text": conditions[:400],
        "condition_label": classify_condition(conditions),
    }


def _parse_generic_html(text: str) -> dict:
    """
    Fallback parser for simple resort HTML pages (Falls Creek old site, Cardrona, etc.).
    Tries common label patterns; returns zeros for anything not found.
    """
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-z#\d]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    def _num(pattern: str) -> float:
        m = re.search(pattern, clean, re.I)
        return float(m.group(1)) if m else 0.0

    snow_24h = _num(
        r"(?:new\s+snow|fresh\s+snow|24\s*h[rs]?\s*snow(?:fall)?|snow(?:fall)?\s*24|overnight\s+snow)"
        r"[:\s]*([\d.]+)\s*cm"
    )
    snow_7d  = _num(r"(?:7\s*day|week[ly]?\s+snow|snow\s*(?:last)?\s*7)[:\s]*([\d.]+)\s*cm")
    base     = _num(
        r"(?:base\s+depth|snow\s+base|natural\s+depth|base|snow\s+depth)"
        r"[:\s]*([\d.]+)\s*cm"
    )
    temp_m   = re.search(r"([\-\d.]+)\s*°C", clean, re.I)
    wind_m   = re.search(r"([\d.]+)\s*km/?h\s*([NSEW]{1,3})", clean, re.I)

    conditions_m = re.search(
        r"(?:conditions?|snow\s+report|snow\s+summary)[:\s]+(.{20,400}?)"
        r"(?:lifts?|trail|road|forecast|webcam|\Z)",
        clean, re.I,
    )
    conditions = conditions_m.group(1).strip() if conditions_m else ""

    return {
        "snow_24h_cm":    snow_24h,
        "snow_7d_cm":     snow_7d,
        "base_cm":        base,
        "temp_now_c":     float(temp_m.group(1)) if temp_m else 0.0,
        "wind_speed_kmh": float(wind_m.group(1)) if wind_m else 0.0,
        "wind_direction": wind_m.group(2) if wind_m else "",
        "conditions_text": conditions[:400],
        "condition_label": classify_condition(conditions),
    }


_EXTENDED_PARSERS = {
    **_PARSERS,
    "nzski_html":   _parse_nzski_html,
    "generic_html": _parse_generic_html,
}


# ── Wayback Machine helpers ────────────────────────────────────────────────────

def cdx_get_snapshots(url: str, year: int) -> list[tuple[str, str]]:
    """
    Query CDX API for one snapshot per day (collapsed) during SH ski season
    (May–Oct) for the given calendar year.
    Returns list of (timestamp_14digit, status_code) tuples.
    """
    params = {
        "url":      url,
        "output":   "json",
        "collapse": "timestamp:8",
        "from":     f"{year}0501",
        "to":       f"{year}1031",
        "filter":   "statuscode:200",
        "limit":    "250",
        "fl":       "timestamp,statuscode",
    }
    try:
        resp = requests.get(CDX_API, params=params, timeout=CDX_TIMEOUT)
        resp.raise_for_status()
        rows = resp.json()
        if not rows or len(rows) < 2:
            return []
        return [(r[0], r[1]) for r in rows[1:]]
    except Exception as exc:
        print(f"      CDX error: {exc}")
        return []


def fetch_wayback_content(timestamp: str, url: str) -> str | None:
    """Fetch raw content from Wayback Machine at the given timestamp."""
    wb_url = f"{WB_BASE}/{timestamp}if_/{url}"
    try:
        resp = requests.get(
            wb_url, headers=HEADERS, timeout=FETCH_TIMEOUT, allow_redirects=True
        )
        if resp.status_code == 200 and len(resp.text) > 300:
            return resp.text
        return None
    except Exception as exc:
        print(f"      fetch error ts={timestamp}: {exc}")
        return None


# ── Main backfill logic ────────────────────────────────────────────────────────

def _save_row(data: dict, out_path: Path, existing_dates: set[str]) -> bool:
    """Append one row to CSV; return True if saved."""
    snap_date = data["date"]
    if snap_date in existing_dates:
        return False

    row = pd.DataFrame([data]).reindex(columns=CSV_COLS)
    if out_path.exists():
        row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        row.to_csv(out_path, index=False)
    existing_dates.add(snap_date)
    return True


def backfill_resort(
    resort_id: str,
    years: list[int],
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Backfill historical snapshots for one resort. Returns count of new rows."""
    url_variants = URL_HISTORY.get(resort_id)
    if not url_variants:
        print(f"  {resort_id}: no URL history configured — skipping")
        return 0

    out_path = RAW_DIR / f"{resort_id}.csv"
    if out_path.exists():
        existing = set(pd.read_csv(out_path, usecols=["date"])["date"].astype(str))
    else:
        existing = set()

    total_saved = 0

    for year in sorted(years):
        variants = [(y0, y1, u, f) for y0, y1, u, f in url_variants if y0 <= year <= y1]
        if not variants:
            continue

        for _y0, _y1, url, fmt in variants:
            parser = _EXTENDED_PARSERS.get(fmt)
            if parser is None:
                print(f"    [{resort_id}] {year}  unknown format {fmt!r} — skipping")
                continue

            print(f"    [{resort_id}] {year}  querying CDX: {url[:70]} ...")
            snapshots = cdx_get_snapshots(url, year)
            print(f"    [{resort_id}] {year}  {len(snapshots)} snapshots from CDX")

            saved = 0
            skipped = 0
            failed = 0
            for ts, _status in snapshots:
                snap_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

                if snap_date in existing:
                    skipped += 1
                    continue

                content = fetch_wayback_content(ts, url)
                time.sleep(SLEEP_AFTER_FETCH)

                if content is None:
                    failed += 1
                    continue

                try:
                    data = parser(content)
                except Exception as exc:
                    if verbose:
                        print(f"      parse error {snap_date}: {exc}")
                    failed += 1
                    continue

                data["date"]      = snap_date
                data["resort_id"] = resort_id

                snow = data.get("snow_24h_cm", 0)
                base = data.get("base_cm", 0)
                cond = data.get("condition_label", "?")

                if dry_run:
                    print(f"      {snap_date}  24h={snow}cm  base={base}cm  cond={cond}")
                    existing.add(snap_date)  # prevent duplicate dry-run prints
                else:
                    if _save_row(data, out_path, existing):
                        saved += 1
                        total_saved += 1
                        if verbose:
                            print(f"      {snap_date}  24h={snow}cm  base={base}cm  cond={cond}")

            print(
                f"    [{resort_id}] {year}  saved={saved}  skipped={skipped}  failed={failed}"
            )

    return total_saved


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill historical snow-report data from the Wayback Machine"
    )
    ap.add_argument("--resort",    metavar="ID", help="single resort to backfill")
    ap.add_argument("--year",      type=int,    help="single year to backfill")
    ap.add_argument("--from-year", type=int, default=2018, dest="from_year")
    ap.add_argument("--to-year",   type=int, default=date.today().year - 1, dest="to_year")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--verbose",   action="store_true", help="print each saved row")
    args = ap.parse_args()

    years = [args.year] if args.year else list(range(args.from_year, args.to_year + 1))

    if args.resort:
        targets = [args.resort]
    else:
        targets = list(URL_HISTORY.keys())

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for rid in targets:
        label = f"  [{rid}]  years={years}"
        if args.dry_run:
            label += "  (dry-run)"
        print(label)
        n = backfill_resort(rid, years, dry_run=args.dry_run, verbose=args.verbose)
        print(f"  [{rid}]  total new rows: {n}\n")
        total += n

    print(f"Done. Total new rows saved: {total}")


if __name__ == "__main__":
    main()
