"""
Generic daily resort snow-report snapshot collector.

Polls each configured resort's snow-report API (XML or JSON) once per day
and appends the reading to:
    data/raw/snow_reports/{resort_id}.csv

Designed to run daily (e.g., 8am via Task Scheduler or cron) during ski
season.  After one or more seasons of collection the CSVs become proper
training labels for non-Japan resorts, replacing the Open-Meteo NWP proxy
used by calibrate_transfer.py.

The collector is idempotent — re-running on the same calendar day will not
create a duplicate row.

──────────────────────────────────────────────────────────────
Supported snow_report_format values (set in regions.yaml):
──────────────────────────────────────────────────────────────
  livepass_xml   Vail Resorts LivePass XML schema.
                 Used by: Thredbo, Perisher, Whistler, Vail, Breckenridge,
                          Park City, Stowe, Northstar, and ~30 other Vail
                          Resorts properties worldwide.
                 Fields:  snow24Hours, snow48Hours, snow72Hours, base, season

  alterra_json   Alterra Mountain Company JSON (planned).
                 Used by: Mammoth, Jackson Hole, Steamboat, Snowbird, etc.

──────────────────────────────────────────────────────────────
Adding a new global resort
──────────────────────────────────────────────────────────────
1. Add entry in regions.yaml with lat, lon, elevation, region, hemisphere,
   snow_report_url and snow_report_format.
2. For Vail Resorts properties the LivePass XML URL is typically:
     https://www.{resort-domain}/mountain-information/snow-report/
   or the centralised endpoint:
     https://www.vailresorts.com/b2c/xml/snow/SnowReport.aspx?mtnCode=XXX
3. Verify with:
     python -m collectors.resort_snapshot --resort {id} --dry-run
4. Schedule daily collection and after ≥1 season retrain with AU/global data.

──────────────────────────────────────────────────────────────
CLI
──────────────────────────────────────────────────────────────
  python -m collectors.resort_snapshot              # all configured resorts
  python -m collectors.resort_snapshot --resort thredbo
  python -m collectors.resort_snapshot --dry-run    # fetch + print, no write
  python -m collectors.resort_snapshot --list       # show configured resorts
  python -m collectors.resort_snapshot --show thredbo  # print parsed snapshot
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
import yaml

RAW_DIR  = Path("data/raw/snow_reports")
REGIONS  = Path("regions.yaml")

CSV_COLS = [
    "date", "resort_id",
    "snow_24h_cm", "snow_48h_cm", "snow_72h_cm",
    "base_cm", "season_cm",
]

HEADERS = {
    "User-Agent": "powder-ml-research/1.0 (will.lehmann4@gmail.com)",
    "Accept":     "application/xml, application/json, */*",
}
TIMEOUT = 30


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_livepass_xml(text: str) -> dict:
    """
    Parse a Vail Resorts LivePass XML snow report.

    Schema: LivePassSnowReport.Generic.xsd
    The root element is <snowReport>; it contains a <mountain> child with
    child elements <snow24Hours amount="N"/>, <snow48Hours/>, etc.
    """
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"XML parse error: {exc}") from exc

    # Root may be <snowReport> wrapping <mountain>, or <mountain> itself
    mountain = root.find("mountain") if root.tag != "mountain" else root
    if mountain is None:
        raise ValueError("No <mountain> element in LivePass XML response")

    def _amt(tag: str) -> float:
        el = mountain.find(tag)
        if el is None:
            return 0.0
        try:
            return max(0.0, float(el.get("amount") or 0))
        except (ValueError, TypeError):
            return 0.0

    return {
        "snow_24h_cm": _amt("snow24Hours"),
        "snow_48h_cm": _amt("snow48Hours"),
        "snow_72h_cm": _amt("snow72Hours"),
        "base_cm":     _amt("base"),
        "season_cm":   _amt("season"),
    }


_PARSERS: dict[str, callable] = {
    "livepass_xml": _parse_livepass_xml,
    # "alterra_json": _parse_alterra_json,  # add when needed
}


# ── Core functions ────────────────────────────────────────────────────────────

def fetch_snapshot(resort_id: str, url: str, fmt: str) -> dict:
    """
    Fetch and parse one resort's snow report.
    Returns a flat dict ready for CSV storage.
    """
    parser = _PARSERS.get(fmt)
    if parser is None:
        supported = list(_PARSERS)
        raise ValueError(
            f"Unsupported snow_report_format {fmt!r} for {resort_id}. "
            f"Supported: {supported}"
        )

    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    data = parser(resp.text)
    data["date"]      = date.today().isoformat()
    data["resort_id"] = resort_id
    return data


def save_snapshot(data: dict, dry_run: bool = False) -> str:
    """
    Append snapshot row to data/raw/snow_reports/{resort_id}.csv.
    Skips if a row for today already exists (idempotent).
    Returns a short status string.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{data['resort_id']}.csv"

    row = pd.DataFrame([data]).reindex(columns=CSV_COLS)

    if path.exists():
        existing = pd.read_csv(path, dtype={"date": str})
        if data["date"] in existing["date"].values:
            return f"already collected for {data['date']}, skipped"
        if not dry_run:
            row.to_csv(path, mode="a", header=False, index=False)
    else:
        if not dry_run:
            row.to_csv(path, index=False)

    suffix = " (dry-run)" if dry_run else ""
    return (
        f"saved{suffix}  "
        f"24h={data['snow_24h_cm']}cm  "
        f"48h={data['snow_48h_cm']}cm  "
        f"base={data['base_cm']}cm"
    )


def load_snapshots(resort_id: str) -> pd.DataFrame:
    """
    Load the accumulated snapshot CSV for a resort.
    Returns an empty DataFrame if no data collected yet.
    """
    path = RAW_DIR / f"{resort_id}.csv"
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def summary(resort_id: str) -> None:
    """Print a summary of collected snapshots for a resort."""
    df = load_snapshots(resort_id)
    if df.empty:
        print(f"  {resort_id}: no data collected yet")
        return
    print(f"  {resort_id}:")
    print(f"    rows:        {len(df)}")
    print(f"    date range:  {df['date'].min().date()} → {df['date'].max().date()}")
    snow = df["snow_24h_cm"]
    print(f"    snow_24h:    mean={snow.mean():.1f}  max={snow.max():.1f}  "
          f"p90={snow.quantile(0.9):.1f}cm")
    print(f"    powder days: {(snow >= 4).sum()} (≥4cm)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect daily snow-report snapshots from resort APIs"
    )
    ap.add_argument("--resort", metavar="ID",
                    help="Collect a single resort (default: all configured)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and print but do not write to CSV")
    ap.add_argument("--list", action="store_true",
                    help="List resorts that have snow_report_url configured")
    ap.add_argument("--show", metavar="ID",
                    help="Print summary of collected snapshots for a resort")
    args = ap.parse_args()

    with open(REGIONS) as f:
        regions = yaml.safe_load(f)

    # Resorts that have both required fields
    configured = {
        rid: cfg
        for rid, cfg in regions.items()
        if cfg.get("snow_report_url") and cfg.get("snow_report_format")
    }

    if args.list:
        if not configured:
            print("No resorts with snow_report_url configured in regions.yaml.")
            return
        print(f"{'Resort':<25}  {'Format':<15}  URL")
        print("-" * 80)
        for rid, cfg in configured.items():
            print(f"  {rid:<23}  {cfg['snow_report_format']:<15}  {cfg['snow_report_url']}")
        return

    if args.show:
        summary(args.show)
        return

    # ── Determine targets ─────────────────────────────────────────────────────
    if args.resort:
        if args.resort not in configured:
            if args.resort in regions:
                sys.exit(
                    f"  {args.resort} is in regions.yaml but has no "
                    f"snow_report_url / snow_report_format configured."
                )
            sys.exit(f"  Unknown resort: {args.resort}")
        targets = {args.resort: configured[args.resort]}
    else:
        targets = configured

    if not targets:
        sys.exit(
            "No resorts with snow_report_url configured in regions.yaml.\n"
            "Add snow_report_url and snow_report_format fields to a resort entry."
        )

    # ── Collect ───────────────────────────────────────────────────────────────
    ok, failed = 0, []
    for resort_id, cfg in targets.items():
        try:
            data   = fetch_snapshot(resort_id,
                                    cfg["snow_report_url"],
                                    cfg["snow_report_format"])
            status = save_snapshot(data, dry_run=args.dry_run)
            print(f"  [{resort_id}]  {status}")
            ok += 1
        except Exception as exc:
            print(f"  [{resort_id}]  ERROR: {exc}")
            failed.append(resort_id)

    print(f"\n  Done: {ok} collected, {len(failed)} failed"
          + (f": {failed}" if failed else ""))


if __name__ == "__main__":
    main()
