"""
Daily resort snow-report snapshot collector.

Polls each resort's snow-report API once per day and appends to:
    data/raw/snow_reports/{resort_id}.csv

Designed to run daily during ski season. After one or more seasons of
collection the CSVs become proper training labels for AU/NZ resorts,
replacing the ERA5 proxy used by calibrate_transfer.py.

──────────────────────────────────────────────────────────────
Supported snow_report_format values (set in regions.yaml):
──────────────────────────────────────────────────────────────
  livepass_xml      Vail Resorts LivePass XML schema.
                    Used by: Thredbo (works), any Vail property with XML endpoint.
                    Fields: snow24h, snow48h, snow72h, snow7d, base, season,
                            conditions_text, weather_top, weather_village,
                            bureau_forecast.

  falls_creek_json  Falls Creek dedicated WordPress JSON endpoint.
                    URL: /wp-content/uploads/FCSnowReport_2021.json
                    Fields: snow24h, base, conditions_text, forecast, temp_now,
                            temp_min, temp_max, wind_speed, wind_direction,
                            visibility, precipitation.

  perisher_html     Perisher (Joomla CMS) server-rendered HTML scraper.
                    Fields: snow24h, snow7d, base, conditions_text, forecast.

  whakapapa_lit     Whakapapa (Lit SSR web components) HTML scraper.
                    Fields: snow24h, snow7d, base, temp_now, wind_speed.

──────────────────────────────────────────────────────────────
CLI
──────────────────────────────────────────────────────────────
  python -m collectors.resort_snapshot              # all configured resorts
  python -m collectors.resort_snapshot --resort thredbo
  python -m collectors.resort_snapshot --dry-run    # fetch + print, no write
  python -m collectors.resort_snapshot --list       # show configured resorts
  python -m collectors.resort_snapshot --show thredbo
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
import yaml

RAW_DIR = Path("data/raw/snow_reports")
REGIONS = Path("regions.yaml")

CSV_COLS = [
    "date", "resort_id",
    "snow_24h_cm", "snow_48h_cm", "snow_72h_cm", "snow_7d_cm",
    "base_cm", "season_cm",
    "temp_now_c", "temp_min_c", "temp_max_c",
    "wind_speed_kmh", "wind_direction",
    "visibility", "precipitation",
    "conditions_text", "weather_top", "weather_village",
    "bureau_forecast", "groomer_comments",
    "condition_label",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
TIMEOUT = 30


# ── Condition extractor ───────────────────────────────────────────────────────

# Condition taxonomy — evaluated in priority order (first match wins)
_CONDITION_RULES: list[tuple[str, list[str]]] = [
    ("closed",        ["resort closed", "lifts closed", "closed for the season",
                       "no skiing", "no snowboarding", "pre-season", "not yet open"]),
    ("powder",        ["powder", "fresh snow", "fresh powder", "champagne powder",
                       "light dry", "dry fluffy", "blower pow", "untracked",
                       "overnight snow", "fresh overnight"]),
    ("wind_affected", ["wind affected", "wind packed", "wind slab", "wind board",
                       "windboard", "wind crust", "blown snow", "wind scoured"]),
    ("ice",           ["icy", " icy ", "ice patch", "frozen", "hard ice", "blue ice",
                       "bulletproof", "boilerplate", "freeze overnight",
                       "frozen overnight", "hard freeze", "icy patches"]),
    ("slush",         ["slush", "slushy", "wet heavy", "wet snow", "spring slush",
                       "heavy wet", "very wet", "wet and heavy"]),
    ("spring",        ["spring conditions", "spring skiing", "spring snow",
                       "corn snow", "spring corn", "afternoon slush",
                       "softening", "afternoon warm"]),
    ("groomed",       ["groomed", "well groomed", "perfectly groomed",
                       "machine groomed", "corduroy", "freshly groomed"]),
    ("packed",        ["packed powder", "packed", "machine packed",
                       "firm packed", "compacted", "firm"]),
    ("variable",      ["variable", "mixed conditions", "some icy", "some powder",
                       "patchy", "crusty in places", "mixed"]),
    ("natural",       ["natural snow", "ungroomed", "off piste", "off-piste",
                       "backcountry", "natural base"]),
    ("early_season",  ["opening weekend", "season opening", "early season",
                       "first day", "day 1 of winter", "day 2 of winter",
                       "making snow", "snowmaking", "guns cranking",
                       "snow guns", "getting white"]),
]


def classify_condition(text: str) -> str:
    """Return a condition label from free-text snow condition description."""
    if not text:
        return "unknown"
    t = text.lower()
    for label, keywords in _CONDITION_RULES:
        if any(kw in t for kw in keywords):
            return label
    # Fallback: if there's any mention of snow amount context
    if re.search(r'\d+\s*cm', t):
        return "natural"
    return "unknown"


def _clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


# ── Parser: LivePass XML ──────────────────────────────────────────────────────

def _parse_livepass_xml(text: str) -> dict:
    """Parse Vail Resorts LivePass XML snow report."""
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"XML parse error: {exc}") from exc

    mountain = root.find("mountain") if root.tag != "mountain" else root
    if mountain is None:
        raise ValueError("No <mountain> element in LivePass XML")

    def _amt(tag: str) -> float:
        el = mountain.find(tag)
        if el is None:
            return 0.0
        try:
            return max(0.0, float(el.get("amount") or 0))
        except (ValueError, TypeError):
            return 0.0

    def _msg(title: str) -> str:
        for el in mountain.findall("snowMessage"):
            t = (el.get("title") or "").strip()
            if t.lower() == title.lower():
                return _clean(el.text or "")
        return ""

    def _msg_default() -> str:
        """The untitled snowMessage (main conditions description)."""
        for el in mountain.findall("snowMessage"):
            if not (el.get("title") or "").strip():
                return _clean(el.text or "")
        return ""

    conditions = _msg_default()
    return {
        "snow_24h_cm":     _amt("snow24Hours"),
        "snow_48h_cm":     _amt("snow48Hours"),
        "snow_72h_cm":     _amt("snow72Hours"),
        "snow_7d_cm":      _amt("snow7Days"),
        "base_cm":         _amt("base"),
        "season_cm":       _amt("season"),
        "conditions_text": conditions,
        "weather_top":     _msg("Weather Top"),
        "weather_village": _msg("Weather Village"),
        "bureau_forecast": _msg("Bureau Forecast"),
        "groomer_comments": _msg("Groomer's Comments"),
        "condition_label": classify_condition(conditions),
    }


# ── Parser: Falls Creek JSON ──────────────────────────────────────────────────

def _parse_falls_creek_json(text: str) -> dict:
    """Parse Falls Creek WordPress JSON snow report."""
    import json as _json
    data = _json.loads(text)

    patrol   = data.get("Patrol", {})
    weather  = data.get("Weather", {})
    overview = data.get("Overview", {})
    maint    = data.get("SlopeMaintenance", {})

    def _num(val, default=0.0) -> float:
        try:
            return float(str(val).replace("cm", "").replace("°", "").strip())
        except (ValueError, TypeError):
            return default

    conditions = _clean(patrol.get("PatrolSnowCover", ""))
    forecast   = _clean(patrol.get("PatrolForecast", ""))

    # Wind speed: "1.8 km/h" → 1.8
    wind_raw = str(weather.get("WindSpeed", "0"))
    wind_speed = _num(re.sub(r"[^\d.]", "", wind_raw))

    return {
        "snow_24h_cm":     _num(patrol.get("PatrolFreshSnow", 0)),
        "snow_7d_cm":      0.0,
        "base_cm":         _num(patrol.get("PatrolNaturalSnowDepth", 0)),
        "season_cm":       _num(patrol.get("SeasonalSnowfallToDate", 0)),
        "temp_now_c":      _num(re.sub(r"[^\d.\-]", "", str(weather.get("TempNow", "0")))),
        "temp_min_c":      _num(re.sub(r"[^\d.\-]", "", str(weather.get("TodaysLow", "0")))),
        "temp_max_c":      _num(re.sub(r"[^\d.\-]", "", str(weather.get("TodaysHigh", "0")))),
        "wind_speed_kmh":  wind_speed,
        "wind_direction":  str(weather.get("WindDirection", "")),
        "visibility":      str(patrol.get("PatrolVisibility", "")),
        "precipitation":   str(patrol.get("PatrolPrecipitation", "")),
        "conditions_text": conditions,
        "bureau_forecast": forecast,
        "groomer_comments": _clean(maint.get("Comments", "")),
        "condition_label": classify_condition(conditions),
    }


# ── Parser: Perisher HTML ─────────────────────────────────────────────────────

def _parse_perisher_html(text: str) -> dict:
    """Parse Perisher Joomla server-rendered HTML snow report."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    def _num_after(pattern: str) -> float:
        m = re.search(pattern, clean, re.I)
        return float(m.group(1)) if m else 0.0

    snow_24h = _num_after(r"Snowfall\s+24\s+Hr[s]?\s+([\d.]+)\s*cm")
    snow_7d  = _num_after(r"7\s+Days?\s+([\d.]+)\s*cm")
    # "Natural Depth* 22 cm" — allow for asterisk between label and value
    base     = _num_after(r"Natural\s+Depth\s*\*?\s*([\d.]+)\s*cm")

    # Temps: "Top: 5.9°C Village: 9.7°C"
    temp_top = _num_after(r"Top:\s*([\-\d.]+)\s*°?C")
    # Wind: "2 km/h S"
    wind_m   = re.search(r"([\d.]+)\s*km/h\s*([NSEW]{1,3})", clean, re.I)

    # Forecast: "Today: 9°C Tonight: -3°C 4 km/h WSW"
    forecast_m = re.search(r"Forecast\s+Today[:\s]+(.{10,200?}?)(?:Check out|Red Energy|17 June|Pass)", clean, re.I)
    forecast = forecast_m.group(1).strip() if forecast_m else ""
    # Clean degree entity
    forecast = forecast.replace("&deg;", "°")

    # Conditions: Perisher has a Snow Cover / daily briefing text (mid-season).
    # Avoid matching promo copy or road/lift headings.
    cond_m = re.search(
        r"(?:Snow\s+Cover|Snow\s+Conditions?|Conditions?\s+Report)[:\s]+(.{30,500}?)"
        r"(?:Lifts?\s+Open|Road\s+Conditions?|Weather\s+Forecast|Groomed|\d+\s+cm|\Z)",
        clean, re.I,
    )
    candidates = [cond_m.group(1).strip()] if cond_m else []
    # Filter out strings that are clearly not condition descriptions
    conditions = next(
        (c for c in candidates if len(c.split()) > 5 and not c.startswith("Supported")),
        "",
    )
    if not conditions and forecast:
        conditions = f"Forecast: {forecast}"

    return {
        "snow_24h_cm":     snow_24h,
        "snow_7d_cm":      snow_7d,
        "base_cm":         base,
        "temp_now_c":      temp_top,
        "wind_speed_kmh":  float(wind_m.group(1)) if wind_m else 0.0,
        "wind_direction":  wind_m.group(2) if wind_m else "",
        "conditions_text": conditions[:500],
        "bureau_forecast": forecast[:300],
        "condition_label": classify_condition(conditions),
    }


# ── Parser: Whakapapa Lit SSR HTML ───────────────────────────────────────────

def _parse_whakapapa_lit(text: str) -> dict:
    """Parse Whakapapa Lit SSR web-component HTML snow report."""
    clean = re.sub(r"<!--[^>]*-->", "", text)  # remove lit comments
    clean_text = re.sub(r"<[^>]+>", " ", clean)
    clean_text = re.sub(r"\s+", " ", clean_text)

    def _after_label(label: str) -> float:
        m = re.search(rf"{re.escape(label)}\s*([\d.]+)\s*(?:cm|°)?", clean_text, re.I)
        return float(m.group(1)) if m else 0.0

    def _str_after(label: str) -> str:
        m = re.search(rf"{re.escape(label)}\s*([^\d\s][^.!?\n]{{2,60}})", clean_text, re.I)
        return m.group(1).strip() if m else ""

    snow_24h  = _after_label("24 hr Snowfall")
    snow_7d   = _after_label("7 day Snowfall")
    base      = _after_label("Snow Base")
    temp_now  = _after_label("Temperature")
    wind_raw  = re.search(r"Wind\s*([\d.]+)\s*km/h\s*([NSEW]{1,3})", clean_text, re.I)
    wind_spd  = float(wind_raw.group(1)) if wind_raw else 0.0
    wind_dir  = wind_raw.group(2) if wind_raw else ""

    # Conditions: look for any free-text paragraph near snow data
    conditions_match = re.search(
        r"(?:conditions?|report)[:\s]+(.{20,400?}?)(?:Lifts|Road|Trail|Webcam|$)",
        clean_text, re.I,
    )
    conditions = conditions_match.group(1).strip() if conditions_match else ""

    return {
        "snow_24h_cm":    snow_24h,
        "snow_7d_cm":     snow_7d,
        "base_cm":        base,
        "temp_now_c":     temp_now,
        "wind_speed_kmh": wind_spd,
        "wind_direction": wind_dir,
        "conditions_text": conditions[:400],
        "condition_label": classify_condition(conditions),
    }


_PARSERS: dict[str, callable] = {
    "livepass_xml":      _parse_livepass_xml,
    "falls_creek_json":  _parse_falls_creek_json,
    "perisher_html":     _parse_perisher_html,
    "whakapapa_lit":     _parse_whakapapa_lit,
}


# ── Core functions ────────────────────────────────────────────────────────────

def fetch_snapshot(resort_id: str, url: str, fmt: str) -> dict:
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise ValueError(f"Unsupported snow_report_format {fmt!r}. Supported: {list(_PARSERS)}")

    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    data = parser(resp.text)
    data["date"]      = date.today().isoformat()
    data["resort_id"] = resort_id
    return data


def save_snapshot(data: dict, dry_run: bool = False) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{data['resort_id']}.csv"

    row = pd.DataFrame([data]).reindex(columns=CSV_COLS)
    for col in CSV_COLS:
        if col not in row.columns:
            row[col] = None

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
    snow = data.get("snow_24h_cm", 0)
    base = data.get("base_cm", 0)
    cond = data.get("condition_label", "?")
    return f"saved{suffix}  24h={snow}cm  base={base}cm  condition={cond}"


def load_snapshots(resort_id: str) -> pd.DataFrame:
    path = RAW_DIR / f"{resort_id}.csv"
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def summary(resort_id: str) -> None:
    df = load_snapshots(resort_id)
    if df.empty:
        print(f"  {resort_id}: no data collected yet")
        return
    print(f"  {resort_id}:")
    print(f"    rows:        {len(df)}")
    print(f"    date range:  {df['date'].min().date()} → {df['date'].max().date()}")
    snow = df["snow_24h_cm"].fillna(0)
    print(f"    snow_24h:    mean={snow.mean():.1f}  max={snow.max():.1f}  "
          f"p90={snow.quantile(0.9):.1f}cm")
    print(f"    powder days: {(snow >= 4).sum()} (≥4cm)")
    if "condition_label" in df.columns:
        counts = df["condition_label"].value_counts()
        print(f"    conditions:  {dict(counts)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect daily snow-report snapshots")
    ap.add_argument("--resort", metavar="ID")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="ID")
    args = ap.parse_args()

    with open(REGIONS) as f:
        regions = yaml.safe_load(f)

    configured = {
        rid: cfg
        for rid, cfg in regions.items()
        if cfg.get("snow_report_url") and cfg.get("snow_report_format")
    }

    if args.list:
        if not configured:
            print("No resorts with snow_report_url configured in regions.yaml.")
            return
        print(f"{'Resort':<25}  {'Format':<20}  URL")
        print("-" * 90)
        for rid, cfg in configured.items():
            print(f"  {rid:<23}  {cfg['snow_report_format']:<20}  {cfg['snow_report_url']}")
        return

    if args.show:
        summary(args.show)
        return

    if args.resort:
        if args.resort not in configured:
            if args.resort in regions:
                sys.exit(f"  {args.resort} has no snow_report_url configured.")
            sys.exit(f"  Unknown resort: {args.resort}")
        targets = {args.resort: configured[args.resort]}
    else:
        targets = configured

    if not targets:
        sys.exit("No resorts configured. Add snow_report_url and snow_report_format to regions.yaml.")

    ok, failed = 0, []
    for resort_id, cfg in targets.items():
        try:
            data   = fetch_snapshot(resort_id, cfg["snow_report_url"], cfg["snow_report_format"])
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
