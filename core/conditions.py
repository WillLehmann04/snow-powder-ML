"""
Rule-based surface condition estimator for ski resort forecasts.

Rules evaluated in priority order — the first match wins. To see what each day
receives, check: pred_snow_cm (model output), snowfall_72h (NWP rolling 72h sum),
temp_min/temp_max, wind_max, month, hemisphere.

Condition taxonomy:
  powder        fresh light dry snow dominant
  wind_affected fresh snow + strong wind (slabs, scoured sections)
  natural       moderate fresh snow (2–5 cm), not a full powder day
  ice           freeze-thaw cycle or extreme cold bulletproof surface
  slush         warm daytime + meaningful recent snow = wet heavy surface
  spring        late-season softening / corn snow
  packed        settled old snow without fresh top layer (post-storm)
  groomed       groomed piste dominant, minimal recent snow
  early_season  opening weeks: snowmaking / thin cover (last resort before variable)
  variable      genuinely ambiguous surface — should rarely appear
"""

from __future__ import annotations

_EARLY_MONTHS = {"south": {6}, "north": {11, 12}}
_LATE_MONTHS  = {"south": {9, 10}, "north": {3, 4}}


def estimate_condition(
    pred_snow_cm: float,
    snowfall_72h: float,
    temp_min: float,
    temp_max: float,
    wind_max: float,
    month: int,
    hemisphere: str = "north",
) -> str:
    """Return a surface condition label from forecast weather inputs.

    Parameters
    ----------
    pred_snow_cm : float  Model-predicted overnight snowfall (cm).
    snowfall_72h : float  3-day accumulated NWP snowfall (cm) — rolling window.
    temp_min     : float  Daily min temperature (°C).
    temp_max     : float  Daily max temperature (°C).
    wind_max     : float  Daily max wind speed (km/h).
    month        : int    Calendar month (1–12).
    hemisphere   : str    "north" or "south".
    """
    hemi = hemisphere if hemisphere in _EARLY_MONTHS else "north"
    late_season = month in _LATE_MONTHS[hemi]

    # ── Spring: late season + warm daytime ────────────────────────────────────
    if late_season and temp_max >= 9:
        return "spring"

    # ── Powder ────────────────────────────────────────────────────────────────
    if pred_snow_cm >= 15 and temp_max <= 1:
        return "powder"
    if pred_snow_cm >= 10 and temp_max <= 2:
        return "powder"
    if pred_snow_cm >= 6  and temp_max <= 0:
        return "powder"

    # ── Wind affected: fresh snow + strong wind ────────────────────────────────
    if pred_snow_cm >= 4 and wind_max >= 60:
        return "wind_affected"

    # ── More powder: lighter amounts + cold ───────────────────────────────────
    if pred_snow_cm >= 5 and temp_max <= 3:
        return "powder"

    # ── Natural: 2–5 cm fresh snow, cold enough ───────────────────────────────
    if pred_snow_cm >= 2 and temp_max <= 5:
        return "natural"

    # ── Below here: pred_snow_cm < 2 OR temp_max > 5 ─────────────────────────

    # ── Ice: strong freeze-thaw (very cold nights, warm days, no fresh snow) ──
    if temp_min <= -10 and temp_max >= 2 and pred_snow_cm < 2:
        return "ice"

    # ── Slush: warm daytime + meaningful recent snowfall ─────────────────────
    # snowfall_72h captures the rolling NWP accumulation; pred_snow_cm covers
    # today's snow falling into warm conditions.
    if temp_max >= 7 and (snowfall_72h + pred_snow_cm) >= 3:
        return "slush"

    # ── Spring softening: late season, some recent snow, warm ────────────────
    if late_season and temp_max >= 7 and (snowfall_72h + pred_snow_cm) >= 1:
        return "spring"

    # ── Packed: settled old snow from recent heavy falls ─────────────────────
    # Fires after a big storm day when pred_snow_cm is small but 72h sum is large.
    if pred_snow_cm <= 2 and snowfall_72h >= 8 and temp_max <= 5:
        return "packed"

    # ── Groomed: minimal fresh snow, mountain operational ────────────────────
    # Threshold 14°C — groomers run regardless of afternoon temp up to ~14°C.
    if pred_snow_cm <= 2 and snowfall_72h <= 6 and temp_max <= 14:
        return "groomed"

    # ── Early season: opening month, minimal snow activity ───────────────────
    # Only fires as last resort in known opening months.
    if month in _EARLY_MONTHS.get(hemi, set()):
        return "early_season"

    return "variable"
