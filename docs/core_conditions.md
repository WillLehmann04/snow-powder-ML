# core/conditions.py

**Purpose:** Rule-based surface condition estimator — maps forecast weather inputs to a human-readable ski surface condition label.

**Inputs:** Called by `forecast.py` with per-day values from Open-Meteo forecast and the model's predicted snowfall.

**Outputs:** Returns one string condition label per forecast day. Labels match the taxonomy used by `collectors/resort_snapshot._CONDITION_RULES`:
- `powder`, `wind_affected`, `natural`, `ice`, `slush`, `spring`, `packed`, `groomed`, `variable`

**Key parameters / constants:**
Rules evaluated in priority order (first match wins):
1. `spring` — late season (NH: Mar/Apr, SH: Sep/Oct) AND temp_max ≥ 9°C
2. `powder` — pred_snow ≥ 15cm AND temp_max ≤ 1°C; or ≥ 10cm AND ≤ 2°C; or ≥ 6cm AND ≤ 0°C
3. `wind_affected` — pred_snow ≥ 4cm AND wind_max ≥ 60 km/h
4. `powder` (lighter) — pred_snow ≥ 5cm AND temp_max ≤ 3°C
5. `natural` — pred_snow ≥ 3cm AND temp_max ≤ 5°C
6. `ice` — temp_min ≤ -10°C AND temp_max ≥ 2°C AND pred_snow < 3cm
7. `slush` — temp_max ≥ 7°C AND snowfall_72h ≥ 3cm
8. `spring` — temp_max ≥ 8°C AND snowfall_72h ≥ 1cm
9. `packed` — pred_snow ≤ 2cm AND snowfall_72h ≥ 5cm AND temp_max ≤ 5°C
10. `groomed` — pred_snow ≤ 1cm AND snowfall_72h ≤ 3cm AND temp_max ≤ 6°C
11. `variable` — default fallback

**Notes:**
- This is intentionally deterministic — no ML training needed. Once enough labeled snapshot data accumulates (one full season from `collectors/resort_snapshot.py`), this can be replaced with an XGBoost classifier trained on `(weather features → condition_label)` pairs.
- The `pred_snow_cm` input is the model output after calibration and physical gating in `forecast.py`, not raw NWP snowfall. This means the thresholds are calibrated against the model's scale (which underestimates real snowfall, as documented in `forecast.py`).
- Does not account for existing base depth — the model has no signal for whether the mountain has a deep packed base underneath.

**Last updated:** 2026-06-07
