"""
DEPRECATED — this was the original prototype pipeline.

All logic has been promoted to:
  collectors/weather.py   — fetch_hourly, fetch_and_cache
  core/features.py        — build_features

Do not use this file. It will be removed in a future cleanup.
"""

raise RuntimeError(
    "test.py is deprecated. Use collectors/weather.py and core/features.py instead."
)
