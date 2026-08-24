from __future__ import annotations

import datetime as dt

US_HOLIDAYS: dict[dt.date, str] = {
    dt.date(2023, 12, 25): "Christmas",
    dt.date(2024, 1, 1): "New Year",
    dt.date(2024, 1, 15): "MLK Day",
    dt.date(2024, 2, 19): "Presidents Day",
    dt.date(2024, 5, 27): "Memorial Day",
    dt.date(2024, 6, 19): "Juneteenth",
    dt.date(2024, 7, 4): "Independence Day",
    dt.date(2024, 9, 2): "Labor Day",
    dt.date(2024, 10, 14): "Columbus Day",
    dt.date(2024, 11, 11): "Veterans Day",
    dt.date(2024, 11, 28): "Thanksgiving",
    dt.date(2024, 11, 29): "Day After Thanksgiving",
    dt.date(2024, 12, 24): "Christmas Eve",
    dt.date(2024, 12, 25): "Christmas",
    dt.date(2024, 12, 31): "New Year Eve",
    dt.date(2025, 1, 1): "New Year",
    dt.date(2025, 1, 20): "MLK Day",
    dt.date(2025, 2, 17): "Presidents Day",
    dt.date(2025, 5, 26): "Memorial Day",
    dt.date(2025, 6, 19): "Juneteenth",
    dt.date(2025, 7, 4): "Independence Day",
    dt.date(2025, 9, 1): "Labor Day",
    dt.date(2025, 10, 13): "Columbus Day",
    dt.date(2025, 11, 11): "Veterans Day",
    dt.date(2025, 11, 27): "Thanksgiving",
    dt.date(2025, 11, 28): "Day After Thanksgiving",
    dt.date(2025, 12, 24): "Christmas Eve",
    dt.date(2025, 12, 25): "Christmas",
    dt.date(2025, 12, 31): "New Year Eve",
    dt.date(2026, 1, 1): "New Year",
    dt.date(2026, 1, 19): "MLK Day",
    dt.date(2026, 2, 16): "Presidents Day",
    dt.date(2026, 5, 25): "Memorial Day",
}

HOLIDAY_DEMAND_MULTIPLIER: dict[str, float] = {
    "Christmas": 0.42,
    "Christmas Eve": 0.72,
    "Thanksgiving": 0.48,
    "Day After Thanksgiving": 0.83,
    "New Year": 0.66,
    "New Year Eve": 1.18,
    "Independence Day": 0.61,
    "Labor Day": 0.74,
    "Memorial Day": 0.76,
    "MLK Day": 0.88,
    "Presidents Day": 0.90,
    "Juneteenth": 0.87,
    "Columbus Day": 0.92,
    "Veterans Day": 0.94,
}


def holiday_name(day: dt.date) -> str | None:
    return US_HOLIDAYS.get(day)


def holiday_multiplier(day: dt.date) -> float:
    name = US_HOLIDAYS.get(day)
    return HOLIDAY_DEMAND_MULTIPLIER.get(name, 1.0) if name else 1.0
