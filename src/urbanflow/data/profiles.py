from __future__ import annotations

import numpy as np

CORE_WEEKDAY = np.array(
    [
        0.34, 0.21, 0.14, 0.10, 0.09, 0.14, 0.34, 0.68,
        0.95, 0.98, 0.86, 0.84, 0.88, 0.90, 0.92, 0.95,
        1.00, 1.12, 1.15, 1.02, 0.90, 0.82, 0.68, 0.50,
    ]
)

CORE_SATURDAY = np.array(
    [
        0.86, 0.72, 0.55, 0.36, 0.22, 0.15, 0.16, 0.22,
        0.34, 0.50, 0.66, 0.78, 0.86, 0.90, 0.92, 0.93,
        0.95, 0.98, 1.00, 1.00, 0.98, 0.96, 0.94, 0.92,
    ]
)

CORE_SUNDAY = np.array(
    [
        0.90, 0.80, 0.64, 0.44, 0.26, 0.16, 0.15, 0.19,
        0.29, 0.45, 0.62, 0.75, 0.83, 0.87, 0.89, 0.90,
        0.90, 0.88, 0.84, 0.76, 0.66, 0.56, 0.46, 0.38,
    ]
)

AIRPORT_WEEKDAY = np.array(
    [
        0.20, 0.10, 0.06, 0.05, 0.08, 0.24, 0.52, 0.72,
        0.84, 0.92, 0.98, 1.00, 0.97, 0.94, 0.94, 0.96,
        1.00, 1.02, 1.00, 0.94, 0.84, 0.70, 0.52, 0.34,
    ]
)

AIRPORT_WEEKEND = np.array(
    [
        0.24, 0.12, 0.07, 0.05, 0.08, 0.22, 0.46, 0.66,
        0.80, 0.90, 0.97, 1.00, 0.99, 0.96, 0.95, 0.96,
        0.99, 1.00, 0.97, 0.90, 0.80, 0.66, 0.48, 0.32,
    ]
)

OUTER_WEEKDAY = np.array(
    [
        0.30, 0.20, 0.14, 0.11, 0.12, 0.22, 0.44, 0.72,
        0.88, 0.84, 0.74, 0.72, 0.74, 0.78, 0.84, 0.92,
        1.00, 1.04, 1.00, 0.90, 0.78, 0.66, 0.52, 0.40,
    ]
)

OUTER_WEEKEND = np.array(
    [
        0.60, 0.46, 0.34, 0.24, 0.18, 0.16, 0.22, 0.34,
        0.48, 0.62, 0.74, 0.82, 0.88, 0.90, 0.92, 0.94,
        0.96, 0.98, 1.00, 0.96, 0.88, 0.80, 0.74, 0.68,
    ]
)


def hour_of_week_profile(kind: str) -> np.ndarray:
    if kind == "airport":
        weekday, saturday, sunday = AIRPORT_WEEKDAY, AIRPORT_WEEKEND, AIRPORT_WEEKEND
    elif kind == "outer":
        weekday, saturday, sunday = OUTER_WEEKDAY, OUTER_WEEKEND, OUTER_WEEKEND
    else:
        weekday, saturday, sunday = CORE_WEEKDAY, CORE_SATURDAY, CORE_SUNDAY
    week = np.concatenate([weekday] * 5 + [saturday, sunday])
    friday = slice(4 * 24, 5 * 24)
    week[friday] = np.maximum(week[friday], CORE_SATURDAY * 0.92)
    return week


PROFILES: dict[str, np.ndarray] = {
    kind: hour_of_week_profile(kind) for kind in ("core", "outer", "airport")
}
