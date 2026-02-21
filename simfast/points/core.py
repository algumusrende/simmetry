from __future__ import annotations

import math
from typing import Tuple


def euclidean_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    d = math.hypot(ax - bx, ay - by)
    return 1.0 / (1.0 + d)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return similarity = 1/(1+distance_km) using Haversine distance."""
    lat1, lon1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lat2, lon2 = math.radians(float(b[0])), math.radians(float(b[1]))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    h = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    c = 2.0 * math.asin(min(1.0, math.sqrt(h)))

    R = 6371.0088  # mean Earth radius (km)
    dist_km = R * c
    return 1.0 / (1.0 + dist_km)
