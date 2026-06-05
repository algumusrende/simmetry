from __future__ import annotations

import math

_EARTH_RADIUS_KM: float = 6371.0088
_EARTH_HALF_CIRCUMFERENCE_KM: float = math.pi * _EARTH_RADIUS_KM


def euclidean_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Similarity in [0, 1] between two 2D Cartesian points; 1.0 = same point."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return 1.0 / (1.0 + math.hypot(ax - bx, ay - by))


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometers between two (lat, lon) pairs.

    This is a utility function, not a registered similarity metric.
    Use ``haversine_sim`` for a [0, 1] similarity score.
    """
    lat1, lon1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lat2, lon2 = math.radians(float(b[0])), math.radians(float(b[1]))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    h = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    c = 2.0 * math.asin(min(1.0, math.sqrt(h)))

    return _EARTH_RADIUS_KM * c


def haversine_sim(
    a: tuple[float, float],
    b: tuple[float, float],
    scale_km: float = _EARTH_HALF_CIRCUMFERENCE_KM,
) -> float:
    """Similarity in [0, 1] between two (lat, lon) pairs.

    Scores 1.0 for the same point and approaches 0.0 for antipodal points.
    The default ``scale_km`` is half the Earth's circumference (~20 015 km).
    Pass a smaller ``scale_km`` to spread the score over a regional area.
    """
    return max(0.0, 1.0 - haversine_km(a, b) / scale_km)
