from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MetricFn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFn
    kind: str


_REGISTRY: dict[str, Metric] = {}


def register(name: str, fn: MetricFn, kind: str = "generic") -> None:
    """Register a metric function under a unique name."""
    key = name.lower().strip()
    if key in _REGISTRY:
        raise ValueError(f"Metric already registered: {name}")
    _REGISTRY[key] = Metric(name=key, fn=fn, kind=kind)


def get(name: str) -> Metric:
    """Return a registered metric definition by name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]


def available(kind: str | None = None) -> tuple[str, ...]:
    """Return registered metric names, optionally filtered by kind."""
    if kind is None:
        return tuple(sorted(_REGISTRY.keys()))
    kind = kind.lower().strip()
    return tuple(sorted(k for k, m in _REGISTRY.items() if m.kind == kind))
