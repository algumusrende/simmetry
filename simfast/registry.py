from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

MetricFn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFn
    kind: str  # "string" | "vector" | "point" | "set" | "generic"


_REGISTRY: Dict[str, Metric] = {}


def register(name: str, fn: MetricFn, kind: str = "generic") -> None:
    key = name.lower().strip()
    if key in _REGISTRY:
        raise ValueError(f"Metric already registered: {name}")
    _REGISTRY[key] = Metric(name=key, fn=fn, kind=kind)


def get(name: str) -> Metric:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]


def available(kind: Optional[str] = None) -> Tuple[str, ...]:
    if kind is None:
        return tuple(sorted(_REGISTRY.keys()))
    kind = kind.lower().strip()
    return tuple(sorted(k for k, m in _REGISTRY.items() if m.kind == kind))
