from __future__ import annotations

import numpy as np


def as_2d(x) -> np.ndarray:
    """Return ``x`` as a 2D array; promotes 1D input to shape (1, n)."""
    a = np.asarray(x)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}.")
    return a
