from __future__ import annotations


def hamming_str(a: str, b: str) -> float:
    """Normalized Hamming similarity for equal-length strings; 1.0 = identical.

    Raises ``ValueError`` when lengths differ.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Inputs must have the same length for hamming_str, got {len(a)} vs {len(b)}."
        )
    if len(a) == 0:
        return 1.0
    return sum(ca == cb for ca, cb in zip(a, b)) / len(a)
