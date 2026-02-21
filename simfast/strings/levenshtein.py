from __future__ import annotations


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is shorter => less memory
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def levenshtein(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0,1]."""
    d = levenshtein_distance(a, b)
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - (d / m)
