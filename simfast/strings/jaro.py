from __future__ import annotations


def _jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    match_dist = max(0, max(la, lb) // 2 - 1)

    a_matches = [False] * la
    b_matches = [False] * lb

    matches = 0
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    t = 0
    j = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[j]:
            j += 1
        if a[i] != b[j]:
            t += 1
        j += 1
    transpositions = t / 2

    return (matches / la + matches / lb + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a: str, b: str, prefix_scale: float = 0.1, max_prefix: int = 4) -> float:
    j = _jaro(a, b)
    p = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        p += 1
        if p == max_prefix:
            break
    return j + p * prefix_scale * (1.0 - j)
