from __future__ import annotations


def bm25(query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 text relevance score normalized to [0, 1]; 1.0 when document == query.

    Uses BM25 term-frequency weighting with uniform IDF (no external corpus
    needed). Designed as a ranking helper — use with ``topk_strings(...,
    metric="bm25")`` to rank documents against a query.

    Args:
        query: Query string (whitespace-tokenized).
        document: Document string to score.
        k1: Term-frequency saturation parameter (default 1.5).
        b: Document-length normalisation factor (default 0.75).

    Returns:
        Score in [0, 1].
    """
    q_tokens = query.split()
    d_tokens = document.split()

    if not q_tokens and not d_tokens:
        return 1.0
    if not q_tokens or not d_tokens:
        return 0.0

    d_len = len(d_tokens)
    q_len = len(q_tokens)

    tf_doc: dict[str, int] = {}
    for t in d_tokens:
        tf_doc[t] = tf_doc.get(t, 0) + 1

    tf_query: dict[str, int] = {}
    for t in q_tokens:
        tf_query[t] = tf_query.get(t, 0) + 1

    def _score(q_toks: list[str], d_tf: dict[str, int], dl: int) -> float:
        s = 0.0
        for t in q_toks:
            tf = d_tf.get(t, 0)
            s += (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * dl / q_len))
        return s

    raw = _score(q_tokens, tf_doc, d_len)
    best = _score(q_tokens, tf_query, q_len)

    if best == 0.0:
        return 0.0
    return min(1.0, raw / best)
