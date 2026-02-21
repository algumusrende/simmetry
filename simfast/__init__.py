from __future__ import annotations

from .api import pairwise, similarity, topk
from .index import SimIndex
from .registry import available, get, register

# Import metric implementations
from .points.core import euclidean_2d, haversine_km
from .sets.core import dice, jaccard, overlap
from .strings.jaro import jaro_winkler
from .strings.levenshtein import levenshtein
from .strings.ngrams import ngram_jaccard, token_jaccard
from .vectors.core import cosine, dot, euclidean_sim, manhattan_sim, pearson

# Register built-ins
register("cosine", cosine, kind="vector")
register("dot", dot, kind="vector")
register("euclidean_sim", euclidean_sim, kind="vector")
register("manhattan_sim", manhattan_sim, kind="vector")
register("pearson", pearson, kind="vector")

register("levenshtein", levenshtein, kind="string")
register("jaro_winkler", jaro_winkler, kind="string")
register("ngram_jaccard", ngram_jaccard, kind="string")
register("token_jaccard", token_jaccard, kind="string")

register("euclidean_2d", euclidean_2d, kind="point")
register("haversine_km", haversine_km, kind="point")

register("jaccard", jaccard, kind="set")
register("dice", dice, kind="set")
register("overlap", overlap, kind="set")

__all__ = [
    "similarity",
    "SimIndex",
    "pairwise",
    "topk",
    "register",
    "get",
    "available",
]
