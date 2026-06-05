from __future__ import annotations

from .api import infer_metric, pairwise, similarity, topk
from .index import SimIndex
from .points.core import euclidean_2d, haversine_km, haversine_sim
from .registry import available, get, register
from .sets.core import dice, jaccard, overlap, tversky
from .strings.jaro import jaro_winkler
from .strings.levenshtein import levenshtein
from .strings.ngrams import ngram_jaccard, token_jaccard
from .vectors.core import cosine, cosine_distance, dot, euclidean_sim, hamming, manhattan_sim, pearson

register("cosine", cosine, kind="vector")
register("cosine_distance", cosine_distance, kind="vector")
register("dot", dot, kind="vector")
register("euclidean_sim", euclidean_sim, kind="vector")
register("hamming", hamming, kind="vector")
register("manhattan_sim", manhattan_sim, kind="vector")
register("pearson", pearson, kind="vector")

register("levenshtein", levenshtein, kind="string")
register("jaro_winkler", jaro_winkler, kind="string")
register("ngram_jaccard", ngram_jaccard, kind="string")
register("token_jaccard", token_jaccard, kind="string")

register("euclidean_2d", euclidean_2d, kind="point")
register("haversine_sim", haversine_sim, kind="point")

register("jaccard", jaccard, kind="set")
register("dice", dice, kind="set")
register("overlap", overlap, kind="set")
register("tversky", tversky, kind="set")

__all__ = [
    "similarity",
    "infer_metric",
    "SimIndex",
    "pairwise",
    "topk",
    "register",
    "get",
    "available",
    "haversine_km",
]
