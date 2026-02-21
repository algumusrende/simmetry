from __future__ import annotations
import time
import numpy as np
from simfast import pairwise, topk

def main():
    X = np.random.randn(5000, 128)
    t0 = time.time()
    S = pairwise(X[:1000], X[:1000], metric="cosine")
    t1 = time.time()
    print("pairwise cosine (1000x1000) seconds:", round(t1 - t0, 4), "shape:", S.shape)

    q = np.random.randn(128)
    t0 = time.time()
    idx, scores = topk(q, X, k=10, metric="cosine")
    t1 = time.time()
    print("topk cosine seconds:", round(t1 - t0, 6), "idx:", idx[:3], "scores:", scores[:3])

if __name__ == "__main__":
    main()
