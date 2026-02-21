"""Approximate Nearest Neighbor (ANN) helpers (optional dependencies).

Install extras:
- pip install "simfast[ann]"       # hnswlib
- pip install "simfast[ann-hnsw]"  # hnswlib only
- pip install "simfast[ann-faiss]" # faiss-cpu only
"""

from .hnsw import HNSWIndex, build_hnsw
from .faiss_ import FaissIndex, build_faiss

__all__ = ["HNSWIndex", "build_hnsw", "FaissIndex", "build_faiss"]
