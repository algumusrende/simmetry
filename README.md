# simfast

Blazing-fast similarity scores for **strings**, **vectors**, **points**, and **sets** — with a simple API.

## Install

```bash
pip install simfast
pip install "simfast[fast]"
```

## Quickstart

### One function
```python
from simfast import similarity

similarity("kitten", "sitting", metric="levenshtein")     
similarity([1,2,3], [1,2,4], metric="cosine")             
similarity((41.1, 29.0), (41.2, 29.1), metric="haversine_km")
similarity({1,2,3}, {2,3,4}, metric="jaccard")
```

### Pairwise matrices (fast for vectors)
```python
import numpy as np
from simfast import pairwise

X = np.random.randn(1000, 128)
S = pairwise(X, metric="cosine")          
```

### Top-k search
```python
import numpy as np
from simfast import topk

X = np.random.randn(5000, 64)
q = np.random.randn(64)
idx, scores = topk(q, X, k=10, metric="cosine")
```

## Available metrics

```python
from simfast import available
available()             
available("vector")     
available("string")     
available("point")      
available("set")        
```

### Vectors
- `cosine`, `dot`, `euclidean_sim`, `manhattan_sim`, `pearson`

### Strings
- `levenshtein` (normalized similarity)
- `jaro_winkler`
- `ngram_jaccard` (character n-gram set Jaccard)
- `token_jaccard` (whitespace token set Jaccard)

### Points / Geo
- `euclidean_2d`
- `haversine_km`

### Sets
- `jaccard`, `dice`, `overlap`

## License
MIT
## Batch string APIs

If you need many string-to-string similarities (e.g., deduping names), use:

```python
from simfast.strings import pairwise_strings, topk_strings

S = pairwise_strings(["ali", "veli"], ["ali", "val"], metric="jaro_winkler")
idx, scores = topk_strings("akbank", ["akbank", "isbank", "yapikredi"], k=2, metric="levenshtein")
```

## ANN top-k (optional, does NOT bloat core)

For very large vector corpora (100k+), exact `topk()` can be slow. ANN gives fast approximate results.

### hnswlib (recommended)
```bash
pip install "simfast[ann-hnsw]"
```

```python
import numpy as np
from simfast.ann import build_hnsw

X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)  

index = build_hnsw(X, space="cosine")
labels, distances = index.query(X[0], k=10)
```

### faiss
```bash
pip install "simfast[ann-faiss]"
```

```python
import numpy as np
from simfast.ann import build_faiss

X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)  

index = build_faiss(X, metric="ip")
labels, scores = index.query(X[0], k=10)
```


## SimIndex (exact or ANN)

Exact search (no extras):

```python
import numpy as np
from simfast import SimIndex

X = np.random.randn(50_000, 128).astype("float32")
index = SimIndex(metric="cosine", backend="exact").add(X)

idx, scores = index.query(X[0], k=10)
```

ANN (optional):

```bash
pip install "simfast[ann-hnsw]"
```

```python
import numpy as np
from simfast import SimIndex

X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)

index = SimIndex(metric="cosine", backend="hnsw").add(X)
labels, distances = index.query(X[0], k=10)
```

## Auto similarity and composite records

Auto metric selection:

```python
from simfast import similarity

similarity("akbank", "ak bank")      
similarity((41.0, 29.0), (41.1, 29.1)) 
similarity({1,2,3}, {2,3,4})         
```

Composite similarity over dict fields:

```python
a = {"name": "Ali Can", "city": "Istanbul", "loc": (41.0, 29.0)}
b = {"name": "Ali Can Gumus", "city": "İstanbul", "loc": (41.01, 28.99)}

score = similarity(
    a, b,
    metric={"name": "jaro_winkler", "loc": "haversine_km"},
    weights={"name": 0.7, "loc": 0.3},
)
```
