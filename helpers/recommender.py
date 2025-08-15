from typing import List, Tuple, Dict, Any
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def recommend_similar_movies(base_movie: str, assets: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, float]]:
    cents = assets["centroids"]
    if base_movie not in cents or len(cents) < 2:
        return []
    base_vec = cents[base_movie]
    scored = []
    for m, v in cents.items():
        if m == base_movie:
            continue
        scored.append((m, _cosine(base_vec, v)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
