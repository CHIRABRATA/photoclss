"""
utils/cluster.py — Normalised DBSCAN clustering on 512-D face embeddings.

Key improvements over the previous version:
  • L2-normalise embeddings *before* DBSCAN so cosine distances are well-behaved.
  • Default eps=0.50 with min_samples=1 — every face gets a cluster (no noise=-1).
  • Configurable via a sidebar slider exposed in app.py.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def cluster_face_embeddings(
    embeddings: np.ndarray,
    eps: float = 0.50,
    min_samples: int = 1,
) -> np.ndarray:
    """Cluster 512-D face embeddings using DBSCAN with cosine distance.

    Parameters
    ----------
    embeddings : ndarray of shape (N, 512)
        Raw or pre-normalised face embeddings.
    eps : float
        Maximum cosine distance between two samples in the same neighbourhood.
        Lower → stricter (more clusters); higher → looser (fewer clusters).
    min_samples : int
        Minimum number of faces required to form a cluster.  1 = every face
        gets its own cluster at worst (no noise label -1).

    Returns
    -------
    labels : ndarray of shape (N,)
        Cluster label for each embedding.  Labels start at 0.
    """
    if len(embeddings) == 0:
        return np.array([], dtype=int)

    # L2-normalise so ||x|| = 1 → cosine_distance = 1 − dot(a, b)
    normed = normalize(embeddings, norm="l2")

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clusterer.fit_predict(normed)

    return labels