import numpy as np
from sklearn.cluster import DBSCAN

def cluster_face_embeddings(embeddings, eps=0.50, min_samples=1):
    """
    Clusters 512-D face embeddings using DBSCAN with Cosine Distance.
    Defaults to eps=0.50 and min_samples=1 for better performance on smaller photo sets.
    """
    if len(embeddings) == 0:
        return np.array([])

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
    
    return labels