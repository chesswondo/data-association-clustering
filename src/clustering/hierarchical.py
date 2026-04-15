import numpy as np
from .base import BaseClusterer

class HierarchicalClustering(BaseClusterer):
    """
    Agglomerative Hierarchical Clustering using Single Linkage
    (Nearest Neighbor Clustering).
    """
    def __init__(self, n_clusters: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        # Transductive algorithm, fit does nothing special without predict
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Hierarchical clustering is transductive. Use fit_predict() instead of predict()."
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        
        # Active clusters map: {cluster_id: list_of_point_indices}
        clusters = {i: [i] for i in range(n_samples)}
        
        # Point-to-cluster mapping for O(1) safe lookups
        point_to_cluster = {i: i for i in range(n_samples)}
        current_cluster_id = n_samples
        
        # Compute distance matrix
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        while len(clusters) > self.n_clusters:
            # Find the minimum distance pair
            min_idx = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
            idx1, idx2 = int(min_idx[0]), int(min_idx[1])
            
            # Safe O(1) lookup
            c1_id = point_to_cluster[idx1]
            c2_id = point_to_cluster[idx2]
            
            if c1_id == c2_id:
                dist_matrix[idx1, idx2] = np.inf
                dist_matrix[idx2, idx1] = np.inf
                continue
                
            # Merge clusters
            new_members = clusters[c1_id] + clusters[c2_id]
            clusters[current_cluster_id] = new_members
            
            # Update lookup map
            for pt in new_members:
                point_to_cluster[pt] = current_cluster_id
                
            # Delete old clusters
            del clusters[c1_id]
            del clusters[c2_id]
            
            # Mask distances inside the new merged cluster
            for i in new_members:
                for j in new_members:
                    dist_matrix[i, j] = np.inf
                    dist_matrix[j, i] = np.inf
                    
            current_cluster_id += 1
            
        # Format labels to 0, 1, 2...
        labels = np.zeros(n_samples, dtype=int)
        for label_idx, (cid, members) in enumerate(clusters.items()):
            for point_idx in members:
                labels[point_idx] = label_idx
                
        return labels