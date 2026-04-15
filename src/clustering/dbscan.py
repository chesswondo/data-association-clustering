import numpy as np
from .base import BaseClusterer

class CustomDBSCAN(BaseClusterer):
    """
    Density-Based Spatial Clustering of Applications with Noise.
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> 'CustomDBSCAN':
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DBSCAN is a transductive algorithm. Use fit_predict()."
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        # Labels: -2 means unvisited, -1 means noise, >= 0 are cluster IDs
        labels = np.full(n_samples, -2)
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Skip if already visited
            if labels[point_idx] != -2:
                continue
                
            # Find neighbors within eps radius
            neighbors = self._region_query(X, point_idx)
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                # Core point found, create a new cluster and expand it
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
                cluster_id += 1
                
        return labels

    def _region_query(self, X: np.ndarray, point_idx: int) -> list:
        """Find all points within eps distance from the given point."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray, point_idx: int, neighbors: list, cluster_id: int):
        """Expand the cluster using Breadth-First Search (BFS)."""
        # Assign the core point to the cluster
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If it was marked as noise, it's a border point. Assign to cluster.
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                
            # If it's completely unvisited
            elif labels[neighbor_idx] == -2:
                # Mark as part of the cluster
                labels[neighbor_idx] = cluster_id
                
                # Check if this neighbor is also a core point
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    # Append new neighbors to our BFS queue
                    neighbors.extend(new_neighbors)
            i += 1