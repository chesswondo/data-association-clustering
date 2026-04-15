import numpy as np
from abc import abstractmethod
from .base import BaseClusterer

class _BaseKClusterer(BaseClusterer):
    """
    Internal base class for centroid-based clustering (K-Means, K-Medians).
    Handles the Expectation-Maximization (EM) loop.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol  # Tolerance for early stopping
        self.random_state = random_state
        self.centroids = None

    @abstractmethod
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate distance between points and centroids. Must return shape (N, K)."""
        pass

    @abstractmethod
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate new centroids based on current cluster assignments."""
        pass

    def fit(self, X: np.ndarray) -> '_BaseKClusterer':
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Random Initialization
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_idx]
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Expectation step: Assign each point to the nearest centroid
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Maximization step: Update centroids
            self.centroids = self._update_centroids(X, labels)
            
            # Check for convergence: if centroids didn't move much, stop early
            shift = np.linalg.norm(self.centroids - old_centroids)
            if shift < self.tol:
                break
                
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Model is not fitted yet.")
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)


class KMeans(_BaseKClusterer):
    """K-Means clustering using Euclidean distance and Mean."""
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        # Broadcasting X (N, 1, D) - centroids (K, D) -> (N, K, D) -> norm along axis 2 -> (N, K)
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster loses all points, randomly reinitialize its centroid
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids


class KMedians(_BaseKClusterer):
    """K-Medians clustering using Manhattan distance and Median."""
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        # Manhattan distance: sum(|x - c|)
        return np.sum(np.abs(X[:, np.newaxis] - self.centroids), axis=2)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.median(cluster_points, axis=0)
            else:
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids