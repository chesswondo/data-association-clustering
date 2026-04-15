from abc import ABC, abstractmethod
import numpy as np

class BaseClusterer(ABC):
    """Abstract base class for all clustering algorithms."""
    
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Fit the clustering model to the data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels for the training data."""
        self.fit(X)
        return self.predict(X)