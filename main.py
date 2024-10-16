import numpy as np
from sklearn.cluster import KMeans  # type: ignore

BITS2TYPE = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}


class CustomIndexPQ:

    def __init__(
        self,
        d: int,
        m: int,
        nbits: int,
        **estimator_kwargs: str | int
    ):
        if d % m != 0:
            raise ValueError("d must be divisible by m")

        self.m = m           # number of subquantizers
        self.k = 2**nbits    # number of centroids per subquantizer
        self.nbits = nbits   # number of bits per centroid
        self.ds = d // m     # dimensionality of each subquantizer

        self.estimators = [
            KMeans(
                n_clusters=self.k,
                **estimator_kwargs
            )
            for _ in range(self.m)
        ]

        self.is_trained = False
        self.dtype = BITS2TYPE[nbits]
        self.dtype_orig = np.float32
        self.codes: np.ndarray | None = None

    def train(self, X: np.ndarray) -> None:
        """Train all KMeans estimators on the input data."""
        if self.is_trained:
            raise ValueError("Index already trained.")
        
        _, d = X.shape
        if d != self.ds * self.m:
            raise ValueError(f"Invalid data dimensionality: {d}. Expected {self.ds * self.m}.")

        for i, estimator in enumerate(self.estimators):
            X_i = X[:, i * self.ds:(i + 1) * self.ds]
            estimator.fit(X_i)

        self.is_trained = True

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode the input data using the trained KMeans estimators."""
        if not self.is_trained:
            raise ValueError("The subquantizers have not been trained yet.")
        
        n, d = X.shape
        if d != self.ds * self.m:
            raise ValueError(f"Invalid data dimensionality: {d}. Expected {self.ds * self.m}.")
        
        res = np.empty((n, self.m), dtype=self.dtype)
        for i, estimator in enumerate(self.estimators):
            X_i = X[:, i * self.ds:(i + 1) * self.ds]
            res[:, i] = estimator.predict(X_i)
        
        return res
    
    def add(self, X: np.ndarray) -> None:
        """Add data to the index."""
        if not self.is_trained:
            raise ValueError("The subquantizers have not been trained yet.")
        
        if self.codes is None:
            self.codes = self.encode(X)
        else:
            self.codes = np.concatenate((self.codes, self.encode(X)), axis=0)

