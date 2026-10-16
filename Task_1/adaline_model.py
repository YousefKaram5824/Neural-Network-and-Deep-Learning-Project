"""
Adaline (Adaptive Linear Neuron) binary classifier.

Features:
- Linear activation (y = w^T x + b) during training
- LMS / Widrow-Hoff update on every sample
- Shuffles training data every epoch
- Early stopping INSIDE the epoch
- Tracks validation MSE (on the test set) after each epoch
- Ensures the last sample in the epoch is always processed
"""

import numpy as np


class Adaline:
    def __init__(
        self,
        eta=0.01,
        epochs=500,
        use_bias=True,
        mse_threshold=0.001,
        random_state=None,
        early_stop_check_freq=5,
    ):
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.mse_threshold = mse_threshold
        self.random_state = random_state
        self.early_stop_check_freq = early_stop_check_freq

        self.w = None
        self.b = 0.0

        self.mse_history = []
        self.val_mse_history = []
        self.final_mse = None
        self.final_val_mse = None
        self.epochs_run = 0
        self.stopped_early = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _signum(v):
        return np.where(v >= 0, 1, -1)

    def _net_input(self, X):
        return X @ self.w + self.b

    def _mse(self, X, y):
        """MSE using LINEAR output (Adaline convention)."""
        v = self._net_input(X)
        err = y - v
        return float(np.sum(err**2) / (2.0 * len(y)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.w = rng.uniform(-0.01, 0.01, size=n_features)
        self.b = 0.0

        self.mse_history = []
        self.val_mse_history = []
        self.stopped_early = False

        for epoch in range(self.epochs):
            # ---- Shuffle every epoch ----
            perm = rng.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            stopped_this_epoch = False

            # ---- Loop over ALL samples ----
            for i, (xi, ti) in enumerate(zip(X_shuf, y_shuf)):
                v = xi @ self.w + self.b  # linear activation
                err = ti - v
                self.w += self.eta * err * xi
                if self.use_bias:
                    self.b += self.eta * err

                # ---- Early stopping INSIDE the epoch ----
                is_last = i == n_samples - 1
                if (not is_last) and ((i + 1) % self.early_stop_check_freq == 0):
                    current_mse = self._mse(X, y)
                    if current_mse < self.mse_threshold:
                        stopped_this_epoch = True
                        self.stopped_early = True
                        break

            # ---- End-of-epoch MSE ----
            train_mse = self._mse(X, y)
            self.mse_history.append(train_mse)

            # ---- Validation MSE ----
            if X_val is not None and y_val is not None:
                val_mse = self._mse(X_val, y_val)
                self.val_mse_history.append(val_mse)
                self.final_val_mse = val_mse

            self.epochs_run = epoch + 1

            if stopped_this_epoch or train_mse < self.mse_threshold:
                break

        self.final_mse = self.mse_history[-1] if self.mse_history else None
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """Return +1 / -1 for each row of X (signum on linear net input)."""
        X = np.asarray(X, dtype=float)
        return self._signum(self._net_input(X))
