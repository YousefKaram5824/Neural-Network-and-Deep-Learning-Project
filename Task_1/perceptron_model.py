"""
Single Layer Perceptron (binary classifier).

Features:
- Signum activation (non-linear, hard limiting)
- Update only when the predicted label differs from the target
    w <- w + eta * (t - y) * x
    b <- b + eta * (t - y)          (if bias enabled)
- Shuffles training data every epoch (model sees all data in random order)
- Early stopping INSIDE the epoch (checks MSE after every few updates)
- Tracks validation MSE (on the test set) after each epoch
- Ensures the last sample in the epoch is always processed
"""

import numpy as np


class Perceptron:
    def __init__(
        self,
        eta=0.01,
        epochs=500,
        use_bias=True,
        mse_threshold=0.001,
        random_state=None,
        early_stop_check_freq=5,
    ):
        """
        Parameters
        ----------
        eta                    : learning rate
        epochs                 : maximum number of epochs
        use_bias               : whether to include a bias term b
        mse_threshold          : stop when MSE < this threshold
        random_state           : seed for reproducibility
        early_stop_check_freq  : check MSE every N updates inside the epoch
        """
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.mse_threshold = mse_threshold
        self.random_state = random_state
        self.early_stop_check_freq = early_stop_check_freq

        # Learned parameters
        self.w = None
        self.b = 0.0

        # Diagnostics
        self.mse_history = []  # training MSE per epoch
        self.val_mse_history = []  # validation MSE per epoch
        self.final_mse = None
        self.final_val_mse = None
        self.epochs_run = 0
        self.stopped_early = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _signum(v):
        """Symmetric threshold: +1 for v >= 0, else -1."""
        return np.where(v >= 0, 1, -1)

    def _net_input(self, X):
        return X @ self.w + self.b

    def _mse(self, X, y):
        """MSE using signum output (perceptron convention)."""
        y_pred = self._signum(self._net_input(X))
        err = y - y_pred
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
            # ---- Shuffle training data each epoch ----
            perm = rng.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            stopped_this_epoch = False

            # ---- Loop over ALL training samples ----
            for i, (xi, ti) in enumerate(zip(X_shuf, y_shuf)):
                v = xi @ self.w + self.b
                yi = 1 if v >= 0 else -1
                err = ti - yi

                if err != 0:
                    self.w += self.eta * err * xi
                    if self.use_bias:
                        self.b += self.eta * err

                # ---- Early stopping check INSIDE the epoch ----
                # Don't check on the very last sample -- we want to finish
                # the epoch and record a full MSE first.
                is_last = i == n_samples - 1
                if (not is_last) and ((i + 1) % self.early_stop_check_freq == 0):
                    current_mse = self._mse(X, y)
                    if current_mse < self.mse_threshold:
                        stopped_this_epoch = True
                        self.stopped_early = True
                        break

            # ---- End-of-epoch MSE (on full training set) ----
            train_mse = self._mse(X, y)
            self.mse_history.append(train_mse)

            # ---- Validation MSE ----
            if X_val is not None and y_val is not None:
                val_mse = self._mse(X_val, y_val)
                self.val_mse_history.append(val_mse)
                self.final_val_mse = val_mse

            self.epochs_run = epoch + 1

            # ---- Stop condition ----
            if stopped_this_epoch or train_mse < self.mse_threshold:
                break

        self.final_mse = self.mse_history[-1] if self.mse_history else None
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        """Return +1 / -1 for each row of X."""
        X = np.asarray(X, dtype=float)
        return self._signum(self._net_input(X))
