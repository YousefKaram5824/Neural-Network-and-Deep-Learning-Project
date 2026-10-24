import numpy as np


class PerceptronModel:
    def __init__(
        self,
        eta,
        epochs,
        use_bias,
        mse_threshold,
        random_state,
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

    @staticmethod
    def _signum(v):  # Threshold the linear output into +1 / -1
        return np.where(v >= 0, 1, -1)

    def _net_input(self, X):  # Linear score or net input
        return X @ self.w + self.b

    def mse_calc(self, X, y):  # Perceptron's cost (2 * err)
        y_pred = self._signum(self._net_input(X))
        err = y - y_pred
        return float(np.sum(err**2) / (2.0 * len(y)))

    def fitting(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        has_val = X_val is not None and y_val is not None

        rng = np.random.default_rng(self.random_state)
        self.w = rng.uniform(-0.01, 0.01, size=n_features)
        self.b = 0.0
        self.mse_history = []
        self.val_mse_history = []
        self.stopped_early = False

        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            stop_now = False

            for i, idx in enumerate(perm):
                xi, ti = X[idx], y[idx]
                v = self._net_input(xi)
                yi = 1 if v >= 0 else -1
                err = ti - yi

                if err != 0:
                    self.w += self.eta * err * xi
                    if self.use_bias:
                        self.b += self.eta * err

                if (i + 1) % self.early_stop_check_freq == 0 and i != n_samples - 1:
                    if self.mse_calc(X, y) < self.mse_threshold:
                        self.stopped_early = True
                        stop_now = True
                        break

            train_mse = self.mse_calc(X, y)
            self.mse_history.append(train_mse)

            if has_val:
                self.val_mse_history.append(self.mse_calc(X_val, y_val))

            self.epochs_run = epoch + 1

            if stop_now or train_mse < self.mse_threshold:
                break

        self.final_mse = self.mse_history[-1] if self.mse_history else None
        self.final_val_mse = self.val_mse_history[-1] if self.val_mse_history else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._signum(self._net_input(X))
