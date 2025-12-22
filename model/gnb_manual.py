import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = float(var_smoothing)

    def _to_numpy(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.asarray(X)

    def fit(self, X, y):
        X = self._to_numpy(X).astype(float)
        y = self._to_numpy(y).ravel()

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.class_prior_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for k in range(n_classes):
            Xk = X[y_idx == k]
            self.class_prior_[k] = Xk.shape[0] / n_samples
            self.theta_[k] = Xk.mean(axis=0)
            self.var_[k] = Xk.var(axis=0)

        eps = self.var_smoothing * max(self.var_.max(), 1.0)
        self.var_ += eps
        return self

    def _joint_log_likelihood(self, X):
        X = self._to_numpy(X)
        jll = []
        for i in range(len(self.classes_)):
            prior = np.log(self.class_prior_[i])
            likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[i]) +
                ((X - self.theta_[i]) ** 2) / self.var_[i],
                axis=1
            )
            jll.append(prior + likelihood)
        return np.array(jll).T

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
