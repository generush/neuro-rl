# utils/scalers.py

import numpy as np

class RangeScaler:
    def __init__(self):
        self.mean_ = None
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1  # Prevent division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.range_

    def inverse_transform(self, X_scaled):
        return X_scaled * self.range_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RangeSoftScaler(RangeScaler):
    def __init__(self, softening_factor=5):
        super().__init__()  # Initialize parent class
        self.softening_factor = softening_factor

    def fit(self, X):
        super().fit(X)  # Call the fit method from RangeScaler
        # Apply softening factor only to non-zero ranges
        self.range_ += self.softening_factor
        return self