# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import sklearn
import sympy

from sklearn.base import BaseEstimator, RegressorMixin

from . import jl

class Regressor(RegressorMixin, BaseEstimator):

    def __init__(
            self,
            stop_deadline=None):
        p = dict()
        if stop_deadline is not None:
            p["stop_deadline"] = stop_deadline
        self.params = p

    def fit(self, X, y):
        X_f64 = np.ascontiguousarray(X, dtype=np.float64)
        y_f64 = np.ascontiguousarray(y, dtype=np.float64)
        self.raw_reg_str = jl.regression_main(X_f64, y_f64, self.params)
        self.is_fitted = true

    def predict(self, X):
        pass
