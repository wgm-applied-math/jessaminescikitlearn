# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime as dt
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
        if stop_deadline is None:
            n = dt.datetime.now(tz=None)
            deltat = dt.timedelta(seconds=20)
            stop_deadline = n + deltat
            # Bizarre: If the number of microseconds is not a
            # multiple of 1000, Julia's DateTime can't handle it
            # because it only represents miliseconds.
            # So pyconvert fails, no explanation.
            stop_deadline = stop_deadline.replace(microsecond=0)
        p["stop_deadline"] = stop_deadline

        self.params = p

    def fit(self, X, y):
        X_f64 = np.ascontiguousarray(X, dtype=np.float64)
        y_f64 = np.ascontiguousarray(y, dtype=np.float64)
        self.raw_reg_str = jl.regression_main(X_f64, y_f64, self.params)
        self.is_fitted = True

    def predict(self, X):
        pass
