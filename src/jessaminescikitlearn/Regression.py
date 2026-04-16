# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime as dt
import numpy as np
import sklearn
import sympy


from typing import Any, Optional

from sklearn.base import BaseEstimator, RegressorMixin

from . import jl


class Regressor(RegressorMixin, BaseEstimator):

    def __init__(
            self,
            stop_deadline : Optional[dt.datetime] = None,
            rng_seed : Optional[int] = None,
            genome_spec : Optional[dict] = None,
            lambda_b : float = 1e-10,
            lambda_p : float = 1e-10,
            lambda_op : float = 1e-10,
            num_islands : int = 1,
            stop_threshold : Optional[float] = None,
            exploration_spec : Optional[dict] = None,
            simplification_spec : Optional[dict] = None):
        p : dict[str,Any] = dict()
        if stop_deadline is None:
            n = dt.datetime.now(tz=None)
            deltat = dt.timedelta(seconds=45)
            stop_deadline = n + deltat
        # Bizarre: If the number of microseconds is not a
        # multiple of 1000, Julia's DateTime can't handle it
        # because it only represents miliseconds.
        # So pyconvert fails, no explanation.
        # Simplest solution is to zero out the microseconds field.
        stop_deadline = stop_deadline.replace(microsecond=0)
        p["stop_deadline"] = stop_deadline
        if rng_seed:
            p["rng_seed"] = rng_seed
        if genome_spec:
            p["genome_spec"] = genome_spec
        if stop_threshold:
            p["stop_threshold"] = stop_threshold
        if exploration_spec:
            p["exploration_spec"] = exploration_spec
        if simplification_spec:
            p["simplification_spec"] = simplification_spec
        self.params = p

    def fit(self, X, y):
        X_f64 = np.ascontiguousarray(X, dtype=np.float64)
        y_f64 = np.ascontiguousarray(y, dtype=np.float64)
        self.raw_reg_str = jl.regression_main(X_f64, y_f64, self.params)
        self.sym = sympy.parsing.sympy_parser.parse_expr(
            self.raw_reg_str)
        self.is_fitted = True

    def predict(self, X):
        pass
