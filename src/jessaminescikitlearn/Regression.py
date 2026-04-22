# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime as dt
import numpy as np
import sklearn
import sympy

from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
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

        # The BaseEstimator.get_params function uses python magic
        # to go through the definition of __init__ and fish out
        # keyword parameters, which are pulled from attributes of self
        # into a dict().
        if stop_deadline is None:
            n = dt.datetime.now(tz=None)
            deltat = dt.timedelta(seconds=45)
            stop_deadline = n + deltat
        assert isinstance(stop_deadline, dt.datetime)
        # Bizarre: If the number of microseconds is not a
        # multiple of 1000, Julia's DateTime can't handle it
        # because it only represents miliseconds.
        # So pyconvert fails, no explanation.
        # Simplest solution is to zero out the microseconds field.
        stop_deadline = stop_deadline.replace(microsecond=0)
        self.stop_deadline = stop_deadline
        self.rng_seed = rng_seed
        self.genome_spec = genome_spec
        self.lambda_p = float(lambda_p)
        self.lambda_b = float(lambda_b)
        self.lambda_op = float(lambda_op)
        self.num_islands = int(num_islands)
        self.stop_threshold = float(stop_threshold) if stop_threshold is not None else None
        self.exploration_spec = exploration_spec
        self.simplification_spec = simplification_spec

    def fit(self, X, y):
        # Capture feature names before check_X_y converts DataFrame to ndarray
        self.feature_names = None
        if hasattr(X, "columns"):
            self.feature_names = sympy.symbols(list(X.columns), real=True)
        X, y = check_X_y(
            X, y, dtype=np.float64,
            estimator="Jessamine")
        n_points, n_vars = X.shape
        self.n_vars = n_vars

        # The 1+ here is because symbols() creates x0, x1, x2...
        # but the Julia output involves x1, x2, ..., no x0.
        # So we let sympy create x0, but then we skip over it here.
        xv = sympy.symbols(f"x:{1+n_vars}", real=True)[1:]
        xvd = { str(x): x for x in xv }
        self.raw_reg_str = jl.regression_main(X, y, self.get_params())
        # For use during testing:
        #self.raw_reg_str = "((0.544091161224765 * x1) + ((-2.999999999999381 * (x1 * x2)) + ((0.8186362795911761 * (2.443087424614468 + (3 * x1))) + (2.9999999999994373 * x2))))"

        self.sym = sympy.parsing.sympy_parser.parse_expr(
            self.raw_reg_str,
            xvd)

        f = sympy.lambdify(xv, self.sym)
        self.f = f
        if self.feature_names is None:
            self.feature_names = xv
            self.model_sym = self.sym
        else:
            translation = [ (xv[j], self.feature_names[j])
                            for j in range(n_vars) ]
            self.model_sym = self.sym.subs(translation)
        self.is_fitted = True

    def predict(self, X):
        check_is_fitted(self, "is_fitted")
        X = check_array(
            X, dtype=np.float64,
            estimator="Jessamine")
        assert self.n_vars == X.shape[1]
        x_cols = np.unstack(X, axis=1)
        return self.f(*x_cols)

    def model(self):
        return self.model_sym

# From AR's AI generated code:

# TODO Maybe bring over complexity()
