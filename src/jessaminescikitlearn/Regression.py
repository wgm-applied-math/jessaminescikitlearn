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

        # TODO I think sklearn expects all of those parameters to
        # correspond to fields on self.

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

        # TODO handle DataFrame columns
        X, y = check_X_y(
            X, y, dtype=np.float64,
            estimator="Jessamine")

        n_points, n_vars = X.shape
        self.raw_reg_str = jl.regression_main(X, y, self.params)
        #self.raw_reg_str = "((0.544091161224765 * x1) + ((-2.999999999999381 * (x1 * x2)) + ((0.8186362795911761 * (2.443087424614468 + (3 * x1))) + (2.9999999999994373 * x2))))"
        self.sym = sympy.parsing.sympy_parser.parse_expr(
            self.raw_reg_str)
        # The 1+ here is because symbols() creates x0, x1, x2...
        # but the Julia output involves x1, x2, ..., no x0.
        xv = sympy.symbols(f"x:{1+n_vars}")
        # So we let sympy create x0, but then we skip over it here.
        f = sympy.lambdify(xv[1:], self.sym)
        self.f = f
        self.n_vars = n_vars
        self.is_fitted = True

    def predict(self, X):
        check_is_fitted(self, "is_fitted")
        # TODO handle DataFrame columns
        X = check_array(
            X, dtype=np.float64,
            estimator="Jessamine")
        assert self.n_vars == X.shape[1]
        x_cols = np.unstack(X, axis=1)
        return self.f(*x_cols)

    # TODO Also check:
    # get_params
    # set_params

# From AR's AI generated code:

# TODO Bring over model() for SRBench

# TODO Maybe bring over complexity()
