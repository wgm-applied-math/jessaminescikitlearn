# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime as dt
import numpy as np
import sklearn
import sympy

from sklearn.utils.validation import check_is_fitted, validate_data
from typing import Any, Optional

from sklearn.base import BaseEstimator, RegressorMixin, _fit_context

from . import jl


class Regressor(RegressorMixin, BaseEstimator):

    # SKL Used by @_fit_context() for validation
    _parameter_constraints = {
        "stop_deadline" : [dt.datetime, None],
        "random_state" : ["random_state", None],
        "genome_spec" : [dict, None],
        "lambda_b" : [float],
        "lambda_p" : [float],
        "lambda_op" : [float],
        "num_islands" : [int],
        "stop_threshold" : [float, None],
        "op_inventory" : [str],
        "exploration" : [dict, None],
        "simplification" : [dict, None],
        }

    def __init__(
            self,
            stop_deadline : Optional[dt.datetime] = None,
            random_state : Optional[int] = None,
            genome_spec : Optional[dict] = None,
            lambda_b : float = 1e-10,
            lambda_p : float = 1e-10,
            lambda_op : float = 1e-10,
            num_islands : int = 1,
            stop_threshold : Optional[float] = None,
            op_inventory : str = "Polynomial",
            exploration : Optional[dict] = None,
            simplification : Optional[dict] = None):

        # SKL conventions:
        #
        # - All parameters are stored unmodified in
        # parallel attributes.
        #
        # - When instantiating with no given parameters, __init__
        # is not supposed to use a dict as a default value, I
        # assume for immutability reasons.  So genome_spec,
        # etc. have to default to None rather than {}.

        self.stop_deadline = stop_deadline
        self.random_state = random_state
        self.genome_spec = genome_spec
        self.lambda_p = lambda_p
        self.lambda_b = lambda_b
        self.lambda_op = lambda_op
        self.num_islands = num_islands
        self.stop_threshold = stop_threshold
        self.op_inventory = op_inventory
        self.exploration = exploration
        self.simplification = simplification

        # For future compatibility, include a version
        self._version = (1,0,0)


    def get_validated_params(self):
        # The BaseEstimator.get_params function uses python magic
        # to go through the definition of __init__ and fish out
        # keyword parameters, which are pulled from attributes of self
        # into a dict().

        self._validate_params()
        prespec = self.get_params()

        if prespec["stop_deadline"] is None:
            n = dt.datetime.now(tz=None)
            deltat = dt.timedelta(seconds=40)
            prespec["stop_deadline"] = n + deltat

        # Bizarre: If the number of microseconds is not a
        # multiple of 1000, Julia's DateTime can't handle it
        # because it only represents miliseconds.
        # So in Julia, PythonCall.pyconvert fails, no explanation.
        # Simplest solution is to zero out the microseconds field.
        prespec["stop_deadline"] = prespec["stop_deadline"].replace(microsecond=0)

        # Heuristic:
        # If there are n inputs
        # set up 1.5 n outputs
        # and 0.5 n scratch
        assert self.n_features_in_ > 0
        n = self.n_features_in_

        # SKL We can't modify any part of any given parameters,
        # so we need to copy the genome spec dictionary rather
        # than change it in place.
        g_spec_defaults = {
            "input_size": n,
            "output_size": n + (1 + n) // 2,
            "scratch_size": (1 + n) // 2}
        prespec.setdefault("genome_spec", {})
        prespec["genome_spec"] = g_spec_defaults | prespec["genome_spec"]

        prespec.setdefault("exploration", {})
        return prespec

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        print("in jessaminescikitlearn.Regression.Regressor.fit: X and y are")
        print(X)
        print(y)

        # SKL Sets self.n_features_in_ and self.feature_names_in_
        # if X is a table of some kind.
        # So this has to be done before...
        X, y = validate_data(self,
            X, y,
            reset=True,
            dtype=np.float64,
            y_numeric=True)

        assert X.shape[1] == self.n_features_in_

        # ... get_validated_params(), which
        # uses self.n_features_in_.
        prespec = self.get_validated_params()

        # validate_data is supposed to set feature_names_in_.
        # If that happened, X is a DataFrame or similar, and the
        # columns should have names appropriate for a symbol.
        if (hasattr(self, "feature_names_in_")
            and self.feature_names_in_ is not None):
            self.feature_names_in_sym_ = sympy.symbols(
                list(self.feature_names_in_),
                real=True)
        else:
            self.feature_names_in_sym_ = None

        # The 1+ here is because symbols() uses python's range convention,
        # so 1:5 means 1 <= j < 5.
        n_vars = self.n_features_in_
        assert n_vars is not None
        xv = sympy.symbols(f"x1:{1+n_vars}", real=True)
        vd = { str(x): x for x in xv }
        epsilon = sympy.symbols("ϵ")
        vd["epsilon"] = epsilon

        # Turn the crank
        self.raw_reg_str_ = jl.regression_main(X, y, prespec)

        # For use during testing:
        #self.raw_reg_str_ = "((0.544091161224765 * x1) + ((-2.999999999999381 * (x1 * x2)) + ((0.8186362795911761 * (2.443087424614468 + (3 * x1))) + (2.9999999999994373 * x2))))"

        self.sym_init_ = sympy.parsing.sympy_parser.parse_expr(
            self.raw_reg_str_,
            vd)

        # Handle Jessamine's use of extended real numbers
        # where 1/0 = Inf:
        # In the Julia result string, 1/0 is changed into 1/epsilon,
        # and the parser converts `epsilon` into `ϵ`.
        # Then we do this limit:
        self.sym_ = sympy.limit(self.sym_init_, epsilon, 0, dir="+-")
        self.xv_ = xv

        print(f"Regression.fit: sym: {self.sym_}")
        if self.feature_names_in_sym_ is None:
            # Vanilla feature names, no need to substitute
            self.feature_names_in_sym_ = xv
            self.model_sym_ = self.sym_
        else:
            # Columns have symbolic names, need to substitute
            translation = [ (xv[j], self.feature_names_in_sym_[j])
                            for j in range(n_vars) ]
            self.model_sym_ = self.sym_.subs(translation)

        # SKL: fit() must return self
        return self

    def get_f(self):
        if not hasattr(self, "f_"):
            self.f_ = sympy.lambdify(self.xv_, self.sym_)
        return self.f_


    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self,
            X, "no_validation",
            reset=False,
            dtype=np.float64)
        x_cols = np.unstack(X, axis=1)

        return self.get_f()(*x_cols)

    def model(self):
        check_is_fitted(self)
        return self.model_sym_

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove un-pickle-able attributes
        state.pop("f_", None)
        return state

    def __setstate__(self, state):
        # Restore pickle-able attributes
        self.__dict__.update(state)


# From AR's AI generated code:

# TODO Maybe bring over complexity()
