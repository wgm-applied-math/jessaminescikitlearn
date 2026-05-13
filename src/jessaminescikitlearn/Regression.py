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
        "random_state": ["random_state", None],
        # Genome spec
        "output_size": [int, None],
        "scratch_size": [int, None],
        "parameter_size": [int, None],
        "num_time_steps": [int, None],
        # Mutation spec
        "op_inventory": [str],
        "p_mutate_op": [float, None],
        "p_mutate_index": [float, None],
        "p_duplicate_index": [float, None],
        "p_delete_index": [float, None],
        "p_duplicate_instruction": [float, None],
        "p_delete_instruction": [float, None],
        "p_hop_instruction": [float, None],
        # Selection spec
        "num_to_keep": [int, None],
        "num_to_generate": [int, None],
        "p_take_better": [float, None],
        "p_take_very_best": [float, None],
        # Regularization
        "lambda_b": [float, None],
        "lambda_p": [float, None],
        "lambda_op": [float, None],
        # Search
        "max_time": [float, None],
        "stop_deadline": [dt.datetime, None],
        "num_islands": [int, None],
        "stop_threshold": [float, None],
        "simplify": [bool],
    }

    def __init__(
        self,
        random_state: Optional[int] = None,
        # Genome spec
        output_size: Optional[int] = None,
        scratch_size: Optional[int] = None,
        parameter_size: Optional[int] = None,
        num_time_steps: Optional[int] = None,
        # Mutation spec
        op_inventory: str = "Polynomial",
        p_mutate_op: Optional[float] = None,
        p_mutate_index: Optional[float] = None,
        p_duplicate_index: Optional[float] = None,
        p_delete_index: Optional[float] = None,
        p_duplicate_instruction: Optional[float] = None,
        p_delete_instruction: Optional[float] = None,
        p_hop_instruction: Optional[float] = None,
        # Selection spec
        num_to_keep: Optional[int] = None,
        num_to_generate: Optional[int] = None,
        p_take_better: Optional[float] = None,
        p_take_very_best: Optional[float] = None,
        # Regularization
        lambda_b: Optional[float] = None,
        lambda_p: Optional[float] = None,
        lambda_op: Optional[float] = None,
        # Search
        max_time: Optional[float] = None,
        stop_deadline: Optional[dt.datetime] = None,
        num_islands: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        simplify: bool = True,
    ):

        # SKL conventions:
        #
        # - All parameters are stored unmodified in
        # parallel attributes.
        #
        # - When instantiating with no given parameters, __init__
        # is not supposed to use a dict as a default value, I
        # assume for immutability reasons.  So genome_spec,
        # etc. have to default to None rather than {}.

        self.random_state = random_state
        # Genome spec
        self.output_size = output_size
        self.scratch_size = scratch_size
        self.parameter_size = parameter_size
        self.num_time_steps = num_time_steps
        # Mutation spec
        self.op_inventory = op_inventory
        self.p_mutate_op = p_mutate_op
        self.p_mutate_index = p_mutate_index
        self.p_duplicate_index = p_duplicate_index
        self.p_delete_index = p_delete_index
        self.p_duplicate_instruction = p_duplicate_instruction
        self.p_delete_instruction = p_delete_instruction
        self.p_hop_instruction = p_hop_instruction
        # Selection spec
        self.num_to_keep = num_to_keep
        self.num_to_generate = num_to_generate
        self.p_take_better = p_take_better
        self.p_take_very_best = p_take_very_best
        # Regularization
        self.lambda_p = lambda_p
        self.lambda_b = lambda_b
        self.lambda_op = lambda_op
        # Search
        self.max_time = max_time
        self.stop_deadline = stop_deadline
        self.num_islands = num_islands
        self.stop_threshold = stop_threshold
        self.simplify = simplify

        # For future compatibility, include a version
        self._version = (1, 0, 0)

    def get_validated_params(self):
        # The BaseEstimator.get_params function uses python magic
        # to go through the definition of __init__ and fish out
        # keyword parameters, which are pulled from attributes of self
        # into a dict().

        self._validate_params()
        prespec0 = self.get_params()

        # Just leave out anything None.
        prespec = {}
        for k, v in prespec0.items():
            if v is not None:
                prespec[k] = v

        # There has to be a stop deadline
        if not "stop_deadline" in prespec or prespec["stop_deadline"] is None:
            # Maximum time in seconds:
            max_time_seconds = prespec.get("max_time", 30)
            prespec["stop_deadline"] = dt.datetime.now(tz=None) + dt.timedelta(
                seconds=max_time_seconds
            )

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

        # Genome
        # Defaults
        g_spec = {
            "input_size": n,
            "output_size": n + (1 + n) // 2,
            "scratch_size": (1 + n) // 2,
            "parameter_size": (1 + n) // 2,
        }
        for k in ["input_size", "output_size", "scratch_size", "parameter_size"]:
            if k in prespec and prespec[k] is not None:
                g_spec[k] = prespec[k]
        prespec["genome"] = g_spec

        # Mutation
        m_spec = {}
        for k in [
            "p_mutate_op",
            "p_mutate_index",
            "p_duplicate_index",
            "p_delete_index",
            "p_duplicate_instruction",
            "p_delete_instruction",
            "p_hop_instruction",
        ]:
            if k in prespec and prespec[k] is not None:
                m_spec[k] = prespec[k]

        # Selection
        s_spec = {}
        for k in [
            "num_to_keep",
            "num_to_generate",
            "p_take_better",
            "p_take_very_best",
        ]:
            if k in prespec and prespec[k] is not None:
                s_spec[k] = prespec[k]

        # Exploration
        prespec["exploration"] = m_spec | s_spec

        return prespec

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # print("in jessaminescikitlearn.Regression.Regressor.fit: X and y are")
        # print(X)
        # print(y)

        # SKL Sets self.n_features_in_ and self.feature_names_in_
        # if X is a table of some kind.
        # So this has to be done before...
        X, y = validate_data(self, X, y, reset=True, dtype=np.float64, y_numeric=True)

        assert X.shape[1] == self.n_features_in_

        # ... get_validated_params(), which
        # uses self.n_features_in_.
        prespec = self.get_validated_params()

        # validate_data is supposed to set feature_names_in_.
        # If that happened, X is a DataFrame or similar, and the
        # columns should have names appropriate for a symbol.
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            self.feature_names_in_sym_ = sympy.symbols(
                list(self.feature_names_in_), real=True
            )
        else:
            self.feature_names_in_sym_ = None

        # The 1+ here is because symbols() uses python's range convention,
        # so 1:5 means 1 <= j < 5.
        n_vars = self.n_features_in_
        assert n_vars is not None
        xv = sympy.symbols(f"x1:{1+n_vars}", real=True)
        vd = {str(x): x for x in xv}
        epsilon = sympy.symbols("ϵ")
        vd["epsilon"] = epsilon

        # Turn the crank
        self.raw_reg_str_ = jl.regression_main(X, y, prespec)

        # For use during testing:
        # self.raw_reg_str_ = "((0.544091161224765 * x1) + ((-2.999999999999381 * (x1 * x2)) + ((0.8186362795911761 * (2.443087424614468 + (3 * x1))) + (2.9999999999994373 * x2))))"

        self.sym_init_ = sympy.parsing.sympy_parser.parse_expr(self.raw_reg_str_, vd)

        # Handle Jessamine's use of extended real numbers
        # where 1/0 = Inf:
        # In the Julia result string, 1/0 is changed into 1/epsilon,
        # and the parser converts `epsilon` into `ϵ`.
        # Then we do this limit:
        self.sym_ = sympy.limit(self.sym_init_, epsilon, 0, dir="+-")
        self.xv_ = xv
        # SKL See comment in set_f().
        self.set_f()

        print(f"Regression.fit: sym: {self.sym_}")
        if self.feature_names_in_sym_ is None:
            # Vanilla feature names, no need to substitute
            self.feature_names_in_sym_ = xv
            self.model_sym_ = self.sym_
        else:
            # Columns have symbolic names, need to substitute
            translation = [
                (xv[j], self.feature_names_in_sym_[j]) for j in range(n_vars)
            ]
            self.model_sym_ = self.sym_.subs(translation)

        # SKL: fit() must return self
        return self

    def set_f(self):
        # SKL predict() is not allowed to modify self's
        # attributes.  So we have to cache the lambdified f in
        # fit() and restore it during unpickling.
        # Hence this method.
        if not hasattr(self, "f_"):
            self.f_ = sympy.lambdify(self.xv_, self.sym_)
        return self.f_

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, "no_validation", reset=False, dtype=np.float64)
        x_cols = np.unstack(X, axis=1)

        return self.f_(*x_cols)

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
        # SKL See comment in set_f()
        self.set_f()

    def __str__(self):
        # srbench seems to expect the result of str() to be a
        # Sympy-compatible string
        if hasattr(self, "model_sym_"):
            return str(self.model_sym_)
        else:
            return repr(self)
