# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from contextlib import contextmanager
import datetime as dt
from numbers import Number
import numpy as np
import signal
import sympy
from typing import Optional

from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context

from . import jl


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds=60):
    def _handler(signum, frame):
        raise TimeoutError(f"Calculation exceeded {seconds}s time limit")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)



class Regressor(RegressorMixin, BaseEstimator):

    # SKL Used by @_fit_context() for validation
    _parameter_constraints = {
        "random_state": ["random_state", None],
        "verbosity": [int, None],
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
        "max_time": [Number, None],
        "stop_deadline": [dt.datetime, None],
        "num_islands": [int, None],
        "stop_threshold": [float, None],
        "simplify": [bool],
    }

    def __init__(
        self,
        random_state: Optional[int] = None,
        verbosity: Optional[int] = None,
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
        max_time: Optional[Number] = None,
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

        # General
        self.random_state = random_state
        self.verbosity = verbosity
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
            max_time_seconds = prespec.get("max_time", 30.0)
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
        g_spec_auto = {
            "input_size": n,
            "output_size": n + (1 + n) // 2,
            "scratch_size": (1 + n) // 2,
            "parameter_size": (1 + n) // 2,
        }
        for k in ["input_size", "output_size", "scratch_size", "parameter_size"]:
            if prespec.get(k, None) is None:
                prespec[k] = g_spec_auto[k]

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
        epsilon = sympy.symbols("ϵ", real=True)
        Inf = sympy.symbols("Inf", real=True)
        vd["epsilon"] = epsilon
        vd["Inf"] = Inf

        # Turn the crank
        result = jl.regression_main(X, y, prespec)

        # Go through and try to find one that can be properly
        # processed.
        raw_reg_str = None
        expr = None
        # Look for the best usable agent.  The wrinkle is that
        # Julia handles division by zero differently from sympy
        # and Python, so some agents that work well enough within
        # Jessamine yield expressions that sympy can't handle.
        # So we go through the list of discoveries until we find
        # one that works.
        for r in result.discoveries:
            raw_reg_str = r.y_num_str
            try:
                with time_limit():
                    expr = sympy.parsing.sympy_parser.parse_expr(raw_reg_str, vd)
                    # These show up in certain cases of division by zero.
                    # In Julia, 1.0 / 0.0 is Inf.
                    if epsilon in expr.free_symbols:
                        expr_simp = sympy.simplify(expr)
                        expr = sympy.limit(expr_simp, epsilon, 0, dir="+")
                    # These also show up sometimes
                    if Inf in expr.free_symbols:
                        expr_simp = sympy.simplify(expr)
                        expr = sympy.limit(expr_simp, Inf, sympy.oo)
                    expr = sympy.simplify(expr)
                    expr = expr.evalf()
                    # If all of that works, we've found a good one, exit the loop
                    break
            except Exception as e:
                raise e

        self.sym_ = expr
        self.raw_reg_str_ = raw_reg_str
        self.xv_ = xv
        # SKL See comment in set_f().
        self.set_f()

        # print(f"Regression.fit: sym: {self.sym_}")
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
        if hasattr(self, "xv_") and hasattr(self, "sym_") and not hasattr(self, "f_"):
            self.f_ = sympy.lambdify(self.xv_, self.sym_)
            return self.f_
        else:
            return None

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, "no_validation", reset=False, dtype=np.float64)
        x_cols = np.unstack(X, axis=1)

        return self.f_(*x_cols)

    def model_sympy(self):
        check_is_fitted(self)
        return self.model_sym_

    def model_str(self):
        check_is_fitted(self)
        return str(self.model_sympy())

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
