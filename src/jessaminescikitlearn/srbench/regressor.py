# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module jessaminescikitlearn.srbench.regressor implements the interface
necessary for incorporation into the srbench benchmark suite.
"""

from .. import Regression

# hyper_params = []

"sklearn-compatible `Regressor` object"
est = Regression.Regressor()

def model(est, X=None):
    "extract a Sympy-compatible string representation of the fitted model"
    # The X here is so that the native symbolic representation of
    # columns produced by an estimator can be translated into
    # column names in a DataFrame X.
    return est.model()

# This is not needed since model() can produce a Sympy-compatible string
# "function to count the number of nodes"
# def complexity(est):
#     pass

# "dictionary of model-specific arguments to srbench's `evaluate_model.py`
# eval_kwargs = {}
