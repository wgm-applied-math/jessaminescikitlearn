# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

#import pytest

import pandas as pd
import pickle
import numpy as np
from scipy import stats
import sympy

import jessaminescikitlearn
import jessaminescikitlearn.Regression as JR

def f(x1, x2):
    return 2 + 3 * x1 + 3* x2 - 3 * x1 * x2

def make_data():
    x1_dist = stats.Normal(mu=0.0, sigma=1.0)
    x2_dist = stats.Normal(mu=0.0, sigma=2.0)
    rng = np.random.default_rng(95548004)
    n_points = 30
    # Just so this is clear,
    # for the types to check out properly in Julia
    # while conforming to the conventions of scikit-learn:
    # X needs to be a matrix in column-tabular form
    # y needs to be a vector, not a single-column matrix
    # which means x1 and x2 have to start as vectors
    x1 = x1_dist.sample(shape=(n_points,), rng=rng)
    x2 = x2_dist.sample(shape=(n_points,), rng=rng)
    # Extra column, which Jessamine should figure out it should ignore
    x3 = x2_dist.sample(shape=(n_points,), rng=rng)
    y = f(x1, x2)
    # This is how to stack x1 and x2 as columns
    # into a matrix in column-tabular form
    X = np.block(
        [x1.reshape(n_points,1),
         x2.reshape(n_points,1),
         x3.reshape(n_points,1)])
    assert X.shape[0] == n_points
    assert X.shape[1] == 3
    # print("in test_fit.make_data")
    # print(X)
    # print(y)
    return (X, y)


def make_data_as_dataframe():
    X, y = make_data()
    y_s = pd.Series(y, name="y")
    u = pd.Series(X[:,0], name="u")
    v = pd.Series(X[:,1], name="v")
    w = pd.Series(X[:,2], name="w")
    X_df = pd.DataFrame({ "u": u, "v": v, "w": w })
    return (X_df, y_s)


def test_apply():
    raw_str = "((0.544091161224765 * x1) + ((-2.999999999999381 * (x1 * x2)) + ((0.8186362795911761 * (2.443087424614468 + (3 * x1))) + (2.9999999999994373 * x2))))"
    sym = sympy.parsing.sympy_parser.parse_expr(raw_str)
    x0, x1, x2 = sympy.symbols("x:3")
    f = sympy.lambdify([x1, x2], sym)
    X, y = make_data()
    yHat = f(X[:,0], X[:,1])
    discrepancy = sum((yHat - y)**2)
    print(f"test_apply: discrepancy = {discrepancy}")
    assert discrepancy < 1e-10

def test_fit_predict_pickle():
    X, y = make_data()
    r = fit_and_predict(X, y)

    # Test pickle
    r_pickled = pickle.dumps(r)
    r_unpickled = pickle.loads(r_pickled)
    yHat = r.predict(X)
    discrepancy = sum((yHat - y)**2)
    assert discrepancy < 1e-10

    return r

def test_fit_predict_dataframe():
    X, y = make_data_as_dataframe()
    return fit_and_predict(X, y)

def fit_and_predict(X, y):
    # disable for a moment
    # return
    r = JR.Regressor()
    r.fit(X, y)
    print(r.raw_reg_str_)
    print(r.sym_)
    print(r.model())
    yHat = r.predict(X)
    discrepancy = sum((yHat - y)**2)
    print(f"fit_and_predict: discrepancy = {discrepancy}")
    assert discrepancy < 1e-10
    return r
