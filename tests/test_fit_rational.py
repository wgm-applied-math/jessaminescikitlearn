# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

import pandas as pd
import numpy as np
from scipy import stats
import sympy

print(dir())
print(__package__)

import jessaminescikitlearn
import jessaminescikitlearn.Regression as JR

def f(x1, x2):
    return (x1 + 2)/(x1 - x2)

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
    y = f(x1, x2)
    # This is how to stack x1 and x2 as columns
    # into a matrix in column-tabular form
    X = np.block(
        [x1.reshape(n_points,1),
         x2.reshape(n_points,1)])
    return (X, y)


def make_data_as_dataframe():
    X, y = make_data()
    y_s = pd.Series(y, name="y")
    u = pd.Series(X[:,0], name="u")
    v = pd.Series(X[:,1], name="v")
    X_df = pd.DataFrame({ "u": u, "v": v })
    return (X_df, y_s)


def test_fit_predict():
    X, y = make_data()
    fit_and_predict(X, y)

# def test_fit_predict_dataframe():
#     X, y = make_data_as_dataframe()
#     fit_and_predict(X, y)

def fit_and_predict(X, y):
    # disable for a moment
    # return
    r = JR.Regressor(
        op_inventory="Polynomial; RationalFunction",
    )
    r.fit(X, y)
    print(r.raw_reg_str)
    print(r.sym)
    print(r.model())
    yHat = r.predict(X)
    discrepancy = sum((yHat - y)**2)
    print(f"test_fit_predict: discrepancy = {discrepancy}")
    assert discrepancy < 1e-10
