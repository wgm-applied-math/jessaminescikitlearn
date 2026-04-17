# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
from scipy import stats
import sympy

print(dir())
print(__package__)

import jessaminescikitlearn
import jessaminescikitlearn.Regression as JR

def f(x1, x2):
    return 2 + 3 * x1 + 3* x2 - 3 * x1 * x2

def make_data():
    x1_dist = stats.Normal(mu=0.0, sigma=1.0)
    x2_dist = stats.Normal(mu=0.0, sigma=2.0)
    np.random.seed(95548004)
    n_points = 30
    # Just so this is clear,
    # for the types to check out properly in Julia
    # while conforming to the conventions of scikit-learn:
    # X needs to be a matrix in column-tabular form
    # y needs to be a vector, not a single-column matrix
    # which means x1 and x2 have to start as vectors
    x1 = x1_dist.sample(shape=(n_points,))
    x2 = x2_dist.sample(shape=(n_points,))
    y = f(x1, x2)
    # This is how to stack x1 and x2 as columns
    # into a matrix in column-tabular form
    X = np.block(
        [x1.reshape(n_points,1),
         x2.reshape(n_points,1)])
    return (X, y)

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

def test_fit():
    # disable for a moment
    return
    X, y = make_data()
    r = JR.Regressor()
    r.fit(X, y)
    print(r.raw_reg_str)
    print(r.sym)
