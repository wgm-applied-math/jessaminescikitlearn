# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import numpy as np
from scipy import stats

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
    x1 = x1_dist.sample(shape=(n_points,1))
    x2 = x2_dist.sample(shape=(n_points,1))
    y = f(x1, x2)
    X = np.block([x1, x2])
    return (X, y)

def test_fit():
    X, y = make_data()
    r = JR.Regressor()
    print(r.fit(X, y))
