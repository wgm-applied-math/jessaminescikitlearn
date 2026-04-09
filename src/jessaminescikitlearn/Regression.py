# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import sklearn
import sympy

from juliacall import Main as jl, convert as jlconvert

from sklearn.base import BaseEstimator, RegressorMixin

class Regressor(RegressorMixin, BaseEstimator):

    def __init__(self, params):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
