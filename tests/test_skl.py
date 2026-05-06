# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

import pandas as pd
import numpy as np
from scipy import stats
import sympy

from sklearn.utils.estimator_checks import check_estimator

import jessaminescikitlearn
import jessaminescikitlearn.Regression as JR

def te_DISABLE_st_skl():
    r = JR.Regressor()
    # SKL This does a lot of validation including testing r on a
    # bunch of pre-defined data sets:
    check_estimator(r)
