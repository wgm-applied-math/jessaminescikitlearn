# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# import pytest

import pickle

print(dir())
print(__package__)

import jessaminescikitlearn
import jessaminescikitlearn.Regression

def test_init():
    assert True

def test_default_pickle():
    r = jessaminescikitlearn.Regression.Regressor()
    r_pickled = pickle.dumps(r)
    r_unpickled = pickle.loads(r_pickled)
