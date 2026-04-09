# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import sklearn
import sympy

from juliacall import Main as jl
import juliapkg

juliapkg.resolve()

jl.seval("""
import JessamineSciKitLearn
""")
