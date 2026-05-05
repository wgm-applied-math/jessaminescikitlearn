# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# One Julia thread to avoid segfaults
import os
os.environ["JULIA_NUM_THREADS"] = "1"


import juliapkg
from juliacall import Main as jl

jl.seval("""
using JessamineSciKitLearn
""")
