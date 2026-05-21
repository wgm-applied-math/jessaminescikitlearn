# SPDX-FileCopyrightText: 2026-present W. Garrett Mitchener <garrett.mitchener@gmail.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# One Julia thread to avoid segfaults
import os

# HACK 2026-05-21: Julia uses signals, including SIGSEGV, as part
# of its memory management.  However, the Python interpreter
# intercepts signals so that it can respond to Ctrl-C, for
# example.  The result is that when Julia's memory managment
# creates intentional SIGSEGVs, it triggers a signal handler in
# the Python interpreter that halts the program.
# For more information:
# https://github.com/JuliaPy/PythonCall.jl/issues/219

# One workaround is to restrict Julia to a single thread.
# I'm unclear why this works.  Here's how to do it:

os.environ["JULIA_NUM_THREADS"] = "1"

# A second workaround is to tell Python to leave signal handling
# to Julia.  Then Python doesn't respond to SIGSEGV and the
# program works as usual.  However, then the Python interpreter
# can't respond to Ctrl-C via SIGINT, which messes up interactive use.
# Here's how to do it:

# os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import juliapkg
from juliacall import Main as jl

jl.seval(
    """
using JessamineSciKitLearn
"""
)
