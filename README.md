# JessamineSciKitLearn

[![PyPI - Version](https://img.shields.io/pypi/v/jessaminescikitlearn.svg)](https://pypi.org/project/jessaminescikitlearn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jessaminescikitlearn.svg)](https://pypi.org/project/jessaminescikitlearn)

Jessamine is a symbolic regression method with an evolutionary algorithm.
The genome architecture is based on an abstract reaction network rather than an expression tree.
Each genome consists of lists of instructions in an iterated static-single-assignment form.
    
The `jessaminescikitlearn` package provides a Python interface to a [scikit-learn](http://scikit-learn.org)-compatible regressor that is essentially a generalized additive model, and produces [SymPy](http://www.sympy.org)-compatible results.
Given a table or DataFrame with sample points (rows) and baseline features (columns), a genome computes a new basis of features, which are fed to ridge regression.

The Jessamine engine has other capabilities that are not yet available through this Python interface.


-----

## Table of Contents

- [Installation](#installation)
- [Related](#related)
- [License](#license)

## Installation

_Note:_ The current release-in-progress is v0.3.0.
It isn't finalized.
I'll be rebuilding it frequently as I learn GitHub's functionality and make fixes.
Expect it to change daily.

Install a recent version of [Python](http://python.org) and [Julia](http://julialang.org).

Eventually you will be able to do this:
```console
pip install jessaminescikitlearn
```
but currently, this package is not registered with PyPI.

Instead, install directly from GitHub.
To install v0.3.0, for example, run
```console
pip install 'jessaminescikitlearn @ git+https://github.com/wgm-applied-math/jessaminescikitlearn.git@v0.3.0'
```
You may also need to install [`hatch`](http://hatch.pypa.io/latest/).

You may also download wheels directly from this GitHub site under Releases and use pip to install from those files.

Note that when you run Python and run `import jessaminescikitlearn.Regression`, it will resolve the underlying Julia environment, which usually involves downloading and precompiling several updated Julia packages.
Expect this to take some time, especially the first time you do so.

## Related

[Jessamine.jl](https://github.com/wgm-applied-math/Jessamine.jl):
The core of the evolutionary algorithm.

[JessamineSymbolics.jl](https://github.com/wgm-applied-math/JessamineSymbolics.jl):
Companion package for using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) with Jessamine genomes.

[JessamineSciKitLearn.jl](https://github.com/wgm-applied-math/JessamineSciKitLearn.jl):
Julia half of the scikit-learn interface to Jessamine.
It uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl).


## License

`jessaminescikitlearn` is distributed under the terms of the [GPL](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
