
# On Ubuntu, installing `conda`, etc

I'm using `github:cavalab/srbench` branch `srbench_2025`. 

To get this to clone successfully, you have to disable git LFS smudging:

```sh
env GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>
```


## Install `conda`

To install Miniconda3 for controlled multi-user use:

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
mkdir /usr/local/conda
groupadd conda
chgrp -R conda /usr/local/conda
chmod -R 770 /usr/local/conda
chmod g+s /usr/local/conda
adduser garrett conda
bash Miniconda3-latest-Linux-x86_64.sh
```
and tell it to install into `/usr/local/conda/miniconda3`.

To use it from `bash`:

```sh
eval "$(/usr/local/conda/miniconda3/bin/conda shell.bash hook)"
```

And it seems to be a good idea to do

```sh
apt install pipx
pipx install hatch
pipx install mu-repo
```

## New environment

To make this work, you need to _not_ start with the base environment, because it has an old version of Python, and an old version of `pcre2`, which makes it impossible to install the most recent version of Julia.
So create a minimal `environment.yml`:

```yaml
name: JBench
channels:
  - conda-forge
dependencies:
  - julia=1.12.5
  - python=3.14
  - pip
```

and create the practice environment with:

```sh
conda config --add channels conda-forge
conda create --file environment.yml
conda activate JBench
conda install julia
```

Then things seem to mostly work 

## Hmm: Alternative, using `conda` to install `juliaup`

There is a `juliaup` package within `conda`.


# Setting up `srbench`

Clone the repository, switch to the `srbench_2025` branch.

In the root directory, there's a `conda` environment file, so:

```sh
conda create --file base_environment.yml --name srbench_2025
```

_Note:_ In `docs/user_guide.md` this file is referred to as `environment.yml` and the stated `conda` command doesn't include a name.

_Note:_ I had to change `base_environment.yml` to not install `pmlb` via GitHub, but just plain `pip install pmlb`.
The GitHub repository doesn't have enough credits to allow for `git` to do LFS smudging.

_Note:_ On the master branch, in `docs/user_guide.md`, it says to explicitly clone the `pmlb` repository from GitHub, which again, doesn't work because of LFS.
This instruction is not in the `srbench_2025` branch.

_Note:_ In `docs/user_guide.md` it says to run `python download_data.py`, but the `firstprinciples` data sets give errors, so apparently they no longer exist.
Which I don't understand, because they seem to be available at [the PMLB homepage](https://epistasislab.github.io/pmlb/).

_Note:_ The `scripts/install_algorithm.sh` script uses `micromamba`.
The `Dockerfile` lists it.
So I could install that:
```sh
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```
But the script requires many more corrections...



## Trying out `experiments/test_algorithm.py`

_Note:_ Line 85 has a comment suggesting that `est` should be an "estimator class" but based on `evaluate_model.py`, `evaluate_model`, it should be an _instance_ of the estimator class.

_Note:_ `evaluate_model.py` imports `eco2ai`, but I'm getting an error:
```
test_algorithm.py:12: in <module>
    from evaluate_model import evaluate_model
evaluate_model.py:19: in <module>
    import eco2ai
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/eco2ai/__init__.py:1: in <module>
    from .emission_track import (
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/eco2ai/emission_track.py:12: in <module>
    from eco2ai.tools.tools_cpu import CPU, all_available_cpu
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/eco2ai/tools/tools_cpu.py:11: in <module>
    from pkg_resources import resource_stream
E   ModuleNotFoundError: No module named 'pkg_resources'
```
I think `eco2ai` is not functioning in general, so I've disabled the `import eco2ai`

_Note:_ When running `python -m pytest -v test_algorithm.py --ml jessamine`, I get an error in `test_algorithm.py`, `test_evaluate` that has to do with not being able to load the test data file.
```
test_algorithm.py:80:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
evaluate_model.py:88: in evaluate_model
    features, labels, feature_names = read_file(
read_file.py:14: in read_file
    input_data = pd.read_csv(filename, sep=sep, compression=compression)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/pandas/io/parsers/readers.py:873: in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/pandas/io/parsers/readers.py:300: in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/pandas/io/parsers/readers.py:1645: in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/pandas/io/parsers/readers.py:1922: in _make_engine
    return mapping[engine](f, **self.options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../.conda/envs/srbench_2025/lib/python3.14/site-packages/pandas/io/parsers/c_parser_wrapper.py:95: in __init__
    self._reader = parsers.TextReader(src, **kwds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   TypeError: object of type 'NoneType' has no len()
```
The fix is to add `sep="\t"` to the call to `read_file` in `evaluate_model.py`, `evaluate_model()`, around line 90.
The `read_file` function uses `sep=None` as a default, and I believe there's a call to `len(sep)` somewhere in Pandas that results in the above exception.

TODO: In `read_file`, use file name to choose tab or comma as separator if `sep=None`.


_Note:_ Now I get this error:

```
[ Info: regression_main: Best (careful string): ((-3.5775578867488838 * (x2 * (x1 ** 3))) + ((1.0535771201296134 * (x2 * (x1 ** 5))) + (0.5959777726969981 * (x1 + (4 * (x2 * x1))))))
Regression.fit: sym: 1.0535771201296134*x1**5*x2 - 3.5775578867488838*x1**3*x2 + 2.383911090787992*x1*x2 + 0.5959777726969981*x1
Training time measure: 73.28297162055969
fitted est: 1.0535771201296134*lugs_1989**5*lugs_1990 - 3.5775578867488838*lugs_1989**3*lugs_1990 + 2.383911090787992*lugs_1989*lugs_1990 + 0.5959777726969981*lugs_1989
symbolic model: 1.0535771201296134*lugs_1989**5*lugs_1990 - 3.5775578867488838*lugs_1989**3*lugs_1990 + 2.383911090787992*lugs_1989*lugs_1990 + 0.5959777726969981*lugs_1989
Warning: parse expr failed. Msg: 'Add' object has no attribute 'strip'
Warning: simplify failed. Msg: Sympify of expression 'could not parse ''' failed, because of exception being raised:
SyntaxError: invalid syntax (<string>, line 0)
jessamine does not have a complexity() method, and does not generate sympy-compatible expressions. setting to -1
results:
FAILED
test_algorithm.py::test_sympy[jessamine] algorithm imported: <module 'methods.jessamine.regressor' from '/home/garrett/git/Research/srbench/experiment/methods/jessamine/regressor.py'>
FAILED
```

So something is wrong because I've checked and `sympy` can parse Jessamine's output and all of the following strings.

The warnings and exception are generated in `experiment/metrics/evaluation.py`, `get_symbolic_model`, around line 60.

Turns out, `srbench` expects the output of `model(est)` to be a sympy-compatible string, not a sympy symbolic object. so the above error happens because it's effectively trying to parse an expression object as a string.


_Note:_ Next error is that `test_algorithm.py`, `test_sympy()`, around line 120, there's an assertion that the algorithm module must implement `complexity`.
I thought this wasn't actually necessary, because srbench has its own complexity measure that works on sympy-compatible models.
So I commented it out.

_Note:_ With those changes, Jessamine now passes `test_algorithm`.


# Trouble with PMLB Python package

On PyPI, the package `pmlb` is at version 1.0.1.post3.
The wheel file includes `all_summary_stats.tsv` which does not include the `firstprinciples_*` datasets mentioned in the srbench scripts.
Specifically, I think this is why `datasets/download_data.py` fails.

_Note:_
This repository also doesn't have enough credits for LFS.
So:

```sh
dnf install python3-build
env GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/EpistasisLab/pmlb.git
cd pmlb
python -m build
```

Then in `dist/` I get a wheel and a source file, and they have a complete `all_summary_stats.tsv` file.


