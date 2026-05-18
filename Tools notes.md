
# On Ubuntu, installing `conda`, etc

I'm using `github:cavalab/srbench` branch `srbench_2025`. 

To get this to clone successfully, you have to disable git LFS smudging.
To do so temporarily:

```sh
env GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>
```

To do so permanently:

```sh
https://github.com/EpistasisLab/pmlb.git
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

But miniconda is not really working here...?


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
The wheel file includes `all_summary_stats.tsv` which does not include the `firstprinciples_*` datasets mentioned in the `srbench` scripts.
Specifically, I think this is why `datasets/download_data.py` fails.

The current situation in `srbench` is that it installs `pmlb` via `conda`, and the
[package in conda-forge](https://github.com/conda-forge/pmlb-feedstock) is also at version 1.0.1.

_Note:_
The `pmlb` repository also doesn't have enough credits for LFS.
So, here's how to manually make a local wheel for version 1.0.2a:

```sh
dnf install python3-build
env GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/EpistasisLab/pmlb.git
cd pmlb
python -m build
```

Then in `dist/` I get a wheel and a source archive, and they have a complete `all_summary_stats.tsv` file, but none of the actual data files.



_Note:_
Had to install via `apt` on Ubuntu to build the local wheel:
- `python3-build`


_Note:_
I ran
``sh
conda uninstall pmlb
```
Which uninstalled some other stuff, including `requests`.

Continuing to try to get `download_data.py` to work, I have to install these within the `conda` environment using `pip install --force-reinstall`:
- `requests`

Without the `--force-reinstall`, I end up with this bizarre situation where the `requests` package seems to be installed, `pip list` shows it, but when I run python, trying `import requests` says it can't find the module.

And now finally `python download_data.py` runs to completion.

## Trying `install_algorithm.sh`

I've manually installed `micromamba` in the Ubuntu machine.

From `srbench` base directory,

```sh
bash scripts/install_algorithm.sh jessamine
```

This mostly works, but it doesn't know where to put the `environment.lock` file.
It specifies a non-existent directory in the output redirection.
Nowhere in the script does it reference the `algorithms` directory, so I think the documentation in `user_guide.md` is mistaken.

Trying this:
From `srbench` base directory,

```sh
bash scripts/install_algorithm.sh algorithms/jessamine
```

This takes a _long_ time, about 15 minutes.
And if anything goes wrong, you have to start all over again.

## Trying `experiment/analyze.py`

_Note:_ In branch `srbench_2025`, `user_guide.md`, it says to run `python experiment/analyze.py ...`
I wrote a script with `--no_docker` and a few other options.
But that results in errors from `sh` about how `optimize_model.py` is not found.

I don't know why they're trying to run it with `sh`.

In branch `docker-compose`, `user_guide.md`, it says to go into the `experiment` directory and run `python analyze.py ...`

I was able to make that start at least.

_Note:_
I installed `docker.io` and a few companion packages.
I tried running `experiment/analyze.py` with `--local` instead of `--no_docker` and it gives an error for each dataset like this:
```
docker compose run --rm -v /home/garrett/git/Research/srbench/experiment/experiment:/srbench -v /home/garrett/git/Research/srbench/experiment/../datasets/blackbox/:/../datasets/blackbox/ -v /home/garrett/git/Research/srbench/experiment/../results_blackbox_tuning/:/../results_blackbox_tuning/ -v /:/srbench_pretrained jessamine python -u /srbench/optimize_model.py /../datasets/blackbox/1193_BNG_lowbwt/1193_BNG_lowbwt.tsv.gz -ml jessamine -results_path /../results_blackbox_tuning/1193_BNG_lowbwt -seed 15795 -target_noise 0.0 -feature_noise 0.0 -fit_time_limit 60 -max_samples 4 --scale_x --scale_y
no configuration file provided: not found
```
which I think is a `docker` error.


# When running a clean re-do

- Try restoring `base_environment.yml` where I'd commented out the git URL for PMLB.
  Now that I've figured out how to disable git LFS smudging globally, maybe that installation will work properly.


