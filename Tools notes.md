
# On Ubuntu, installing `conda`, etc

I'm using `github:cavalab/srbench` branch `srbench_2025`. 

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

_Note:_ This is not true, but I can't figure out what happened, as I'm quite sure I saw this.
Maybe I was looking at the wrong branch when I saw it?
Anyway:
In `docs/user_guide.md` it says to explicitly clone the `pmlb` repository from GitHub, which again, doesn't work because of LFS.

_Note:_ In `docs/user_guide.md` it says to run `python download_data.py`, but the `firstprinciples` data sets give errors, so apparently they no longer exist.
Which I don't understand, because they seem to be available at [the PMLB homepage](https://epistasislab.github.io/pmlb/).

_Note:_ The `scripts/install_algorithm.sh` script uses `micromamba`.
The `Dockerfile` lists it.
So I could install that:
```sh
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```
But the script requires many more corrections...



## Trying out `xgboost`

