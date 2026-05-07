
# On Ubuntu, installing `conda`, etc

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

and create the environment with:

```sh
conda config --add channels conda-forge
conda create --file environment.yml
conda activate JBench
conda install julia
```

Then things seem to mostly work 

## Hmm: Alternative, using `conda` to install `juliaup`

There is a `juliaup` package within `conda`.




