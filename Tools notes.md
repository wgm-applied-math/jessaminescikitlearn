
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

## New environment

```sh
conda config --add channels conda-forge
conda create --file environment.yml --name Jessamine-2026-05
conda activate Jessamine-2026-05
conda install julia
```

## Hmm: Alternative, using `conda` to install `juliaup`

There is a `juliaup` package within `conda`.




