# Bicycle node network loop analysis¶

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

This is the source code for the scientific project _Bicycle node network loop analysis_. The code assesses the quality of a proposed bicycle node network in Denmark via loop census analysis.

## Installation

First clone the repository:

```
git clone https://github.com/mszell/bikenwloops
```

Go to the cloned folder and create a new virtual environment via `conda` or `mamba` using the `environment.yml` file:

```
mamba env create -f environment.yml
```

Then, install the virtual environment's kernel in Jupyter:

```
mamba activate bikenwloops
ipython kernel install --user --name=bikenwloops
mamba deactivate
```

You can now run `jupyter lab` with kernel `bikenwloops` (Kernel > Change Kernel > bikenwloops).

## Repository structure

```
├── code                    <- Jupyter notebooks and py scripts
├── data
│   ├── processed           <- Modified data
│   └── raw                 <- Original, immutable data
├── dissemination           <- Material for dissemination
├── plots                   <- Generated figures
├── .gitignore              <- Files and folders ignored by git
├── .pre-commit-config.yaml <- Pre-commit hooks used
├── LICENSE.txt
├── README.md
└── environment.yml         <- Environment file to set up the environment using conda/mamba
```
