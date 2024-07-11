# Bicycle node network loop analysis

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

## Data setup

Data of the knudepunkter network comes from [BikeNodePlanner: Data for Denmark](https://github.com/anastassiavybornova/bike-node-planner-data-denmark) and [BikeNodePlanner](https://github.com/anastassiavybornova/bike-node-planner).

### Step 1: Extract data with BikeNodePlanner: Data for Denmark

- Use [BikeNodePlanner: Data for Denmark](https://github.com/anastassiavybornova/bike-node-planner-data-denmark)
- Uncomment the municipalities of your study area in `config-municipalities.yml`
- Set all values in `config-layers-polygon.yml` to `ignore`
- Run the `run.sh` script
- Copy all subfolders of `/input-for-bike-node-planner/` into the `/data/input/` folder of bike-node-planner

### Step 2: Generate network data with BikeNodePlanner

- Use [BikeNodePlanner](https://github.com/anastassiavybornova/bike-node-planner)
- Run scripts 01 to 04
- TO DO: Network simplification?
- Let's call `loopspath` the data/input path to your project, for example `bikenwloops/data/input/fyn/`
- Copy the file `nodes.gpkg` from `bike-node-planner/data/output/network` into `loopspath/network`
- Copy the file `edges_slope.gpkg` from `bike-node-planner/data/output/elevation` into `loopspath/network`
- Copy the whole folder `/input-for-bike-node-planner/point/` into `loopspath`

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
