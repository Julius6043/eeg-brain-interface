# Optimizing EEG Electrode Use for Workload Detection in Noisy and Silent Environments

## Prerequisite

This project uses **Python 3.12** and [uv](https://docs.astral.sh/uv/#installation) as a package manager. The following steps are required to fully run the notebook.

1. Download the data folder from [here](https://nc.uni-bremen.de/index.php/s/nMTQ8wgAm53cycN) and unzip it in the root
   directory (as `./data`). This is necessary since the iteration through participants requires a specific directory
   structure.

2. Create a new virtual environment in the root directory with `uv venv --python 3.12 .venv`.

3. Install all requirements with `uv sync --frozen` (`--frozen` uses the versions in the `uv.lock` file).