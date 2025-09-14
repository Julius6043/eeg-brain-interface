# Optimizing EEG Electrode Use for Workload Detection in Noisy and Silent Environments

## Prerequisite

This project uses **Python 3.12** and [uv](https://docs.astral.sh/uv/#installation) as a package manager. The following
steps are required to fully run the notebook.

1. Download the data folder from [here](https://nc.uni-bremen.de/index.php/s/nMTQ8wgAm53cycN) and unzip it in the root
   directory (as `./data`). This is necessary since the iteration through participants requires a specific directory
   structure.

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

3. Create a new virtual environment in the root directory with `uv venv --python 3.12 .venv`.

4. Install all requirements with `uv sync --frozen` (`--frozen` uses the versions specified in the `uv.lock` file).

5. Activate the virtual environment (on Linux/MacOS: `source .venv/bin/activate`, on Windows:
   `.\.venv\Scripts\activate`).

## Running the Notebook

After installing all the dependencies and activating the virtual environment, you can run the notebook with:

```bash
uv run jupyter lab
```

(Alternatively, you can open this project in VSCode, PyCharm or another IDE that supports Jupyter notebooks.)