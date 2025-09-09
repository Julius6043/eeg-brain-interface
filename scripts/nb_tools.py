import json
from pathlib import Path


def load_nb(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_nb(nb, path: Path):
    with path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write('\n')


def insert_markdown_cell(path: Path, index: int, lines: list[str]):
    nb = load_nb(path)
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith('\n') else l + '\n' for l in lines],
    }
    nb.setdefault('cells', [])
    nb['cells'].insert(index, cell)
    save_nb(nb, path)


if __name__ == '__main__':
    # Example usage (manual):
    # insert_markdown_cell(Path('notebooks/EEGNet_Optimized_Demo_clean.ipynb'), 1, ["## Konfiguration", "..."])
    pass

