import csv
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_notebook_executes_without_the_local_package(tmp_path):
    source = Path("notebooks/xor_feature_space_study.ipynb")
    notebook = nbformat.read(source, as_version=4)

    notebook.cells.insert(
        0, nbformat.v4.new_code_cell("import sys\nsys.modules['xor_study'] = None")
    )

    NotebookClient(notebook, timeout=180, kernel_name="python3").execute(
        cwd=tmp_path
    )

    assert not [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]

    with (tmp_path / "results" / "clean_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {
        row["feature_map"]: float(row["convergence_rate"]) for row in rows
    } == {"raw": 0.0, "square": 0.0, "interaction": 1.0, "quadratic": 1.0}


def test_notebook_is_saved_as_code_without_outputs():
    notebook = nbformat.read("notebooks/xor_feature_space_study.ipynb", as_version=4)
    assert notebook.cells
    assert all(cell.cell_type == "code" for cell in notebook.cells)
    assert all(cell.execution_count is None for cell in notebook.cells)
    assert all(not cell.outputs for cell in notebook.cells)
