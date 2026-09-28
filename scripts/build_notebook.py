"""Build and execute the XOR notebook deterministically."""

import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "xor_feature_space_study.ipynb"


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def _module_source(name: str) -> str:
    source = (ROOT / "src" / "xor_study" / f"{name}.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("xor_study"):
            del lines[node.lineno - 1 : node.end_lineno]
    return "\n".join(lines)


def build_notebook():
    cells = [
        code(
            """
import importlib.util
import subprocess
import sys

if any(importlib.util.find_spec(name) is None for name in ("numpy", "matplotlib")):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "numpy>=1.24,<3", "matplotlib>=3.7,<4",
    ])
"""
        ),
        *(code(_module_source(name)) for name in (
            "data", "features", "perceptron", "experiments", "plots"
        )),
        code(
            """
from pathlib import Path
from IPython.display import Image, display

ROOT = Path.cwd()

np.set_printoptions(suppress=True, precision=3)
manifest = generate_all_outputs(ROOT, seed=42)
"""
        ),
        code(
            """
X, y = clean_xor()
print("X =")
print(X)
print("y =", y)
display(Image(filename=manifest.xor_input_space))

for name in FEATURE_MAPS:
    print(f"{name:11s} ->")
    print(transform_features(X, name))
"""
        ),
        code(
            """
interaction_features = transform_features(X, "interaction")
separator = np.array([0.0, 0.0, 0.0, -1.0])
assert np.array_equal(predict_labels(interaction_features, separator), y)
print("scores =", interaction_features @ separator)
display(Image(filename=manifest.interaction_feature_separation))
"""
        ),
        code(
            """
clean_results = run_clean_experiment(seeds=DEFAULT_TRAINING_SEEDS)
print("map          convergence  mean updates  mean epochs  train error")
for result in clean_results:
    print(
        f"{result.feature_map:11s} {result.convergence_rate:11.2f} "
        f"{result.mean_updates:13.2f} {result.mean_epochs:12.2f} "
        f"{result.mean_train_error:11.3f}"
    )
display(Image(filename=manifest.feature_map_comparison))
display(Image(filename=manifest.mistakes_by_epoch))
"""
        ),
        code(
            """
data = make_noisy_xor_clusters(samples_per_corner=100, noise_std=0.35, seed=42)
split = stratified_split(data.X, data.y, test_fraction=0.25, seed=42)

def count_axis_crossings(X, y):
    ideal_labels = np.where(X[:, 0] * X[:, 1] <= 0.0, 1, -1)
    return int(np.count_nonzero(ideal_labels != y))

print({
    "train_samples": split.y_train.size,
    "test_samples": split.y_test.size,
    "train_axis_crossings": count_axis_crossings(split.X_train, split.y_train),
    "test_axis_crossings": count_axis_crossings(split.X_test, split.y_test),
})

noisy = run_noisy_experiment(seed=42, training_seeds=DEFAULT_TRAINING_SEEDS)
print("map          convergence  train error (mean, SD)  test error (mean, SD)")
for result in noisy.results:
    print(
        f"{result.feature_map:11s} {result.convergence_rate:11.2f} "
        f"{result.mean_train_error:10.4f} {result.std_train_error:8.4f} "
        f"{result.mean_test_error:10.4f} {result.std_test_error:8.4f}"
    )
display(Image(filename=manifest.noisy_decision_regions))
"""
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"xor-{index:02d}"
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        record_timing=False,
    ).execute(cwd=ROOT)
    for cell in notebook.cells:
        cell.execution_count = None
        cell.outputs = []
    nbformat.write(notebook, TARGET)
    print(TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
