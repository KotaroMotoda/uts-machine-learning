"""Generate deterministic tables and figures for the XOR study."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from xor_study.data import clean_xor, make_noisy_xor_clusters
from xor_study.experiments import (
    DEFAULT_TRAINING_SEEDS,
    ExperimentResult,
    NoisyExperiment,
    run_clean_experiment,
    run_noisy_experiment,
)
from xor_study.features import FEATURE_MAPS, transform_features
from xor_study.perceptron import predict_labels


NEGATIVE_COLOUR = "#355C7D"
POSITIVE_COLOUR = "#C06C84"
MAP_COLOURS = ("#6C5B7B", "#4C956C", "#F2A65A", "#D1495B")


@dataclass(frozen=True)
class OutputManifest:
    clean_csv: Path
    noisy_csv: Path
    summary_json: Path
    xor_input_space: Path
    feature_map_comparison: Path
    interaction_feature_separation: Path
    mistakes_by_epoch: Path
    noisy_decision_regions: Path

    @property
    def all_paths(self) -> Tuple[Path, ...]:
        return (
            self.clean_csv,
            self.noisy_csv,
            self.summary_json,
            self.xor_input_space,
            self.feature_map_comparison,
            self.interaction_feature_separation,
            self.mistakes_by_epoch,
            self.noisy_decision_regions,
        )


def _result_row(result: ExperimentResult) -> dict:
    return {
        "feature_map": result.feature_map,
        "runs": len(result.seeds),
        "convergence_rate": result.convergence_rate,
        "mean_updates": result.mean_updates,
        "std_updates": result.std_updates,
        "mean_epochs": result.mean_epochs,
        "std_epochs": result.std_epochs,
        "mean_train_error": result.mean_train_error,
        "std_train_error": result.std_train_error,
        "mean_test_error": result.mean_test_error,
        "std_test_error": result.std_test_error,
    }


def _write_results_csv(path: Path, results: Iterable[ExperimentResult]) -> None:
    rows = [_result_row(result) for result in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_input_space(path: Path) -> None:
    X, y = clean_xor()
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    for label, colour, name in (
        (-1, NEGATIVE_COLOUR, "Class -1"),
        (1, POSITIVE_COLOUR, "Class +1"),
    ):
        mask = y == label
        axis.scatter(
            X[mask, 0],
            X[mask, 1],
            s=130,
            c=colour,
            edgecolors="white",
            linewidths=1.5,
            label=name,
        )
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set(xlabel="$x_1$", ylabel="$x_2$", title="XOR in the original input space")
    axis.set_xticks((-1, 0, 1))
    axis.set_yticks((-1, 0, 1))
    axis.set_aspect("equal")
    axis.legend()
    _save_figure(figure, path)


def _plot_feature_comparison(
    path: Path, results: Tuple[ExperimentResult, ...]
) -> None:
    names = [result.feature_map for result in results]
    x = np.arange(len(names))
    figure, (convergence_axis, error_axis) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    convergence_axis.bar(
        x,
        [result.convergence_rate for result in results],
        color=MAP_COLOURS,
    )
    convergence_axis.set(
        xticks=x,
        xticklabels=names,
        ylim=(0, 1.08),
        ylabel="Convergence rate",
        title=f"{len(results[0].seeds)} shuffled training orders",
    )
    error_axis.bar(
        x,
        [result.mean_train_error for result in results],
        color=MAP_COLOURS,
    )
    error_axis.set(
        xticks=x,
        xticklabels=names,
        ylim=(0, 0.55),
        ylabel="Mean empirical error",
        title="Final error on four XOR points",
    )
    for axis in (convergence_axis, error_axis):
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Feature map determines whether a linear model can represent XOR")
    _save_figure(figure, path)


def _plot_interaction_separation(path: Path) -> None:
    X, y = clean_xor()
    interaction = transform_features(X, "interaction")[:, -1]
    offsets = np.array([-0.06, 0.06, -0.06, 0.06])
    figure, axis = plt.subplots(figsize=(8.0, 3.2))
    axis.axvspan(-1.5, 0.0, color="#F5DDE5", alpha=0.75)
    axis.axvspan(0.0, 1.5, color="#DCEAF3", alpha=0.75)
    axis.axvline(0.0, color="#263238", linestyle="--", linewidth=1.4)
    for label, colour, name in (
        (-1, NEGATIVE_COLOUR, "Class -1"),
        (1, POSITIVE_COLOUR, "Class +1"),
    ):
        mask = y == label
        axis.scatter(
            interaction[mask],
            offsets[mask],
            s=120,
            c=colour,
            edgecolors="white",
            linewidths=1.5,
            label=name,
            zorder=3,
        )
    axis.text(-0.75, 0.14, "predict +1", ha="center", color=POSITIVE_COLOUR)
    axis.text(0.75, 0.14, "predict -1", ha="center", color=NEGATIVE_COLOUR)
    axis.set(
        xlim=(-1.5, 1.5),
        ylim=(-0.16, 0.22),
        xticks=(-1, 0, 1),
        yticks=[],
        xlabel=FEATURE_MAPS["interaction"].feature_names[-1],
        title="One-dimensional separation after adding the interaction feature",
    )
    axis.legend(loc="lower center", ncol=2)
    _save_figure(figure, path)


def _plot_mistakes(path: Path, results: Tuple[ExperimentResult, ...]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for result, colour in zip(results, MAP_COLOURS):
        history = result.mistakes_per_epoch[0]
        axis.plot(
            np.arange(1, len(history) + 1),
            history,
            label=result.feature_map,
            color=colour,
            linewidth=2,
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Number of updates",
        title="Perceptron mistakes: interaction features converge; raw features cycle",
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_noisy_regions(path: Path, experiment: NoisyExperiment) -> None:
    data = make_noisy_xor_clusters(
        experiment.samples_per_corner,
        experiment.noise_std,
        seed=experiment.seed,
    )
    grid_values = np.linspace(-2.1, 2.1, 220)
    grid_x1, grid_x2 = np.meshgrid(grid_values, grid_values)
    grid = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    background = ListedColormap(("#DCEAF3", "#F5DDE5"))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    for axis, result in zip(axes, experiment.results):
        weights = np.asarray(result.weights[0], dtype=np.float64)
        predictions = predict_labels(transform_features(grid, result.feature_map), weights)
        axis.contourf(
            grid_x1,
            grid_x2,
            predictions.reshape(grid_x1.shape),
            levels=(-1.5, 0, 1.5),
            cmap=background,
            alpha=0.8,
        )
        for label, colour in ((-1, NEGATIVE_COLOUR), (1, POSITIVE_COLOUR)):
            mask = data.y == label
            axis.scatter(
                data.X[mask, 0],
                data.X[mask, 1],
                c=colour,
                s=12,
                alpha=0.65,
                edgecolors="none",
                label=f"Class {label:+d}",
            )
        axis.set(
            xlabel="$x_1$",
            ylabel="$x_2$",
            title=(
                f"{result.feature_map}: region seed {result.seeds[0]}; "
                f"mean test error {result.mean_test_error:.3f}"
            ),
        )
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Noisy XOR decision regions (colours show predictions; points show labels)"
    )
    _save_figure(figure, path)


def generate_all_outputs(output_root: Path, seed: int = 42) -> OutputManifest:
    """Generate tables and figures under ``output_root``."""

    root = Path(output_root)
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    training_seeds = DEFAULT_TRAINING_SEEDS
    clean_results = run_clean_experiment(seeds=training_seeds)
    noisy_experiment = run_noisy_experiment(seed=seed, training_seeds=training_seeds)
    manifest = OutputManifest(
        clean_csv=results_dir / "clean_results.csv",
        noisy_csv=results_dir / "noisy_results.csv",
        summary_json=results_dir / "experiment_summary.json",
        xor_input_space=figures_dir / "xor_input_space.png",
        feature_map_comparison=figures_dir / "feature_map_comparison.png",
        interaction_feature_separation=figures_dir
        / "interaction_feature_separation.png",
        mistakes_by_epoch=figures_dir / "mistakes_by_epoch.png",
        noisy_decision_regions=figures_dir / "noisy_decision_regions.png",
    )

    _write_results_csv(manifest.clean_csv, clean_results)
    _write_results_csv(manifest.noisy_csv, noisy_experiment.results)
    summary = {
        "seed": seed,
        "clean": {
            "training_seeds": list(training_seeds),
            "results": [asdict(result) for result in clean_results],
        },
        "noisy": asdict(noisy_experiment),
    }
    manifest.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _plot_input_space(manifest.xor_input_space)
    _plot_feature_comparison(manifest.feature_map_comparison, clean_results)
    _plot_interaction_separation(manifest.interaction_feature_separation)
    _plot_mistakes(manifest.mistakes_by_epoch, clean_results)
    _plot_noisy_regions(manifest.noisy_decision_regions, noisy_experiment)
    return manifest
