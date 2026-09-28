from xor_study.experiments import run_clean_experiment, run_noisy_experiment


def test_clean_experiment_separates_only_interaction_capable_maps():
    results = {r.feature_map: r for r in run_clean_experiment(seeds=(0, 1, 2))}

    assert results["raw"].convergence_rate == 0.0
    assert results["square"].convergence_rate == 0.0
    assert results["interaction"].convergence_rate == 1.0
    assert results["quadratic"].convergence_rate == 1.0


def test_clean_experiment_preserves_per_run_evidence():
    result = {r.feature_map: r for r in run_clean_experiment(seeds=(2, 3))}[
        "interaction"
    ]

    assert result.seeds == (2, 3)
    assert len(result.weights) == 2
    assert len(result.mistakes_per_epoch) == 2
    assert result.mean_train_error == 0.0
    assert result.mean_test_error is None


def test_noisy_experiment_is_reproducible():
    first = run_noisy_experiment(seed=13)
    second = run_noisy_experiment(seed=13)

    assert first == second
    assert {r.feature_map for r in first.results} == {"raw", "interaction"}


def test_noisy_experiment_records_split_and_crossing_metadata():
    experiment = run_noisy_experiment(
        seed=5,
        samples_per_corner=20,
        training_seeds=(0, 1),
    )

    assert experiment.train_samples == 60
    assert experiment.test_samples == 20
    assert 0.0 <= experiment.axis_crossing_rate <= 1.0
    assert all(result.mean_test_error is not None for result in experiment.results)
