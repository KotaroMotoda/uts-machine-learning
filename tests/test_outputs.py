import json
from pathlib import Path

from xor_study.plots import generate_all_outputs


def test_generate_all_outputs_creates_nonempty_evidence(tmp_path: Path):
    manifest = generate_all_outputs(tmp_path, seed=42)

    for path in manifest.all_paths:
        assert path.exists()
        assert path.stat().st_size > 100
    assert manifest.summary_json.name == "experiment_summary.json"
    assert (
        manifest.interaction_feature_separation.name
        == "interaction_feature_separation.png"
    )


def test_generated_summary_records_reproducibility_settings(tmp_path: Path):
    manifest = generate_all_outputs(tmp_path, seed=17)
    summary = json.loads(manifest.summary_json.read_text(encoding="utf-8"))

    assert summary["seed"] == 17
    assert summary["clean"]["training_seeds"] == list(range(20))
    assert summary["noisy"]["samples_per_corner"] == 100
    assert {item["feature_map"] for item in summary["clean"]["results"]} == {
        "raw",
        "square",
        "interaction",
        "quadratic",
    }
