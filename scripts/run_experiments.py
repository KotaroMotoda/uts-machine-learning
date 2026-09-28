"""Regenerate all tables and figures from the repository root."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xor_study.plots import generate_all_outputs  # noqa: E402


def main() -> None:
    manifest = generate_all_outputs(ROOT, seed=42)
    for path in manifest.all_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
