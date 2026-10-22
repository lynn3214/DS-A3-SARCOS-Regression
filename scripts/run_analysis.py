"""Execute the analytical notebook with the currently active Python environment."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DS_A3_regression_diagnostics.ipynb"


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(NOTEBOOK.relative_to(ROOT)),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
