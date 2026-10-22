"""Validate dataset provenance, frozen splits, and expected project artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/sarcos_inv.mat"
SPLIT = ROOT / "data/splits/sarcos_split_indices.npz"
MANIFEST = ROOT / "data/splits/sarcos_split_manifest.json"


def main() -> None:
    required = [DATA, ROOT / "data/raw/sarcos_inv_test.mat", SPLIT, MANIFEST,
                ROOT / "notebooks/DS_A3_regression_diagnostics.ipynb",
                ROOT / "docs/experiment_protocol.md", ROOT / "docs/AI_usage.md",
                ROOT / "scripts/analysis_helpers.py",
                ROOT / "reports/analysis_results.json",
                ROOT / "reports/refactor_consistency_check.json",
                ROOT / "reports/DS_A3_report.pdf",
                ROOT / "tests/test_analysis_helpers.py"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")
    manifest = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != manifest["data_file_sha256"]:
        raise SystemExit(f"Dataset SHA256 mismatch: {digest}")
    matrix = loadmat(DATA)["sarcos_inv"]
    if matrix.shape != (44484, 28) or matrix.shape[0] != manifest["n_rows"]:
        raise SystemExit(f"Unexpected dataset shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise SystemExit("Dataset contains NaN or infinite values")
    notebook = json.loads((ROOT / "notebooks/DS_A3_regression_diagnostics.ipynb").read_text())
    markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    if len(markdown_cells) < 12 or len(code_cells) < 10:
        raise SystemExit("Notebook does not contain the expected section-level analysis cells")
    if "runpy.run_path" in notebook_source:
        raise SystemExit("Notebook delegates the analysis instead of showing its code")
    notebook_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if notebook_errors:
        raise SystemExit(f"Notebook contains {len(notebook_errors)} error output(s)")
    consistency = json.loads((ROOT / "reports/refactor_consistency_check.json").read_text())
    if consistency.get("status") == "passed":
        if not consistency["all_validation_and_test_MAE_RMSE_R2_match"]:
            raise SystemExit("Refactored model metrics differ from the pre-refactor results")
        if not consistency["selected_alphas_match"] or not consistency["slice_thresholds_and_counts_match"]:
            raise SystemExit("Refactor changed selected Ridge alphas or error-slice definitions")
        if consistency["maximum_absolute_metric_delta"] > 1e-9:
            raise SystemExit("Metric consistency delta exceeds tolerance")
        if consistency["maximum_absolute_slice_metric_delta"] > 1e-9:
            raise SystemExit("Error-slice consistency delta exceeds tolerance")
    elif consistency.get("status") != "skipped":
        raise SystemExit("Historical consistency-check status must be passed or skipped")
    expected = {"blocked": {"train": 26665, "val": 8847, "test": 8872},
                "random": {"train": 26690, "val": 8897, "test": 8897}}
    with np.load(SPLIT) as archive:
        for protocol, parts in expected.items():
            arrays = []
            for part, count in parts.items():
                key = f"{protocol}_{part}"
                if key not in archive:
                    raise SystemExit(f"Missing frozen index array: {key}")
                idx = archive[key]
                if idx.dtype.kind not in "iu" or len(idx) != count:
                    raise SystemExit(f"Invalid {key}: dtype={idx.dtype}, n={len(idx)}")
                if np.any(idx < 0) or np.any(idx >= len(matrix)):
                    raise SystemExit(f"Out-of-range indices in {key}")
                if len(np.unique(idx)) != len(idx):
                    raise SystemExit(f"Repeated indices in {key}")
                record = manifest["splits"][protocol][part]
                if record["count"] != len(idx):
                    raise SystemExit(f"Manifest count mismatch for {key}")
                if protocol == "blocked" and (int(idx[0]) != record["first_row_0based"] or
                                                  int(idx[-1]) != record["last_row_0based"]):
                    raise SystemExit(f"Manifest range mismatch for {key}")
                arrays.append(idx)
            if protocol == "blocked":
                cut1, cut2 = map(int, manifest["blocked_cutpoints"])
                purge = int(manifest["blocked_purge_each_side"])
                expected_blocked = {
                    "train": np.arange(0, cut1 - purge),
                    "val": np.arange(cut1 + purge, cut2 - purge),
                    "test": np.arange(cut2 + purge, len(matrix)),
                }
                for part, idx in zip(parts, arrays):
                    if len(idx) > 1 and not np.all(np.diff(idx) > 0):
                        raise SystemExit(f"Blocked {part} indices are not strictly increasing")
                    if len(idx) > 1 and not np.all(np.diff(idx) == 1):
                        raise SystemExit(f"Blocked {part} indices are not contiguous")
                    if not np.array_equal(idx, expected_blocked[part]):
                        raise SystemExit(
                            f"Blocked {part} boundaries or purge gap differ from the manifest"
                        )
                for left, right, boundary in [
                    (arrays[0], arrays[1], cut1),
                    (arrays[1], arrays[2], cut2),
                ]:
                    omitted = np.arange(int(left[-1]) + 1, int(right[0]))
                    intended = np.arange(boundary - purge, boundary + purge)
                    if not np.array_equal(omitted, intended):
                        raise SystemExit("Blocked partitions do not have the intended purge gaps")
            combined = np.concatenate(arrays)
            if len(np.unique(combined)) != len(combined):
                raise SystemExit(f"Split overlap in {protocol}")
            if protocol == "random" and len(combined) != len(matrix):
                raise SystemExit("Random protocol does not partition all rows")
    print(f"PASS: SHA256 {digest}")
    print("PASS: source shape (44484, 28), finite values, split ranges, strict blocked order/contiguity, intended purge gaps, and disjoint partitions")
    print(f"PASS: optional historical consistency status is {consistency['status']}")
    print("PASS: analysis helpers, result snapshot, consistency check, and PDF exist")


if __name__ == "__main__":
    main()
