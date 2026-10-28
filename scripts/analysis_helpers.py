"""Small shared helpers for the SARCOS notebook and its PDF export."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
EXPECTED_COUNTS = {
    "blocked": {"train": 26665, "val": 8847, "test": 8872},
    "random": {"train": 26690, "val": 8897, "test": 8897},
}


def dataset_sha256(path: str | Path) -> str:
    """Return a file's SHA256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_names() -> list[str]:
    """Return names in the documented 7 position, 7 velocity, 7 acceleration order."""
    return (
        [f"joint_{i}_position" for i in range(1, 8)]
        + [f"joint_{i}_velocity" for i in range(1, 8)]
        + [f"joint_{i}_acceleration" for i in range(1, 8)]
    )


def score_regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the project's fixed MAE, RMSE, and R-squared metrics."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def scaled_pipeline(estimator) -> Pipeline:
    """Create a scaler/model pipeline so each fit estimates scaling from its fit rows."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def verify_frozen_indices(archive, manifest: dict, n_rows: int) -> pd.DataFrame:
    """Check counts, bounds, disjointness, manifest boundaries, and blocked purge geometry."""
    audit_rows = []
    for protocol, parts in EXPECTED_COUNTS.items():
        arrays = []
        for part, expected_count in parts.items():
            key = f"{protocol}_{part}"
            if key not in archive:
                raise KeyError(f"Missing frozen split array: {key}")
            indices = np.asarray(archive[key])
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise ValueError(f"{key} must be a one-dimensional integer array")
            if len(indices) != expected_count:
                raise ValueError(f"{key} count {len(indices)} != {expected_count}")
            if np.any(indices < 0) or np.any(indices >= n_rows):
                raise ValueError(f"{key} contains out-of-range rows")
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"{key} contains duplicate row indices")
            manifest_count = manifest["splits"][protocol][part]["count"]
            if len(indices) != manifest_count:
                raise ValueError(f"{key} count does not match the manifest")
            if protocol == "blocked":
                record = manifest["splits"][protocol][part]
                if len(indices) > 1 and not np.all(np.diff(indices) > 0):
                    raise ValueError(f"{key} must be strictly increasing")
                if len(indices) > 1 and not np.all(np.diff(indices) == 1):
                    raise ValueError(f"{key} must be contiguous")
                if int(indices[0]) != record["first_row_0based"]:
                    raise ValueError(f"{key} first row does not match the manifest")
                if int(indices[-1]) != record["last_row_0based"]:
                    raise ValueError(f"{key} last row does not match the manifest")
            arrays.append(indices)
            audit_rows.append(
                {
                    "Protocol": protocol.title(),
                    "Partition": part.title(),
                    "Count": len(indices),
                    "Minimum row": int(indices.min()),
                    "Maximum row": int(indices.max()),
                }
            )
        joined = np.concatenate(arrays)
        if len(np.unique(joined)) != len(joined):
            raise ValueError(f"{protocol} partitions overlap")
        if protocol == "random" and len(joined) != n_rows:
            raise ValueError("Random split does not cover every source row")
    cutpoints = manifest["blocked_cutpoints"]
    purge = manifest["blocked_purge_each_side"]
    first_cut, second_cut = map(int, cutpoints)
    purge = int(purge)
    expected_blocked = {
        "train": np.arange(0, first_cut - purge),
        "val": np.arange(first_cut + purge, second_cut - purge),
        "test": np.arange(second_cut + purge, n_rows),
    }
    for part, expected_indices in expected_blocked.items():
        actual_indices = np.asarray(archive[f"blocked_{part}"])
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(f"blocked_{part} does not match manifest cutpoints and purge")
    for left, right, boundary in [
        ("train", "val", first_cut),
        ("val", "test", second_cut),
    ]:
        left_indices = np.asarray(archive[f"blocked_{left}"])
        right_indices = np.asarray(archive[f"blocked_{right}"])
        omitted = np.arange(int(left_indices[-1]) + 1, int(right_indices[0]))
        expected_omitted = np.arange(boundary - purge, boundary + purge)
        if not np.array_equal(omitted, expected_omitted):
            raise ValueError(f"Blocked {left}/{right} gap does not match the intended purge")
    return pd.DataFrame(audit_rows)


def make_test_error_slices(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    predicted: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Use training quartiles to summarize held-out error slices."""
    train_values = {
        "Velocity norm": np.linalg.norm(train_x[:, 7:14], axis=1),
        "Acceleration norm": np.linalg.norm(train_x[:, 14:21], axis=1),
        "Absolute target": np.abs(train_y),
    }
    test_values = {
        "Velocity norm": np.linalg.norm(test_x[:, 7:14], axis=1),
        "Acceleration norm": np.linalg.norm(test_x[:, 14:21], axis=1),
        "Absolute target": np.abs(test_y),
    }
    residual = test_y - predicted
    tables, serialized = [], {}
    serialized_names = {
        "Velocity norm": "velocity_norm",
        "Acceleration norm": "acceleration_norm",
        "Absolute target": "absolute_torque",
    }
    for label, training_measure in train_values.items():
        training_min = float(np.min(training_measure))
        training_max = float(np.max(training_measure))
        outside_count = int(
            np.count_nonzero((test_values[label] < training_min) | (test_values[label] > training_max))
        )
        edges = np.unique(np.quantile(training_measure, [0, 0.25, 0.5, 0.75, 1]))
        # Repeated quantiles create zero-width bins; collapse duplicates before digitizing.
        bins = np.digitize(test_values[label], edges[1:-1], right=True)
        effective_slice_count = max(len(edges) - 1, 1)
        serialized[serialized_names[label]] = {
            "training_quartile_edges": edges.tolist(),
            "training_minimum": training_min,
            "training_maximum": training_max,
            "outside_training_range_count": outside_count,
            "effective_slice_count": effective_slice_count,
            "test_bins": [],
        }
        for i in range(effective_slice_count):
            mask = bins == i
            if effective_slice_count == 1:
                interval_label = "(-inf, +inf)"
            elif i == 0:
                interval_label = f"(-inf, {edges[i + 1]:.6g}]"
            elif i == len(edges) - 2:
                interval_label = f"({edges[i]:.6g}, +inf)"
            else:
                interval_label = f"({edges[i]:.6g}, {edges[i + 1]:.6g}]"
            row = {
                "Slice variable": label,
                "Slice interval (train-derived cutoffs)": interval_label,
                "n": int(mask.sum()),
                "Outside training min/max": outside_count,
                "Effective slices after duplicate cutoffs": effective_slice_count,
                "MAE": float(mean_absolute_error(test_y[mask], predicted[mask])) if mask.any() else None,
                "RMSE": float(np.sqrt(mean_squared_error(test_y[mask], predicted[mask]))) if mask.any() else None,
                "Mean signed residual": float(np.mean(residual[mask])) if mask.any() else None,
            }
            tables.append(row)
            serialized[serialized_names[label]]["test_bins"].append(
                {
                    "interval": (
                        [float(edges[i]), float(edges[i + 1])]
                        if len(edges) > 1
                        else [training_min, training_max]
                    ),
                    "interval_label": interval_label,
                    "n": row["n"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "mean_signed_residual": row["Mean signed residual"],
                }
            )
    return pd.DataFrame(tables), serialized


def _format_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format numeric report values without changing the full-precision JSON."""
    formatted = table.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            lower_column = column.lower()
            if (
                lower_column in {"n", "count", "outside training min/max", "original row"}
                or lower_column.startswith("outside ")
                or lower_column.endswith("rows")
            ):
                formatted[column] = formatted[column].map(
                    lambda value: "—" if pd.isna(value) else f"{int(value):,}"
                )
            else:
                formatted[column] = formatted[column].map(
                    lambda value: "—" if pd.isna(value) else f"{value:.4f}"
                )
    return formatted.fillna("—").astype(str)


def _add_report_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    tables: list[tuple[pd.DataFrame, list[float] | None, float]],
    *,
    portrait: bool = False,
) -> None:
    """Place brief explanations and one or more sized tables on a report page."""
    fig, ax = plt.subplots(figsize=(8.3, 11.7) if portrait else (11.7, 8.3))
    ax.axis("off")
    fig.text(0.065, 0.95, title, fontsize=17, weight="bold", va="top")
    y = 0.90
    for paragraph in paragraphs:
        fig.text(0.065, y, paragraph, fontsize=9.2, va="top", wrap=True, linespacing=1.35)
        y -= 0.04 + 0.035 * (len(paragraph) // (240 if not portrait else 180))
    for table, widths, height in tables:
        rendered_table = _format_report_table(table)
        rendered = ax.table(
            cellText=rendered_table.values,
            colLabels=list(rendered_table.columns),
            cellLoc="center",
            colLoc="center",
            bbox=[0.055, y - height, 0.89, height],
            colWidths=widths,
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8.0 if len(rendered_table.columns) > 5 else 8.7)
        for (row, _), cell in rendered.get_celld().items():
            cell.set_edgecolor("#aab4bf")
            cell.set_linewidth(0.45)
            cell.get_text().set_wrap(True)
            if row == 0:
                cell.set_facecolor("#dce6f1")
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f7fa")
        y -= height + 0.035
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_figure_page(pdf: PdfPages, title: str, image_path: str | Path, explanation: str) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.065, 0.95, title, fontsize=17, weight="bold", va="top")
    fig.text(0.065, 0.90, explanation, fontsize=9.2, va="top", wrap=True)
    image = plt.imread(image_path)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.78])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    output_path: str | Path,
    results: dict,
    metrics_table: pd.DataFrame,
    split_table: pd.DataFrame,
    alpha_table: pd.DataFrame,
    slice_table: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    figure_paths: list[str | Path],
    environment: dict[str, str],
    supplemental: dict | None = None,
) -> None:
    """Export a self-contained report with concise tables and interpretations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for protocol in ("blocked", "random"):
        for partition in ("validation", "test"):
            for model in ("Dummy mean", "OLS", "Ridge"):
                scores = results["scores"][protocol][partition][model]
                score_rows.append({
                    "Protocol": protocol.title(), "Partition": partition.title(), "Model": model,
                    "MAE": scores["MAE"], "RMSE": scores["RMSE"], "R²": scores["R2"],
                })
    compact_metrics = pd.DataFrame(score_rows)
    compact_splits = split_table[[
        column for column in (
            "Protocol", "Partition", "n", "Target mean",
            "Median velocity norm", "Median acceleration norm",
        ) if column in split_table.columns
    ]]
    compact_slices = slice_table[[
        column for column in (
            "Slice variable", "Slice interval (train-derived cutoffs)", "n", "Outside training min/max", "MAE", "RMSE"
        ) if column in slice_table.columns
    ]]
    compact_slices = compact_slices.rename(columns={
        "Outside training min/max": "Outside train range\ncount",
        "Slice interval (train-derived cutoffs)": "Slice interval\n(train-derived cutoffs)",
        "Absolute target": "Absolute torque",
    })
    if "Slice variable" in compact_slices:
        compact_slices["Slice variable"] = compact_slices["Slice variable"].replace(
            {"Absolute target": "Absolute torque"}
        )
    selected_alphas = pd.DataFrame([
        {"Protocol": name.title(), "Selected alpha": results["scores"][name]["selected_ridge_alpha"]}
        for name in ("blocked", "random")
    ])
    alpha_summary = alpha_table.copy()
    for column in ("Blocked validation MAE", "Random validation MAE"):
        alpha_summary[column] = alpha_summary[column].map(lambda value: f"{value:.4f}")
    coeffs = coefficient_table.rename(columns={
        "OLS coefficient": "OLS coef.", "Ridge coefficient": "Ridge coef."
    })
    with PdfPages(output_path) as pdf:
        _add_report_page(pdf, "Interpretable Regression Baselines and Diagnostics", [
            "SARCOS robot arm inverse dynamics | DS-A3",
            "Research question: how accurately do linear baselines predict first-joint torque from 21 kinematic features, and how does error vary across observed states?",
            "The 44,484 × 28 MATLAB matrix was downloaded from the public Kaixhin/SARCOS GitHub repository. The GPML dataset page documents the data and credits Sethu Vijayakumar. This project does not infer a data license from the repository software license, and the files do not verify physical units.",
            "Target: column 21 (zero-based); predictors: columns 0–20. The original test matrix overlaps the source matrix and was excluded from fitting, selection, and evaluation. Dataset SHA256: " + results["dataset_sha256"],
            "Method: training-mean Dummy, standardized OLS, and standardized Ridge. Scaling is fit within each training fold. Ridge alpha is selected by validation MAE from the preregistered grid; final models are refit on Train + Validation and evaluated once on Test.",
            "Blocked uses contiguous source-row regions with the frozen 25-row boundary purge. Random uses frozen random partitions. Neither establishes episode-independent generalization.",
            "Reproduction: activate nussem1, run the notebook, then run scripts/validate_project.py and python -m unittest discover -s tests -v. Environment: " + " | ".join(f"{key}: {value}" for key, value in environment.items()),
        ], [], portrait=True)

        _add_report_page(pdf, "Predictive performance", [
            "MAE is the primary metric; RMSE emphasizes larger residuals and R² summarizes variance explained. H1 is supported on the tested blocked evaluation: OLS materially improves on the training-mean baseline.",
            "Ridge provides no meaningful improvement in this tested setting. This conclusion is limited to these models, alpha grid, dataset, and evaluation protocols.",
        ], [(compact_metrics, [0.16, 0.16, 0.19, 0.16, 0.16, 0.17], 0.61)])

        _add_report_page(pdf, "Evaluation design and Ridge selection", [
            "The blocked protocol holds out contiguous row regions, with the manifest-defined purge gaps; it is sensitive to source ordering and cannot verify episode independence. The random protocol samples held-out rows from the mixed source distribution and may leave nearby observations across partitions.",
            "Blocked validation has markedly lower velocity and acceleration norms than its training and test regions, indicating distribution shift. Random partitions have more similar summaries. The protocol comparison is descriptive and does not by itself establish leakage.",
            "Alpha was selected using Validation MAE only. Test labels did not enter preprocessing, fitting, hyperparameter selection, slice thresholds, or model decisions.",
        ], [(compact_splits, [0.13, 0.13, 0.10, 0.16, 0.24, 0.24], 0.22),
            (alpha_summary, [0.18, 0.41, 0.41], 0.25),
            (selected_alphas, [0.45, 0.55], 0.10)])

        _add_report_page(pdf, "Descriptive error slices", [
            "Slices use Blocked Train quartile cutoffs, applied unchanged to Blocked Test. Open-ended end labels preserve the actual digitize assignments; the outside-range count indicates test observations beyond the training min/max.",
            "Higher acceleration and higher absolute torque slices have higher absolute errors. Velocity-slice error is non-monotonic: the low-velocity group has higher MAE than the highest-velocity group. These are descriptive associations. Higher absolute error at large torque does not establish higher relative error or a physical causal mechanism.",
        ], [(compact_slices, [0.14, 0.31, 0.07, 0.15, 0.16, 0.17], 0.52)])

        _add_report_page(pdf, "Coefficient summary", [
            "The displayed standardized coefficients are descriptive. Correlated kinematic features limit coefficient-level interpretation; they should not be read as separately identified physical parameters.",
        ], [(coeffs, [0.46, 0.27, 0.27], 0.70)])

        figure_notes = [
            ("Data exploration", "Training data summaries show broad operating-state variation. Marginal distributions do not establish temporal independence or identify physical units."),
            ("Blocked test residual diagnostics", "Residual plots characterize the final Blocked Ridge fit on held-out rows. They are diagnostic descriptions, not evidence of episode-level transfer."),
            ("Absolute error versus absolute torque", "The spread in absolute error increases in parts of the observed torque range. This plot does not normalize error by torque, so it cannot support claims about relative error."),
        ]
        for figure_path, (title, explanation) in zip(figure_paths, figure_notes):
            _add_figure_page(pdf, title, figure_path, explanation)

        if supplemental is not None:
            _add_report_page(pdf, "Additional Blocked OLS diagnostics", [
                "Influence statistics are computed on Blocked Train using an OLS model fit to those rows. Leverage is the diagonal of the projection matrix computed from a reduced QR factorization; no full n-by-n matrix is formed. Cook's distance uses the standard squared-residual/leverage expression.",
                "Heuristic flags use leverage > 2p/n and Cook's distance > 4/n. These thresholds identify observations for inspection; they do not establish data errors. The Breusch–Pagan auxiliary regression uses the 21 standardized predictors and training-fit squared residuals. Its null is constant conditional error variance. The conventional chi-square calibration assumes independent, correctly specified errors; strong serial dependence can make its p-value unreliable.",
                "Compare the test residual-versus-fitted plot with this training-only test. Both are descriptive; the test statistic is not used to change models.",
            ], [
                (supplemental["diagnostics_summary"], [0.24, 0.25, 0.25, 0.26], 0.16),
                (supplemental["influence_cases"], [0.19, 0.20, 0.20, 0.20, 0.21], 0.20),
            ])

            _add_figure_page(
                pdf,
                "OLS learning curve: Blocked Train and Validation",
                supplemental["learning_curve_figure"],
                "Each fixed subset is a nested sample from a seeded permutation of frozen Blocked Train rows. Training MAE is measured on the fitted subset; Validation MAE is measured on the same frozen Blocked Validation set at every size. The curve is descriptive under this sampling design, especially because the Blocked Validation region has a distinct low-motion distribution.",
            )

            _add_report_page(pdf, "Post-hoc Blocked Test error cases", [
                "These are the five largest absolute residuals from the already evaluated Blocked Ridge model, ranked after the original evaluation. They are shown for descriptive inspection only and were not used to tune or select any model. Torque magnitude is outcome-conditioned; velocity and acceleration norms are input-derived.",
                "Large residuals identify cases where this fitted model performed poorly in this held-out segment. The table does not establish a physical cause or imply that the observations are erroneous.",
            ], [(supplemental["critical_cases"], [0.12, 0.11, 0.11, 0.11, 0.12, 0.12, 0.15, 0.16], 0.34)])

            _add_report_page(pdf, "Exploratory Train/Validation model comparison", [
                "Lasso and HistGradientBoostingRegressor are post-hoc exploratory candidates. Their alpha/configuration comparisons use only Blocked Train and Blocked Validation; the previously inspected Test was not used for their selection or tuning. Do not interpret their validation scores as confirmatory estimates.",
                "Lasso zero coefficients are counted using the documented absolute tolerance. The fixed small HistGradientBoosting search provides one modest nonlinear comparison; its added flexibility costs interpretability and computation. Validation improvements, if any, are specific to this shifted validation region.",
            ], [(supplemental["model_comparison"], [0.17, 0.20, 0.20, 0.20, 0.23], 0.30)])

            _add_report_page(pdf, "Negative results and claim boundary", [
                "Ridge's improvement over OLS was negligible in the original tested setting. The exploratory additions are validation-only and do not replace or revise the original reported Test results.",
                "The near-complete overlap in the supplied historical SARCOS files is a data-provenance warning, not a usable independent benchmark: "
                + supplemental["historical_overlap_text"],
                "The project establishes predictive behavior only for these models, data, and frozen protocols. The Blocked Validation segment is markedly low-motion and unrepresentative of the other blocked partitions; this limits its usefulness for broader tuning. We therefore did not add blocked cross-validation or redesign tuning under the deadline. Serial dependence, unknown episode identifiers, and the local purge gap remain limitations.",
                "Standardized regression coefficients summarize conditional linear associations under this fitted feature set, not physical parameters or causal effects. Correlated kinematic predictors make individual coefficients unstable to feature dependence. Inverse-dynamics prediction estimates torque from observed state; it does not estimate the effect of intervening on a joint state while controlling the physical system and confounders.",
            ], [])

        _add_report_page(pdf, "Discussion, limitations, and conclusion", [
            "Conclusion: H1 is supported on the tested blocked evaluation. Ridge does not meaningfully improve on OLS under the tested models, alpha grid, dataset, and protocols. H3 is descriptive: high acceleration and high absolute torque slices have higher errors, while velocity error is non-monotonic.",
            "Limitations: source-row dependence is substantial; timestamps and reliable episode identifiers are unavailable; the local 25-row purge does not establish independence. Blocked regions show distribution shift, and random partitions may retain nearby observations. The original test matrix overlaps the training matrix and is not an independent benchmark. No claim is made about new robots, tasks, or episodes.",
            "The target's physical units and the dataset's redistribution license remain unverified. The public GitHub repository is the verified download source; the upstream GPML page is the dataset description and attribution reference. Further use or redistribution should follow the dataset owner's terms once verified.",
            "References: Kaixhin/SARCOS, https://github.com/Kaixhin/SARCOS ; GPML dataset page, https://gaussianprocess.org/gpml/data/ ; split manifest: data/splits/sarcos_split_manifest.json ; protocol: docs/experiment_protocol.md.",
        ], [], portrait=True)
