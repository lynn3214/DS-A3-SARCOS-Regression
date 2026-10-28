# Interpretable Regression Baselines and Diagnostics for Robot Arm Inverse Dynamics

**Course:** UCAS Data Science — Assignment 3 (DS-A3)

This project studies prediction of first-joint torque from the SARCOS robot arm kinematic state. It compares a training-mean Dummy Regressor with ordinary least squares (OLS) and Ridge regression, using frozen blocked and random train/validation/test splits.

## Research question

How accurately can interpretable regularized linear models predict first-joint torque from 21 state features, and under which observed conditions do predictions become less reliable?

The predictors are seven joint positions, seven velocities, and seven accelerations (columns 0–20). The target is first-joint torque (zero-based column 21). No physical units are asserted.

## Dataset availability and provenance

The upstream [Gaussian Processes for Machine Learning dataset page](https://gaussianprocess.org/gpml/data/) describes SARCOS and credits Sethu Vijayakumar. The public [Kaixhin/SARCOS GitHub repository](https://github.com/Kaixhin/SARCOS) was the mirror used to obtain the original MATLAB files; the verified source revision is bd4800d952ec1a9153cdc84a325ac796f140265e. This project does not treat the mirror's software license as a license for its dataset. The dataset redistribution license has not been independently verified, so the original MATLAB files are intentionally excluded from this repository.

Download sarcos_inv.mat and sarcos_inv_test.mat from that pinned GitHub revision and place them under data/raw/ with those exact filenames. For example, from the project root:

    mkdir -p data/raw
    curl -L --fail -o data/raw/sarcos_inv.mat https://raw.githubusercontent.com/Kaixhin/SARCOS/bd4800d952ec1a9153cdc84a325ac796f140265e/sarcos_inv.mat
    curl -L --fail -o data/raw/sarcos_inv_test.mat https://raw.githubusercontent.com/Kaixhin/SARCOS/bd4800d952ec1a9153cdc84a325ac796f140265e/sarcos_inv_test.mat

sarcos_inv.mat is the source for the regression experiments. sarcos_inv_test.mat is retained locally only as historical data-leakage evidence. It is not used for model training, validation, or final evaluation. No physical units are documented in the supplied files.

The source file SHA256 recorded in data/splits/sarcos_split_manifest.json is:

    b8a249733253ba6097372fedee7696833fcf30de42037d5b4a7227f21a6d1d97

The historical test-file SHA256, recorded in the data audit, is:

    161a59b5c3b4f4b404584323f181607b2acbe620eb134dc720760dc3f38f5cec

## Reproduction

Run commands from the project root. Download both source MATLAB files before data-dependent validation or Notebook execution. The frozen split files are included and must not be regenerated. The explicit environment lock was exported from the existing nussem1 environment on macOS 15.5 ARM64; it is platform-specific.

    conda create -n nussem1 --file environment-lock.txt
    conda activate nussem1
    python -m pip install -r requirements.txt

Then validate the source hash and frozen indices, run the helper tests, and execute the Notebook:

    python scripts/validate_project.py
    python -m unittest discover -s tests -v
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/DS_A3_regression_diagnostics.ipynb

The Notebook presents the analytical steps and their tables and plots. Reusable checks, scoring, slice summaries, and PDF layout helpers are in scripts/analysis_helpers.py. scripts/run_analysis.py is a convenience runner. The historical results snapshot in reports/pre_refactor_analysis_results.json is optional; the Notebook runs without it.

## Methodology

The primary evaluation uses frozen contiguous Blocked partitions with a 25-row purge at each internal boundary. A frozen seed-42 Random split provides sensitivity analysis. Preprocessing is fit on training rows only. Ridge hyperparameters are selected independently for each protocol using Validation MAE on the fixed alpha grid; the selected models are refit on Train + Validation and evaluated once on Test. Test labels do not inform fitting, scaling, hyperparameter selection, slice cutoffs, or analysis decisions.

Reported metrics are MAE, RMSE, and R². The executed Notebook and PDF include residual diagnostics, leverage/Cook's-distance screening, a training-residual Breusch–Pagan test, a seeded OLS learning curve, critical error cases, and error slices for velocity norm, acceleration norm, and absolute target torque. The absolute-target slice is retrospective and outcome-conditioned; velocity and acceleration slices are input-derived. Slice thresholds come from the corresponding training partition and are applied unchanged to held-out rows.

Post-hoc exploratory comparisons add validation-selected Lasso and a small HistGradientBoostingRegressor search. They use Blocked Train/Validation only, do not alter the original three-model Test results, and are not confirmatory performance claims. Full-precision primary results, exploratory results, and audit records are in reports/.

## Repository layout

    data/raw/       Locally downloaded source matrices (not tracked)
    data/splits/    Frozen indices and manifest (tracked)
    notebooks/      Executed analytical Notebook
    scripts/        Analysis helpers and validation tools
    tests/          Synthetic helper tests
    reports/        PDF, figures, primary/exploratory metrics, and audit evidence
    docs/           Experiment protocol and AI-use record

## Limitations

The original SARCOS test matrix substantially overlaps the source training matrix: 4,446 of 4,449 aligned rows match exactly across all columns, and all 4,449 are within 4.0000001e-6 maximum absolute cell difference. It is excluded from predictive evaluation. Row order is dependent and reliable episode identifiers are unavailable. The 25-row boundary purge is a local safeguard and does not guarantee independent observations or episode-independent generalization. Blocked Validation is a distinct low-motion region, limiting its representativeness for tuning; blocked cross-validation and extensive retuning were not added. The Random split may retain nearby samples. The difference between protocols alone does not establish leakage. Dataset redistribution terms and physical units remain unverified.
