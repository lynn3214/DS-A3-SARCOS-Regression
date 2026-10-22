# Interpretable Regression Baselines and Diagnostics for Robot Arm Inverse Dynamics

**Course:** UCAS Data Science — Assignment 3 (DS-A3)

This project studies prediction of first-joint torque from the SARCOS robot arm kinematic state. It compares a training-mean Dummy Regressor with ordinary least squares (OLS) and Ridge regression, using frozen blocked and random train/validation/test splits.

## Research question

How accurately can interpretable regularized linear models predict first-joint torque from 21 state features, and under which observed conditions do predictions become less reliable?

The predictors are seven joint positions, seven velocities, and seven accelerations (columns 0–20). The target is first-joint torque (zero-based column 21). No physical units are asserted.

## Dataset availability and provenance

The upstream [Gaussian Processes for Machine Learning dataset page](https://gaussianprocess.org/gpml/data/) describes SARCOS and credits Sethu Vijayakumar. The public [Kaixhin/SARCOS GitHub repository](https://github.com/Kaixhin/SARCOS) was the mirror used to obtain the original MATLAB files. This project does not treat the mirror's software license as a license for its dataset. The dataset redistribution license has not been independently verified, so the original MATLAB files are intentionally excluded from this repository.

Download sarcos_inv.mat and sarcos_inv_test.mat from the files listed in the GitHub repository and place them under data/raw/ with those exact filenames. For example, from the project root:

    mkdir -p data/raw
    curl -L --fail -o data/raw/sarcos_inv.mat https://raw.githubusercontent.com/Kaixhin/SARCOS/master/sarcos_inv.mat
    curl -L --fail -o data/raw/sarcos_inv_test.mat https://raw.githubusercontent.com/Kaixhin/SARCOS/master/sarcos_inv_test.mat

sarcos_inv.mat is the source for the regression experiments. sarcos_inv_test.mat is retained locally only as historical data-leakage evidence. It is not used for model training, validation, or final evaluation. No physical units are documented in the supplied files.

The source file SHA256 recorded in data/splits/sarcos_split_manifest.json is:

    b8a249733253ba6097372fedee7696833fcf30de42037d5b4a7227f21a6d1d97

## Reproduction

Run commands from the project root. Download both source MATLAB files before data-dependent validation or Notebook execution. The frozen split files are included and must not be regenerated.

    conda create -n nussem1 python=3.10
    conda activate nussem1
    python -m pip install -r requirements.txt

Then validate the source hash and frozen indices, run the helper tests, and execute the Notebook:

    python scripts/validate_project.py
    python -m unittest discover -s tests -v
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/DS_A3_regression_diagnostics.ipynb

The Notebook presents the analytical steps and their tables and plots. Reusable checks, scoring, slice summaries, and PDF layout helpers are in scripts/analysis_helpers.py. scripts/run_analysis.py is a convenience runner. The historical results snapshot in reports/pre_refactor_analysis_results.json is optional; the Notebook runs without it.

## Methodology

The primary evaluation uses frozen contiguous Blocked partitions with a 25-row purge at each internal boundary. A frozen seed-42 Random split provides sensitivity analysis. Preprocessing is fit on training rows only. Ridge hyperparameters are selected independently for each protocol using Validation MAE on the fixed alpha grid; the selected models are refit on Train + Validation and evaluated once on Test. Test labels do not inform fitting, scaling, hyperparameter selection, slice cutoffs, or analysis decisions.

Reported metrics are MAE, RMSE, and R². The executed Notebook and PDF include residual diagnostics and error slices for velocity norm, acceleration norm, and absolute target torque. Slice thresholds come from the corresponding training partition and are applied unchanged to held-out rows. Full-precision results and audit records are in reports/.

## Repository layout

    data/raw/       Locally downloaded source matrices (not tracked)
    data/splits/    Frozen indices and manifest (tracked)
    notebooks/      Executed analytical Notebook
    scripts/        Analysis helpers and validation tools
    tests/          Synthetic helper tests
    reports/        PDF, figures, metrics, and data/split audit evidence
    docs/           Experiment protocol and AI-use record

## Limitations

The original SARCOS test matrix substantially overlaps the source training matrix, so it is excluded from predictive evaluation. Row order is dependent and reliable episode identifiers are unavailable. The 25-row boundary purge is a local safeguard and does not guarantee independent observations or episode-independent generalization. Blocked partitions show feature and target distribution shift; the Random split may retain nearby samples. The difference between protocols alone does not establish leakage. Dataset redistribution terms and physical units remain unverified.
