# Experiment Protocol

## Question and estimand

Estimate held-out predictive performance for first-joint torque (column 21, zero-based) from columns 0–20 of `sarcos_inv.mat`. Inputs are seven joint positions, seven velocities, and seven accelerations. The task is single-output prediction. No physical units are asserted because they are not documented in the supplied files.

## Data and frozen partitions

The MATLAB files were downloaded from the public [Kaixhin/SARCOS GitHub repository](https://github.com/Kaixhin/SARCOS), which lists both files. The upstream [GPML dataset page](https://gaussianprocess.org/gpml/data/) describes the matrix schema and credits Sethu Vijayakumar. Local file hashes are recorded in data/splits/sarcos_split_manifest.json. The repository software license is not assumed to cover the data, and physical units are not stated because they are unverified.

The source matrix is checked against the SHA256 in data/splits/sarcos_split_manifest.json. The pre-existing index arrays are authoritative and are never regenerated during analysis. Primary evaluation uses the contiguous blocked partition with 25 excluded rows around each internal boundary: Train 26,665, Validation 8,847, Test 8,872. Sensitivity evaluation uses the frozen seed-42 random partition: 26,690 / 8,897 / 8,897.

The original `sarcos_inv_test.mat` is excluded because it substantially overlaps the source training file. Feature scaling is fit separately on each protocol's training data. Model selection uses validation MAE only. Ridge alpha candidates are `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`. Once configurations are selected, each final model is refit on Train + Validation and evaluated once on Test.

## Models and metrics

Compare a training-mean DummyRegressor, StandardScaler + LinearRegression, and StandardScaler + Ridge. Report MAE as the primary metric, with RMSE and R² as secondary metrics. Ridge is selected independently for each split protocol using validation labels. No test labels inform fitting, preprocessing, hyperparameter selection, slice thresholds, or diagnostic choices.

## Diagnostics and slices

Residual is defined as `actual - predicted`; positive values mean underprediction. Show actual-versus-predicted, residual-versus-predicted, residual distribution, and absolute error against velocity norm, acceleration norm, and absolute target magnitude. Error-slice thresholds are quartiles calculated from the training partition only and then applied unchanged to validation and test data. Slice tables include count, MAE, RMSE, and mean signed residual.

## Scope and limitations

Blocked evaluation estimates prediction on held-out contiguous rows. Row order shows substantial dependence, but no verified episode identifiers are available. The 25-row gap is a local safeguard and does not establish independence. The blocked validation region has notably lower velocity and acceleration norms than the training and test regions, so distribution shift must be considered. Random evaluation estimates performance on held-out observations from the mixed source distribution and likely retains local sample similarity. Performance differences between protocols alone do not establish leakage.

The absolute-target slice is retrospective and outcome-conditioned because it uses response magnitude; it is not an input-only deployment condition. Velocity and acceleration slices are input-derived. OLS leverage and Cook's distance are screened on fitting data using reduced QR factorization. The Breusch–Pagan auxiliary regression uses squared training-fit residuals and standardized predictors; its conventional chi-square p-value assumes independent errors and must be interpreted cautiously under serial dependence. A seeded nested-subset OLS learning curve uses Blocked Train and evaluates on Blocked Validation only. The five largest previously observed Blocked Test residuals are listed as post-hoc descriptive cases; they do not inform model choice.

Post-hoc exploratory Lasso and HistGradientBoostingRegressor candidates are selected only on Blocked Validation after fitting on Blocked Train. They are not evaluated on Test and do not replace the original Dummy/OLS/Ridge results. Blocked cross-validation and extensive tuning were not added because the frozen Blocked Validation segment is a distinct low-motion regime and is not representative of all blocked regions; redesigning the protocol would also violate the frozen evaluation plan. Standardized coefficients are descriptive conditional associations: correlated features prevent treating each coefficient as an isolated physical parameter, and inverse-dynamics prediction is not a causal intervention analysis.

## Hypotheses

- H1: OLS improves prediction over a training-mean DummyRegressor.
- H2: Ridge may improve or stabilize generalization relative to OLS, but need not outperform it.
- H3: Errors vary across velocity, acceleration, and torque-magnitude conditions.

These are evaluated from actual outputs and are not assumed to be true.

## Course requirement check

The official course assignment page lists DS-A3 as regression baselines and diagnostics, with evidence for interpretable baselines, split design, regularization, residuals, and error slices. It also requires runnable code, a PDF, reproduction instructions, real outputs, failure analysis, a counterexample or sensitivity check, and an AI-use record with independent verification. The assignment page and submission guide were checked on 2026-10-22: https://tjluo-ucas.github.io/ds/assignments/ and https://tjluo-ucas.github.io/ds/submission-guide/. These course-level requirements were checked independently; this notebook specification was supplied in the task brief.
