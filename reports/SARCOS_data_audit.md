# SARCOS Data Audit — DS-A3

Audit performed on the two uploaded MATLAB files. Numerical results are measured; no model has been trained.

## Input and provenance

- `sarcos_inv.mat`: matrix `sarcos_inv`, 44,484 rows × 28 columns, float64, SHA256 `b8a249733253ba6097372fedee7696833fcf30de42037d5b4a7227f21a6d1d97`.
- `sarcos_inv_test.mat`: matrix `sarcos_inv_test`, 4,449 rows × 28 columns, float64, SHA256 `161a59b5c3b4f4b404584323f181607b2acbe620eb134dc720760dc3f38f5cec`.
- Public schema identifies columns 1–21 as positions, velocities and accelerations for seven joints, and columns 22–28 as torques. Physical units and license are not verified by the .mat files alone.
- Proposed target: column 22 (Python index 21), first joint torque. Exclude *all* 7 torque columns from predictors.

## Data quality

| Check | Train | Test |
|---|---:|---:|
| Missing NaN | 0 | 0 |
| Inf | 0 | 0 |
| Constant columns | 0 | 0 |
| Duplicate full rows within file | 0 | 0 |
| Duplicate 21-dimensional input rows within file | 0 | 0 |

First torque column train: mean 13.653835, std (population) 20.464561, median 12.810330, min -94.876411, max 121.381996. Test: mean 13.692101, std 20.351353, median 13.048904, min -84.681953, max 120.974220.

## Proven leakage: original test is a regular subsample of train

For every zero-based test index `i` in `[0,4448]`, compare `test[i]` with `train[10*i]`:

- 4,446/4,449 rows are exactly equal across all 28 columns.
- All 4,449 rows match their corresponding training row with maximum absolute cell deviation at most `4.0000000041118255e-06`.
- Comparing rows without index alignment: 4,368 test rows match an entire training row bit-for-bit and 4,428 test rows match all 21 predictor values bit-for-bit. These counts are smaller than index-aligned exact matches due to tiny floating-point differences in different columns.

**Decision**: Never treat `sarcos_inv_test.mat` as an independent test set. Do not concatenate it with `sarcos_inv.mat` for modeling because it repeats the training records.

## Sequence dependence

On the original training matrix:

- Adjacent first-torque values Pearson r = 0.973613.
- Adjacent sample distance (21 inputs scaled using train-set feature population standard deviations): median 0.482590.
- Random sample pair distance, median 6.079303 (20,000 sampled pairs; seed 42).

This is consistent with sequentially ordered observations, but episode/trajectory identifiers cannot be recovered from these measurements alone.

### Exploratory near-neighbor diagnostic (sample-based, not exhaustive)

Using input standardization fit on the original training file, 500 held-out probes and 10,000 candidate training rows:

- Random 80/20 split within original training file: nearest-candidate distance median 0.7193.
- Contiguous 80/20 split within original training file: nearest-candidate distance median 3.2533.

These exploratory differences indicate evaluation sensitivity to splitting, **not** proof that the contiguous split is episode-independent or universally better. Candidates/probes were sampled separately for each comparison, so do not interpret them as matched formal hypothesis testing.

## Recommended DS-A3 protocol (to confirm before implementation)

1. Use `sarcos_inv.mat` alone (44,484 rows) for modeling; retain uploaded `sarcos_inv_test.mat` only as data-leakage evidence.
2. Baseline in-distribution tabular task: deterministic shuffled train/validation/test partition. Fit preprocessing on training partition only; use train-only CV for Ridge alpha tuning. Test once after choices are frozen.
3. Conduct leakage-aware sensitivity analysis with a contiguous/blocked holdout or a purged blocked split, and acknowledge that trajectory boundary IDs are absent. Avoid claiming this is a true unseen-episode split.
4. Main predictors: 21 state variables; target: torque 1 (column 22). Compare Dummy, OLS and Ridge; report MAE, RMSE and R²; standardize for Ridge and coefficient comparison.
5. Diagnose residuals and error by speed/acceleration regime (bins derived without peeking at test response labels); optionally perform position/velocity/acceleration feature ablation.
6. Document data origin, exact SHA256, inferred schema, unspecified physical units, and unverified redistribution license. Do not upload raw data to public repo before checking license.

## Important limitations

- `sarcos_inv.mat` contains no explicit timestamps, trajectory IDs or episode boundaries.
- Near-neighbor distances above were computed on sampled subsets and are not exact exhaustive nearest-neighbor distances.
- License / precise physical units require independent source verification.
- No models have been fit in this audit. Predictive metrics and conclusions are pending.

## Reproducibility

The accompanying `audit_sarcos.py` loads two files from `/mnt/data/` and writes `sarcos_audit_metrics.json`. Repoint `paths` to local paths to reproduce.
