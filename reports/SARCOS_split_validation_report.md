# SARCOS DS-A3 — Split Validation and Frozen Protocol v1.0

## Status
Executed on uploaded `sarcos_inv.mat` (44,484 rows, 28 numeric columns). This is a *pre-modeling* data/split audit; no regression results have been computed. `sarcos_inv_test.mat` is excluded from regression due to verified near-complete overlap with the training file.

## Actual split diagnostics (before purge)

| Split | Part | N | Torque mean | Torque SD | Median velocity norm | Median acceleration norm |
|---|---|---:|---:|---:|---:|---:|
| Blocked | Train | 26,690 | 13.359 | 22.445 | 2.302 | 21.955 |
| Blocked | Validation | 8,897 | 15.555 | 18.779 | 0.367 | 5.464 |
| Blocked | Test | 8,897 | 12.637 | 15.006 | 2.225 | 24.695 |
| Random | Train | 26,690 | 13.586 | 20.564 | 2.034 | 19.647 |
| Random | Validation | 8,897 | 13.972 | 20.334 | 2.027 | 19.419 |
| Random | Test | 8,897 | 13.539 | 20.293 | 2.012 | 19.487 |

KS distance vs respective training-target distribution: blocked val 0.1142, blocked test 0.1136; random val 0.0144, random test 0.0061.

## Nearest-neighbor diagnostics

21 features were standardized with statistics of the unpurged **blocked training** split solely for comparing split geometries. For each validation and test portion, 900 queries were sampled at random (fixed RNG=42), and exact Euclidean one-nearest-neighbor distances were computed against the *entire* respective train portion.

| Split | Validation NN distance median | Test NN distance median |
|---|---:|---:|
| Blocked | 1.039 | 1.600 |
| Random | 0.427 | 0.447 |

This shows marked difference in proximity, but does **not** prove absence or presence of cross-episode overlap. It is not regression performance.

Adjacent target Pearson correlation is 0.9736; lag 2: 0.9202; lag 5: 0.6505; lag 10: 0.2792; lag 20: -0.0543; lag 50: 0.3533. Thus dependence is nonmonotonic and lag=20 alone should not define an independence guarantee.

A supplementary sampled purge test at 0,10,25,50,100,250 rows showed no monotonic, material improvement in global nearest-neighbor distance. A gap is a local safeguard only, not a solution to trajectory dependence.

## Frozen evaluation splits for modeling

Source: only `sarcos_inv.mat`, 0-based row indices, 21 input columns 0..20, target column 21 (first joint torque).

**Primary (blocked, chronological file order)**: cutpoints `floor(0.6*n)=26690`, `floor(0.8*n)=35587`; purge 25 records on both sides of each split boundary:
- Train: rows `[0, 26665)` => 26,665
- Validation: rows `[26715, 35562)` => 8,847
- Test: rows `[35612, 44484)` => 8,872
- Omitted: `[26665,26715)` and `[35562,35612)`, total 100 records.

**Sensitivity (random)**: `np.random.default_rng(42).permutation(n)`; first 26,690 rows Train, next 8,897 Validation, last 8,897 Test. Random partition is not presented as leak-free trajectory generalization.

Saved exact split arrays in `sarcos_split_indices.npz`, and source hash/metadata in `sarcos_split_manifest.json`.

## Precommitted modeling policy

- Main metric MAE; secondary RMSE and R². Dummy mean, OLS (with feature scaler), Ridge (feature scaler and train/validation alpha tuning) compared on primary blocked protocol.
- Choose models/alpha using training and validation **only**; do not decide changes based on test metrics.
- Fit preprocessing only to training, separately per protocol. Do not reuse the blocked audit scaler for model training or for random protocol.
- Primary final held-out test is blocked Test; report random-split test solely as sensitivity, not as a replacement when its score is more favorable.
- Error slices: norm of joint velocities, norm of accelerations, magnitude of first-joint torque. Precompute bin thresholds from training only, before examining held-out errors.
- Treat blocked validation as a possible low-motion operating regime and report its marked distribution shift; blocked test is a held-out consecutive data segment but **not** verified independent trajectory or new-task evaluation.

## Caveats

- The data have no verified episode/trial identifiers. Original row order likely exhibits serial dependence, but cannot establish trajectory boundaries.
- Blocked split and local 25-row purge cannot guarantee independence. Gap width is a pragmatic protocol choice, not statistically established decorrelation.
- This diagnostic inspected feature and target distributions of held-out portions before model fitting. No model errors have been inspected. Once the indices are frozen, holdout performance should not be used to iterate on the split.
- Physical torque units and data license still require original source confirmation.

## Reproduce

With NumPy, SciPy, scikit-learn installed and `sarcos_inv.mat` in the same directory:

```bash
python validate_sarcos_split.py
python purge_study.py
python freeze_sarcos_splits.py
```

All code and artifacts in the linked files are derived from actual execution.
