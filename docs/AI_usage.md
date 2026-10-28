# AI Use Record

## ChatGPT contributions

ChatGPT was used to formulate the project review requests and specify quality criteria for the notebook, analysis, report, and documentation. Those requests guided the scope of the revisions; ChatGPT did not independently verify the scientific interpretation or dataset licensing.

## OpenAI Codex contributions

Codex assisted with project organization, drafting and refining analytical code, restructuring the Notebook into executable analytical sections, preparing the PDF export helper, and updating documentation. In earlier work, Codex added frozen-split validation, training-derived error-slice summaries, synthetic helper tests, and an optional historical-results comparison. For this final submission-quality pass, Codex improved the PDF layout and explanatory text, refined the report's interpretation and limitations, and aligned the interval heading and provenance wording.

Codex executed the Notebook and project checks in the nussem1 conda environment and compared generated outputs with the retained numerical reference. These are computational checks, not independent student verification.

## Student verification

The independent reproducibility review recorded below was performed manually by the student after the AI-assisted implementation and automated testing. Dataset license and physical units remain unresolved and were not independently verified.

## Independent Reproducibility Verification

I independently restarted the Jupyter kernel and executed all cells of DS_A3_regression_diagnostics.ipynb using the nussem1 Conda environment.

The Notebook completed without execution errors. I checked the frozen data splits, model evaluation metrics, selected Ridge hyperparameters, and diagnostic figures against the previously reported results.

The key numerical results were consistent with the earlier execution, and the conclusions remained supported by the displayed evidence.

This verification was performed manually after the AI-assisted implementation and automated testing.

## Time-critical assignment extension

The user supplied the detailed assignment brief requiring influence diagnostics, heteroscedasticity assessment, learning curves, critical error cases, complexity comparison, Lasso, a nonlinear candidate, an environment lock, and a short submission summary. Codex used that brief to identify the gaps and implement the requested additions while preserving the frozen splits and original Dummy/OLS/Ridge results.

Accepted recommendations: compute training-only OLS leverage and Cook's distance without constructing a full projection matrix; add a training-residual Breusch–Pagan auxiliary regression with serial-dependence caveats; add a seeded Blocked Train/Validation learning curve; report post-hoc Test error cases without using them for selection; and compare Lasso and HistGradientBoostingRegressor using Train/Validation only. The absolute-target slice is explicitly described as retrospective and outcome-conditioned, and standardized coefficients are distinguished from physical or causal parameters.

Recommendations not undertaken: no blocked cross-validation or extensive tuning redesign was added because the frozen Blocked Validation segment is a distinct low-motion region and cannot represent all blocked regions. No new candidate was evaluated or selected on Test, no original Test result was optimized, and no new model was used to revise the original conclusions.

The earlier student verification above applies to the original submitted Notebook version only. No independent student verification of these new AI-assisted extensions is claimed. Codex executed the revised Notebook from a fresh kernel and ran the project validator and helper tests in `nussem1`; these automated checks are not student verification.

## Research narrative revision

For the final narrative revision, ChatGPT/Codex used the assignment criteria and existing audited project artifacts to clarify why inverse-dynamics torque prediction is a meaningful regression task, add a compact SARCOS Dataset Card to the Notebook and PDF, and explain the existing primary and exploratory results, diagnostics, and limits. The Notebook change is Markdown-only; no model, split, analysis code, hyperparameter, or result was changed. The PDF and README were aligned with that narrative. Codex preserved the student's manually revised `docs/submission_summary.md` exactly as received. This documentation assistance does not constitute independent student verification of the exploratory analysis; the student should review the final wording and claims.

## Automated checks for the time-critical extension

- `conda run -n nussem1 jupyter nbconvert --to notebook --execute notebooks/DS_A3_regression_diagnostics.ipynb --output DS_A3_regression_diagnostics.executed.ipynb --output-dir /tmp --ExecutePreprocessor.timeout=600` — completed successfully from a clean kernel.
- `conda run -n nussem1 python scripts/validate_project.py` — passed frozen split, hashes, and retained-result consistency checks.
- `conda run -n nussem1 python -m unittest discover -s tests -v` — run; see the test command output associated with this revision.

Codex-generated extensions and report artifacts were checked computationally against the specified protocols. A student should independently review the new methods and interpretations before submission; this record does not claim that such a review has occurred.
