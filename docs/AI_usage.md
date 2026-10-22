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
