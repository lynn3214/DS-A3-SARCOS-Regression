# Submission Summary

I chose robot inverse dynamics because of my interest in robot learning and control. My research question was whether interpretable regression models could predict joint torque from a robot's kinematic state, and under which operating conditions their predictions become less reliable.

I selected the SARCOS dataset because it provides measurements of joint positions, velocities, accelerations, and torques for a 7-DoF robot arm. These variables offer a physically meaningful setting for regression. I used 21 kinematic features to predict the first joint's torque. During the data audit, I identified substantial overlap between the original training and test files, so I adopted frozen blocked and random evaluation protocols while acknowledging their limitations.

I compared Dummy, OLS, and Ridge regression, then examined residuals, influence, heteroscedasticity, learning curves, and error slices. I also explored Lasso and a nonlinear HistGradientBoosting model to investigate the trade-off between predictive performance and model complexity.

On the blocked test set, OLS reduced MAE from 11.631 to 3.865, while Ridge provided negligible improvement. Error analysis showed higher absolute errors in high-acceleration and high-torque regions, but velocity-related errors were non-monotonic. These patterns suggest that model reliability varies across operating conditions.

The results partially answer my research question: interpretable linear models provide a useful baseline for SARCOS inverse dynamics, but the evaluation does not establish generalization to independent trajectories or new robots. Temporal dependence, distribution shift, and unavailable episode identifiers limit stronger claims. I independently reran the Notebook and checked the reported results rather than relying solely on AI-generated analysis.
