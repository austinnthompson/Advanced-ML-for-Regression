# Advanced-ML-for-Regression

## Executive Summary: Thompson Stucco Project

## Overview
This Jupyter Notebook (`thompson_stucco_project.ipynb`) develops a machine learning pipeline to predict the compressive strength of stucco mixtures. The project includes data loading, exploration, preprocessing, model training, hyperparameter tuning, evaluation, and final predictions on test data. It uses regression techniques to estimate the continuous target variable `strength` based on features like material composition, mixing method, and age, aiming to optimize stucco formulations for construction applications.

## Data
- **Training Data**: Loaded from `stucco.csv` (774 rows, 19 columns, including target `strength`).
  - Features: Categorical (e.g., `method`, `mixing`, `region`) and numerical (e.g., `age`, `cement`, `water`, `ash`).
  - Target: Continuous (`strength`, mean ~56.3, range 3.95–129.42 MPa).
  - No missing values; basic stats and summaries provided.
- **Test Data**: Loaded from `stucco_test.csv` (192 rows, 18 columns, no target).
- **Preprocessing**: One-hot encoding for categoricals, polynomial features (interactions, quadratic, cubic), standard scaling. Train-test split (80/20) on training data for validation.

## Methods
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn (for models, pipelines, CV, metrics).
- **Models Evaluated**:
  - Baseline: Dummy Regressor.
  - Linear: Linear Regression (base, interactions, quadratic, cubic, with/without categoricals, sequential feature selection).
  - Regularized: ElasticNetCV (linear, interactions, quadratic, cubic).
  - Non-parametric: KNN, Decision Tree.
  - Ensemble: Random Forest, Gradient Boosting.
  - Stacking: GBM + ElasticNet with Linear meta-model.
- **Evaluation**: Cross-validation (RepeatedKFold, 5 splits x 3 repeats) with metrics: negative RMSE (primary), R², negative MAE.
- **Hyperparameter Tuning**: Grid/Randomized Search for models like KNN, RF, GBM, ElasticNet; distributions include randint, loguniform, uniform.
- **Final Model**: Stacking ensemble (GBM + ElasticNet), tuned parameters (e.g., GBM: learning_rate ~0.043, max_depth 6), CV negative RMSE ~ -8.62.

## Results
- **Model Comparison** (on validation set, sorted by negative RMSE ascending, i.e., best first):
  - Best: Stacking (-8.62 RMSE, R²: 0.885, MAE: -5.43).
  - GBM: -8.67 RMSE, R²: 0.883.
  - RF: -9.34 RMSE, R²: 0.864.
  - Linear variants: -10.49 to -15.06 RMSE.
  - Worst: Dummy (-25.60 RMSE).
- **Final Evaluation** (on holdout test set):
  - RMSE: 9.66 (consistent with CV, indicating no overfitting).
- **Key Insights**: Ensembles (stacking, GBM) significantly outperform linear models; polynomial features improve fits but ensembles handle non-linearity better; all features retained in top models.

## Outputs
- **Predictions**: Saved to `thompson_stucco_predictions.csv` (column: `strength`) for the test data.
- **Visuals**: None explicitly shown, but Matplotlib imported for potential plots.
- **Logs**: Detailed metrics tables, hyperparameter summaries, predictor lists.

## Recommendations
- Deploy the stacking model for strength predictions in stucco production.
- Explore feature importance (e.g., from GBM/RF) to identify key ingredients like cement or additives for cost optimization.
- Potential Improvements: Incorporate domain-specific features (e.g., curing conditions), advanced boosting (e.g., XGBoost), or ensemble weighting; validate on external data for generalizability.

This summary is based on the notebook's content as of the last execution. For full details, run the notebook in a Python 3.13 environment with listed dependencies.

*Generated on September 12, 2025.*
