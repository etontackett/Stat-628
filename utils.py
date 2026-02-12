# Utility Functions & Classes for Model Training and Evaluation
import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class ModelMetrics:
    """Store model evaluation metrics"""
    r2: float
    adj_r2: float
    rmse: float
    mae: float
    n_params: int
    model_name: str


def plot_residuals_diagnostics(residuals, fitted_values, model_name, color='lightblue', figsize=(8, 6)):
    """
    Plot 2x2 residual diagnostics: Residuals vs Fitted, Q-Q Plot, Distribution, Scale-Location

    Args:
        residuals: Residual values
        fitted_values: Fitted/predicted values
        model_name: Name for title
        color: Color for histogram
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.4, s=8)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title(f'{model_name}: Residuals vs Fitted')
    axes[0, 0].grid(alpha=0.3)

    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title(f'{model_name}: Q-Q Plot')

    # Residuals Distribution
    axes[1, 0].hist(residuals, bins=40, color=color, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_name}: Residuals Distribution')

    # Scale-Location
    std_resid = residuals / residuals.std()
    axes[1, 1].scatter(fitted_values, np.sqrt(np.abs(std_resid)), alpha=0.4, s=8)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Sqrt(|Standardized Residuals|)')
    axes[1, 1].set_title(f'{model_name}: Scale-Location')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def _quantile_pi(y_pred, residuals, alpha=0.05):
    """Build two-sided prediction interval using empirical residual quantile."""
    q = np.quantile(np.abs(np.asarray(residuals)), 1 - alpha)
    return y_pred - q, y_pred + q


def validate_on_dev_set(model, X_dev, y_dev, model_name, color='blue', model_type='OLS', train_residuals=None):
    """
    Validate model on development set and plot actual vs predicted with 95% prediction interval coverage.

    Args:
        model: Trained model (OLS or LASSO)
        X_dev: Dev set features
        y_dev: Dev set target
        model_name: Name for title
        color: Color for scatter plot
        model_type: 'OLS' or 'LASSO'
        train_residuals: Residuals used to estimate uncertainty

    Returns:
        Tuple of (RMSE, MAE, PI_Coverage)
    """
    y_pred = model.predict(X_dev)
    y_dev_arr = np.asarray(y_dev)

    rmse = np.sqrt(np.mean((y_dev_arr - y_pred) ** 2))
    mae = np.mean(np.abs(y_dev_arr - y_pred))

    # OLS: use model-based 95% PI if available; otherwise use empirical residual quantile
    if model_type == 'OLS' and hasattr(model, 'get_prediction'):
        try:
            predictions = model.get_prediction(X_dev)
            pred_summary = predictions.summary_frame(alpha=0.05)
            pi_lower = pred_summary['obs_ci_lower'].to_numpy()
            pi_upper = pred_summary['obs_ci_upper'].to_numpy()
            pi_method = 'OLS obs_ci'
        except Exception:
            residual_source = train_residuals if train_residuals is not None else getattr(model, 'resid', y_dev_arr - y_pred)
            pi_lower, pi_upper = _quantile_pi(y_pred, residual_source, alpha=0.05)
            pi_method = 'Residual quantile fallback'
    else:
        residual_source = train_residuals if train_residuals is not None else y_dev_arr - y_pred
        pi_lower, pi_upper = _quantile_pi(y_pred, residual_source, alpha=0.05)
        pi_method = 'Residual quantile'

    coverage = np.mean((y_dev_arr >= pi_lower) & (y_dev_arr <= pi_upper))

    print(f"\n=== {model_name} Dev Set Performance ===")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"95% Prediction Interval Coverage = {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"PI method = {pi_method}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_dev_arr, y_pred, alpha=0.4, s=8, color=color)
    ax.plot([y_dev_arr.min(), y_dev_arr.max()], [y_dev_arr.min(), y_dev_arr.max()], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual PRSM (Dev)')
    ax.set_ylabel('Predicted PRSM')
    ax.set_title(f'{model_name}: Dev Actual vs Predicted\n(RMSE={rmse:.4f}, PI Coverage={coverage:.1%})')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return rmse, mae, coverage


def fit_ols_model(X_train, y_train, model_name='OLS Model'):
    """
    Fit OLS model and return model + metrics.

    Args:
        X_train: Training features (with constant already added if needed)
        y_train: Training target
        model_name: Name for reporting

    Returns:
        Tuple of (model, metrics_dict)
    """
    if 'const' not in X_train.columns:
        X_train = sm.add_constant(X_train)

    model = sm.OLS(y_train, X_train).fit()

    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    n_params = int(model.df_model)

    print(f"\n=== {model_name} Metrics ===")
    print(f"R2 = {r2:.4f}")
    print(f"Adj R2 = {adj_r2:.4f}")

    return model, {'r2': r2, 'adj_r2': adj_r2, 'n_params': n_params}


def _cv_rmse(X, y, cv_splitter):
    model = LinearRegression()
    rmse_scores = -cross_val_score(
        model,
        X,
        y,
        scoring='neg_root_mean_squared_error',
        cv=cv_splitter,
    )
    return float(np.mean(rmse_scores))


def backward_stepwise_selection(X_train, y_train, threshold_p=0.05, cv=5, random_state=42, min_improvement=1e-4):
    """
    Backward feature selection driven by shuffled CV RMSE (not p-values).

    Args:
        X_train: Training features (with or without constant)
        y_train: Training target
        threshold_p: Unused, kept for backward compatibility
        cv: Number of CV folds
        random_state: Random seed for shuffled CV
        min_improvement: Minimum RMSE improvement required to keep removing features

    Returns:
        Tuple of (selected_predictors, final_model)
    """
    use_const = 'const' in X_train.columns
    predictors = [c for c in X_train.columns if c != 'const']

    if len(predictors) == 0:
        raise ValueError('No predictors available for stepwise selection.')

    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    X_current = X_train[predictors]
    current_rmse = _cv_rmse(X_current, y_train, cv_splitter)

    print('Starting backward stepwise selection (criterion: CV RMSE)...')
    print(f'Initial predictors ({len(predictors)}): {predictors}')
    print(f'Initial CV RMSE: {current_rmse:.6f}\n')

    removed_features = []

    while len(predictors) > 1:
        candidate_results = []

        for feature in predictors:
            candidate_predictors = [p for p in predictors if p != feature]
            rmse_candidate = _cv_rmse(X_train[candidate_predictors], y_train, cv_splitter)
            candidate_results.append((feature, rmse_candidate))

        feature_to_remove, best_candidate_rmse = min(candidate_results, key=lambda x: x[1])

        if current_rmse - best_candidate_rmse > min_improvement:
            predictors.remove(feature_to_remove)
            removed_features.append((feature_to_remove, best_candidate_rmse))
            current_rmse = best_candidate_rmse
            print(f"Removing '{feature_to_remove}' -> CV RMSE: {current_rmse:.6f}")
        else:
            print('\nNo further meaningful CV RMSE improvement. Stopping.')
            break

    print(f"Final predictors ({len(predictors)}): {predictors}")
    if removed_features:
        print('Removed features:')
        for name, rmse in removed_features:
            print(f"  - {name} (best RMSE after removal: {rmse:.6f})")
    print()

    X_final = X_train[predictors].copy()
    if use_const:
        X_final = sm.add_constant(X_final, has_constant='add')

    final_model = sm.OLS(y_train, X_final).fit()
    return predictors, final_model


def fit_lasso_model(X_train, y_train, features_list, model_name='LASSO Model'):
    """
    Fit LASSO model with shuffled cross-validation for alpha selection.

    Args:
        X_train: Raw training features (will be standardized)
        y_train: Training target
        features_list: List of feature names
        model_name: Name for reporting

    Returns:
        Tuple of (lasso_model, scaler, metrics_dict)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print('Running LassoCV to find optimal alpha (shuffled CV)...')
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso_cv = LassoCV(cv=cv_splitter, random_state=42, max_iter=10000, n_alphas=100)
    lasso_cv.fit(X_train_scaled, y_train)

    optimal_alpha = float(lasso_cv.alpha_)
    print(f'Optimal alpha selected: {optimal_alpha:.6f}')

    y_pred = lasso_cv.predict(X_train_scaled)
    r2 = r2_score(y_train, y_pred)
    n = len(y_train)
    p = int(np.sum(lasso_cv.coef_ != 0))

    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan

    print(f"\n=== {model_name} Metrics ===")
    print(f"R2 = {r2:.4f}")
    print(f"Adj R2 = {adj_r2:.4f}")
    print(f"Non-zero coefficients: {p} out of {len(features_list)}")

    lasso_coefs = pd.DataFrame({'Feature': features_list, 'Coefficient': lasso_cv.coef_})
    lasso_coefs['Abs_Coef'] = np.abs(lasso_coefs['Coefficient'])
    lasso_coefs = lasso_coefs.sort_values('Abs_Coef', ascending=False)

    print(f"\n=== {model_name} Selected Features ===")
    print(lasso_coefs[lasso_coefs['Coefficient'] != 0])

    removed = lasso_coefs[lasso_coefs['Coefficient'] == 0]['Feature'].tolist()
    if removed:
        print(f"\nRemoved features (zero coefficient): {removed}")

    residuals = np.asarray(y_train) - y_pred

    return lasso_cv, scaler, {
        'r2': r2,
        'adj_r2': adj_r2,
        'n_params': p,
        'optimal_alpha': optimal_alpha,
        'coefficients': lasso_coefs,
        'train_residuals': residuals,
    }
