import numpy as np
import pandas as pd

def apply_single_cuped(df, target_col, cov_cols):
    """Применяет классический Single-CUPED, выбирая лучшую ковариату по корреляции."""
    correlations = {}
    y = df[target_col].values

    for cov in cov_cols:
        x = df[cov].values
        if np.std(x) > 0 and np.std(y) > 0:
            corr = np.corrcoef(x, y)[0, 1]
            correlations[cov] = abs(corr)

    if not correlations:
        return df[target_col], "None"

    best_cov = max(correlations, key=correlations.get)
    x_best = df[best_cov].values

    cov_matrix = np.cov(x_best, y)
    theta = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] > 0 else 0

    y_cuped = y - theta * (x_best - np.mean(x_best))

    return y_cuped, best_cov
