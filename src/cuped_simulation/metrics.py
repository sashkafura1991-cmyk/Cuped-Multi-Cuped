import numpy as np
import pandas as pd
from scipy import stats

def is_binary_metric(df, target_col):
    """Определяет, является ли метрика бинарной (0 и 1)."""
    return df[target_col].nunique() == 2

def estimate_mde(n_samples, var_metric, mean_metric, alpha=0.05, power=0.8):
    """Рассчитывает относительный MDE в процентах."""
    if var_metric == 0 or mean_metric == 0 or n_samples == 0:
        return 0.0

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)


    mde_abs = np.sqrt((2 * var_metric * (z_alpha + z_beta)**2) / n_samples)
    mde_rel = (mde_abs / abs(mean_metric)) * 100
    return mde_rel

def preprocess_data(df):
    """Базовая очистка данных."""
    df_clean = df.copy()
    df_clean = df_clean.replace(-1, 0).fillna(0)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    return df_clean[numeric_cols]
