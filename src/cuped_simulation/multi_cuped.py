import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold


try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from .metrics import is_binary_metric

def get_cv_splits(df, target_col, n_splits=5):
    """Умное разбиение на фолды."""
    y = df[target_col].values
    if is_binary_metric(df, target_col):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(df, y))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(kf.split(df))

def apply_vanilla_multi_cuped(df, target_col, cov_cols, n_splits=5):
    """Линейный Multi-CUPED на базе LinearRegression."""
    y = df[target_col].values
    y_pred_oof = np.zeros(len(y))
    cv_splits = get_cv_splits(df, target_col, n_splits)

    for train_idx, val_idx in cv_splits:
        X_train, y_train = df.iloc[train_idx][cov_cols], y[train_idx]
        X_val = df.iloc[val_idx][cov_cols]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_oof[val_idx] = model.predict(X_val)

    var_pred = np.var(y_pred_oof)
    theta = np.cov(y, y_pred_oof)[0, 1] / var_pred if var_pred > 0 else 0
    y_cuped = y - theta * (y_pred_oof - np.mean(y_pred_oof))

    return y_cuped

def apply_ml_cuped(df, target_col, cov_cols, n_splits=5):
    """Продвинутый ML-CUPAC на базе CatBoost (Градиентный бустинг)."""
    if not CATBOOST_AVAILABLE:
        print("Внимание: CatBoost не установлен. Возвращаем оригинальную метрику.")
        return df[target_col].values

    y = df[target_col].values
    is_binary = is_binary_metric(df, target_col)
    y_pred_oof = np.zeros(len(y))
    cv_splits = get_cv_splits(df, target_col, n_splits)

    for train_idx, val_idx in cv_splits:
        X_train, y_train = df.iloc[train_idx][cov_cols], y[train_idx]
        X_val = df.iloc[val_idx][cov_cols]

        if is_binary:
            model = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)
            model.fit(X_train, y_train)
            y_pred_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            model = CatBoostRegressor(iterations=100, verbose=0, random_seed=42)
            model.fit(X_train, y_train)
            y_pred_oof[val_idx] = model.predict(X_val)

    var_pred = np.var(y_pred_oof)
    theta = np.cov(y, y_pred_oof)[0, 1] / var_pred if var_pred > 0 else 0
    y_cupac = y - theta * (y_pred_oof - np.mean(y_pred_oof))

    return y_cupac
