import numpy as np
import pandas as pd
import os

from .metrics import estimate_mde, preprocess_data
from .cuped import apply_single_cuped
from .multi_cuped import apply_vanilla_multi_cuped, apply_ml_cuped
from .plots import plot_cuped_variance_reduction

def run_full_segmented_analysis(data_path, targets, cov_cols=None, output_dir='cuped_output_plots'):
    """Главный пайплайн запуска всех методов."""
    print("Загрузка и очистка датасета...")
    df_raw = pd.read_csv(data_path)
    df = preprocess_data(df_raw)


    if cov_cols is None:
        cov_cols = [c for c in df.columns if c not in targets]

    print(f"Используется {len(cov_cols)} предэкспериментальных признаков.")

    for target in targets:
        print(f"\n{'='*50}")
        print(f"АНАЛИЗ МЕТРИКИ: {target.upper()}")
        print(f"{'='*50}")

        y_orig = df[target].values
        var_orig = np.var(y_orig)
        mean_orig = np.mean(y_orig)
        n_samples = len(y_orig)

        if var_orig == 0:
            print("Нулевая дисперсия. Пропуск.")
            continue

        mde_orig = estimate_mde(n_samples, var_orig, mean_orig)

        y_single, best_cov = apply_single_cuped(df, target, cov_cols)
        var_single = np.var(y_single)
        red_single = (1 - var_single / var_orig) * 100

        y_multi = apply_vanilla_multi_cuped(df, target, cov_cols)
        var_multi = np.var(y_multi)
        red_multi = (1 - var_multi / var_orig) * 100

        y_ml = apply_ml_cuped(df, target, cov_cols)
        var_ml = np.var(y_ml)
        red_ml = (1 - var_ml / var_orig) * 100

        series_dict = {
            'Single-CUPED': y_single,
            'Multi-CUPED': y_multi,
            'ML-CUPAC': y_ml
        }
        plot_cuped_variance_reduction(df, target, series_dict, output_dir)

        # Отчет
        print(f"  • Исходные параметры  : var = {var_orig:.4f} | MDE = {mde_orig:.2f}%")
        print(f"  • Single CUPED (-{red_single:.2f}%): лучшая ковариата = {best_cov}")
        print(f"  • Vanilla Multi-CUPED (-{red_multi:.2f}%): линейная регрессия")
        print(f"  • ML-CUPED (-{red_ml:.2f}%): CatBoost OOF")
        print(f"  • Ускорение теста: в {var_orig / var_ml:.2f} раз")

    print(f"\n✅ Все графики сохранены в папку: {os.path.abspath(output_dir)}")
