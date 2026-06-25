import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=5000, effect_size=2.0):
    """
    Генерирует синтетический датасет для А/В тестов с заданными ковариатами.
    Идеально подходит для симуляций Монте-Карло.
    """
    np.random.seed(42)


    cov1 = np.random.normal(10, 2, n_samples)
    cov2 = np.random.normal(50, 10, n_samples)


    groups = np.random.choice(['A', 'B'], size=n_samples)


    treatment_effect = np.where(groups == 'B', effect_size, 0)
    noise = np.random.normal(0, 5, n_samples)

    target = 5 * cov1 + 0.5 * cov2 + treatment_effect + noise

    df = pd.DataFrame({
        'user_id': range(n_samples),
        'group': groups,
        'covariate_1': cov1,
        'covariate_2': cov2,
        'target_metric': target
    })

    return df
