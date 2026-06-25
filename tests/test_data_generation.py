import pandas as pd
from cuped_simulation.data_generation import generate_synthetic_data

def test_generate_synthetic_data_shape():
    """Проверяем, что генерируется правильное количество строк и столбцов."""
    n = 1000
    df = generate_synthetic_data(n_samples=n)

    assert len(df) == n, f"Ожидалось {n} строк, получено {len(df)}"
    assert 'group' in df.columns
    assert 'target_metric' in df.columns
    assert 'covariate_1' in df.columns

def test_generate_synthetic_data_groups():
    """Проверяем, что пользователи разбиты на две группы (А и B)."""
    df = generate_synthetic_data(n_samples=500)

    groups = set(df['group'].unique())
    assert groups == {'A', 'B'}, "Должны присутствовать только группы 'A' и 'B'"

def test_generate_synthetic_data_no_nulls():
    """Проверяем, что в сгенерированных данных нет пропусков."""
    df = generate_synthetic_data(n_samples=100)

    assert df.isnull().sum().sum() == 0, "Синтетические данные не должны содержать NaN"
