from cuped_simulation.metrics import estimate_mde, is_binary_metric
import pandas as pd

def test_estimate_mde():
    mde = estimate_mde(n_samples=10000, var_metric=150, mean_metric=50)
    assert mde > 0
    assert isinstance(mde, float)

def test_is_binary_metric():
    df = pd.DataFrame({'binary_col': [0, 1, 0, 1, 1], 'cont_col': [1.5, 2.3, 5.1, 0.4, 2.2]})
    assert is_binary_metric(df, 'binary_col') == True
    assert is_binary_metric(df, 'cont_col') == False
