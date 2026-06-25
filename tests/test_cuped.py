# tests/test_cuped.py
import pandas as pd
import numpy as np
from cuped_simulation.cuped import apply_single_cuped

def test_apply_single_cuped():
    df = pd.DataFrame({
        'target': [100, 120, 110, 130, 105],
        'covariate': [10, 12, 11, 13, 10]
    })

    y_cuped, best_cov = apply_single_cuped(df, 'target', ['covariate'])

    assert best_cov == 'covariate'
    assert len(y_cuped) == 5
    assert isinstance(y_cuped, np.ndarray)
