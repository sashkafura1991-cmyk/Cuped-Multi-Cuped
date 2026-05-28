import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import shutil
from collections import Counter
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

# ---------- Импорт опциональных библиотек с заглушками ----------
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    class CatBoostRegressor:
        def __init__(self, **kwargs): pass
        def fit(self, *args, **kwargs): return self
        def predict(self, *args, **kwargs): return np.array([])
    class CatBoostClassifier:
        def __init__(self, **kwargs): pass
        def fit(self, *args, **kwargs): return self
        def predict_proba(self, *args, **kwargs): return np.array([])

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    class shap:
        class TreeExplainer:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return None
        def summary_plot(*args, **kwargs): pass

warnings.filterwarnings('ignore')

# =============================================================================
# НАСТРОЙКА ВЫВОДА ГРАФИКОВ В ФАЙЛЫ
# =============================================================================
OUTPUT_DIR = "cuped_output_plots"

def setup_output_dir():
    """Создаёт чистую папку для графиков (удаляет старую)."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Графики будут сохранены в папку: {OUTPUT_DIR}")

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ (ОЧИСТКА, ВСПОМОГАТЕЛЬНЫЕ)
# =============================================================================
def preprocess_data(df):
    """Очистка данных и обработка пропусков."""
    df_clean = df.copy().replace(-1, 0)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean[numeric_cols].fillna(0)
    return df_clean

def is_binary_metric(df, target_col):
    """Определяет, является ли метрика бинарной (конверсией)."""
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) <= 2:
        return True
    return (df[target_col].value_counts(normalize=True).get(0, 0) +
            df[target_col].value_counts(normalize=True).get(1, 0)) > 0.95

def create_ml_covariate(df, target_col, feature_cols, cv_folds=5):
    """Генерация ML-ковариаты через GradientBoosting (устаревший метод, не используется)."""
    # Оставлен для совместимости, но в основном пайплайне не вызывается
    pass

def find_optimal_weights(X, y, alphas=None):
    """Находит стабильные веса Ridge для Multi-CUPED."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    best_score, best_alpha = -np.inf, alphas[0]

    for alpha in alphas:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_i, va_i in cv.split(X_sc):
            r = Ridge(alpha=alpha).fit(X_sc[tr_i], y.iloc[tr_i])
            scores.append(r.score(X_sc[va_i], y.iloc[va_i]))
        if np.mean(scores) > best_score:
            best_score, best_alpha = np.mean(scores), alpha

    model = Ridge(alpha=best_alpha).fit(X_sc, y)
    return model.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_), best_alpha

def estimate_mde(n_samples, var_metric, mean_metric):
    """Расчет относительного MDE (%) по Стьюденту (alpha=5%, power=80%)."""
    n_group = n_samples / 2
    if n_group <= 1 or var_metric <= 0 or mean_metric == 0:
        return 0.0
    t_sum = stats.t.ppf(0.975, int(n_group - 1)) + stats.t.ppf(0.80, int(n_group - 1))
    return (t_sum * np.sqrt((2 * var_metric) / n_group) / mean_metric) * 100

def get_multi_cuped_score(df, target_col, top_covariates, cv_folds=5):
    """Multi-CUPED (линейная комбинация) с защитой от Data Leakage."""
    if not top_covariates:
        return 0.0, [], 0.0
    X = df[top_covariates].dropna()
    y = df[target_col].loc[X.index]

    if len(X) < cv_folds * 2:
        return 0.0, [], 0.0

    global_weights, best_alpha = find_optimal_weights(X, y)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_cuped_oof = np.zeros(len(y))
    thetas = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        local_ridge = Ridge(alpha=best_alpha).fit(X_train_scaled, y_train)
        local_weights = local_ridge.coef_ / scaler.scale_

        cov_train = np.dot(X_train, local_weights)
        cov_val = np.dot(X_val, local_weights)

        cov_mean_train = np.mean(cov_train)

        cov_matrix = np.cov(y_train, cov_train, ddof=1)
        theta_fold = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        thetas.append(theta_fold)

        y_cuped_oof[val_idx] = y.iloc[val_idx] - theta_fold * (cov_val - cov_mean_train)

    y_cuped_oof = y_cuped_oof - y_cuped_oof.mean() + y.mean()
    reduction = max(0.0, (1 - np.var(y_cuped_oof, ddof=1) / y.var(ddof=1)) * 100)
    return reduction, global_weights, np.mean(thetas)

# =============================================================================
# ИСПРАВЛЕННЫЕ ФУНКЦИИ ДЛЯ ML И VANILLA БЕЗ УТЕЧЕК
# =============================================================================
def get_oof_predictions_catboost(X, y, n_splits=5):
    """
    Out-of-fold предсказания CatBoost.
    X, y – pandas объекты (X – только разрешённые предпериодные признаки).
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    is_binary = (y.nunique() == 2)
    print(f"Тип таргета: {'бинарный' if is_binary else 'непрерывный'}")

    if is_binary:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_hat_oof = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y if is_binary else None)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if is_binary:
            model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                       random_seed=42, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
            y_hat_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)
            model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                      random_seed=42, verbose=False)
            model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), early_stopping_rounds=50)
            y_hat_oof[val_idx] = np.expm1(model.predict(X_val))
        print(f"  Fold {fold+1}/{n_splits} готов (деревьев: {model.tree_count_})")
    return y.values, y_hat_oof, is_binary

def apply_ml_cuped(y_real, y_hat, target_name="метрика"):
    """Один theta на основе предсказаний ML-модели."""
    corr = np.corrcoef(y_real, y_hat)[0,1]
    cov = np.cov(y_real, y_hat, ddof=1)[0,1]
    var_hat = np.var(y_hat, ddof=1)
    theta = cov / var_hat if var_hat > 0 else 0
    y_cuped = y_real - theta * (y_hat - np.mean(y_hat))
    var_before = np.var(y_real, ddof=1)
    var_after = np.var(y_cuped, ddof=1)
    reduction = (1 - var_after/var_before)*100 if var_before>0 else 0
    print(f"\nML-CUPED ({target_name}):")
    print(f"  Corr(Y, Y_hat) = {corr:.4f}  (потенциал {corr**2*100:.1f}%)")
    print(f"  Theta = {theta:.4f}")
    print(f"  Дисперсия до: {var_before:.4f}, после: {var_after:.4f}, снижение: {reduction:.2f}%")
    return y_cuped, reduction

def apply_vanilla_multi_cuped(df, target_col, pre_period_cols, top_k=10, n_splits=5):
    """
    Vanilla Multi-CUPED (OLS) с ограничением на предпериодные признаки.
    pre_period_cols – список колонок, доступных до эксперимента.
    """
    df_work = df.copy()
    y_real = df_work[target_col].values
    y_hat_oof = np.zeros(len(df_work))
    best_features_per_fold = []

    # Используем только предпериодные признаки
    available_covs = [c for c in pre_period_cols if c in df_work.columns]
    if len(available_covs) == 0:
        print("Нет предпериодных признаков для Vanilla Multi-CUPED.")
        return y_real, 0.0, 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Обучение OLS по фолдам (только предпериодные признаки)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_work)):
        df_train = df_work.iloc[train_idx]
        df_val = df_work.iloc[val_idx]
        y_train = y_real[train_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            correlations = df_train[available_covs].corrwith(pd.Series(y_train, index=df_train.index)).abs().dropna()

        best_features = correlations.nlargest(top_k).index.tolist()
        best_features_per_fold.extend(best_features)

        if not best_features:
            y_hat_oof[val_idx] = np.mean(y_train)
            continue

        model = LinearRegression()
        model.fit(df_train[best_features], y_train)
        y_hat_oof[val_idx] = model.predict(df_val[best_features])

        print(f"  Фолд {fold+1}/{n_splits} завершен.")

    var_y_hat = np.var(y_hat_oof, ddof=1)
    if var_y_hat > 0:
        theta = np.cov(y_real, y_hat_oof, ddof=1)[0, 1] / var_y_hat
    else:
        theta = 0

    y_cuped = y_real - theta * (y_hat_oof - np.mean(y_hat_oof))

    var_before = np.var(y_real, ddof=1)
    var_after = np.var(y_cuped, ddof=1)
    reduction = (1 - var_after / var_before) * 100 if var_before > 0 else 0

    top_overall = [f[0] for f in Counter(best_features_per_fold).most_common(top_k)]
    print("\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ VANILLA MULTI-CUPED: {target_col}")
    print("="*50)
    print(f"Самые стабильные Топ-{top_k} ковариат: \n{top_overall[:5]}...")
    print(f"Оптимальная Theta (θ): {theta:.4f}")
    print(f"Общая дисперсия ДО:    {var_before:.4f}")
    print(f"Общая дисперсия ПОСЛЕ: {var_after:.4f}")
    print(f"ОБЩЕЕ СНИЖЕНИЕ VAR:    {reduction:.2f}%")
    print("="*50)

    return y_cuped, theta, reduction

# =============================================================================
# ВИЗУАЛИЗАЦИИ (СОХРАНЕНИЕ В ФАЙЛЫ)
# =============================================================================
def visualize_stratification_tree(df, target_col, seg_name, pre_period_cols, max_depth=4, min_samples_leaf=2000):
    """Визуализация дерева стратификации – только предпериодные признаки."""
    print(f"\n{'='*50}")
    print(f"ВИЗУАЛИЗАЦИЯ СТРАТ ДЛЯ: {target_col} ({seg_name})")
    print(f"{'='*50}")

    feature_cols = [c for c in pre_period_cols if c in df.columns and df[c].dtype in [np.number]]
    if len(feature_cols) == 0:
        print("Нет предпериодных числовых признаков для построения дерева.")
        return

    X = df[feature_cols]
    y = df[target_col]

    if len(X) < min_samples_leaf:
        min_samples_leaf = max(1, len(X) // 10)
        print(f"Размер выборки {len(X)} меньше исходного min_samples_leaf, уменьшено до {min_samples_leaf}")

    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree.fit(X, y)

    print("\nПравила разбиения (как алгоритм делит клиентов):")
    print(export_text(tree, feature_names=feature_cols))

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True,
        precision=3
    )
    plt.title(f'Умное разбиение на страты (Target: {target_col}, {seg_name})\nmax_depth={max_depth}, min_samples_leaf={min_samples_leaf}',
              fontsize=16, pad=20)
    filename = f"{OUTPUT_DIR}/strat_tree_{target_col}_{seg_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранено: {filename}")

def plot_cuped_variance_reduction(y_real, y_cuped, target_name, seg_name, covariate_name="OOF-предсказания ML"):
    """Визуализация снижения дисперсии – сохраняет в файл."""
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_font = {'color': 'navy', 'fontweight': 'bold', 'fontsize': 12}

    sns.kdeplot(y_real, fill=True, color='#ff9999', alpha=0.4, label='Original', ax=axes[0])
    sns.kdeplot(y_cuped, fill=True, color='#66b3ff', alpha=0.5, label='CUPED', ax=axes[0])
    axes[0].set_title(f'Плотность: {target_name}\n[Ковариата: {covariate_name}]', fontdict=title_font)
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Плотность')
    axes[0].legend(loc='upper right')

    df_box = pd.DataFrame({'Original': y_real, 'CUPED': y_cuped})
    sns.boxplot(data=df_box, ax=axes[1], palette=['#ffffff', '#80bfff'], width=0.6, fliersize=5, linewidth=1.5)
    axes[1].set_title(f'Изменение разброса (Boxplot)\n[Ковариата: {covariate_name}]', fontdict=title_font)
    axes[1].set_ylabel('')
    plt.tight_layout()
    safe_cov = covariate_name.replace(' ', '_').replace('(', '').replace(')', '')[:30]
    filename = f"{OUTPUT_DIR}/cuped_{target_name}_{seg_name}_{safe_cov}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранено: {filename}")

def plot_sanity_checks(y_real, y_hat, target_name, seg_name):
    """Scatter plot и калибровка – сохраняет в файл."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_hat, y_real, alpha=0.1, color='#2c3e50', s=15)
    axes[0].set_title(f'Облако предсказаний ({target_name})', fontsize=12)
    axes[0].set_xlabel('Предсказание модели (Y_hat)')
    axes[0].set_ylabel('Фактическое значение (Y)')

    df_cal = pd.DataFrame({'y_real': y_real, 'y_hat': y_hat})
    df_cal['decile'] = pd.qcut(df_cal['y_hat'].rank(method='first'), q=10, labels=False)
    cal_means = df_cal.groupby('decile')['y_real'].mean()

    axes[1].plot(cal_means.index, cal_means.values, marker='o', linestyle='-', color='#e74c3c', linewidth=2)
    axes[1].set_title('Калибровка по децилям (Монотонность)', fontsize=12)
    axes[1].set_xlabel('Децили предсказаний (0 - низкие, 9 - высокие)')
    axes[1].set_ylabel('Средний факт (Y)')
    axes[1].set_xticks(range(10))
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/sanity_{target_name}_{seg_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранено: {filename}")

def run_shap_analysis_full(X, y, is_binary, target_col, seg_name):
    """SHAP-интерпретация – только предпериодные признаки."""
    if not (_CATBOOST_AVAILABLE and _SHAP_AVAILABLE):
        print("SHAP или CatBoost не установлены. Пропуск SHAP-анализа.")
        return

    print("\n" + "★"*60)
    print(f" ИНТЕРПРЕТАЦИЯ SHAP ДЛЯ ПРЕЗЕНТАЦИИ: {target_col} ({seg_name})")
    print("★"*60)

    if is_binary:
        model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=False)
        model.fit(X, y)
    else:
        model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=False)
        model.fit(X, np.log1p(y))

    print("Вычисление SHAP значений...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    vals = np.abs(shap_values.values).mean(0)
    if len(vals.shape) > 1:
        vals = vals[:, 1] if vals.shape[1] > 1 else vals[:, 0]

    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
    top_10 = feature_importance.head(10)

    print("\nТоп-10 признаков по SHAP:")
    print(top_10.to_string(index=False))

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'Summary Plot: Топ фичей\nТаргет: {target_col}', fontsize=16, pad=20, fontweight='bold', color='navy')
    plt.tight_layout()
    filename_summary = f"{OUTPUT_DIR}/shap_summary_{target_col}_{seg_name}.png"
    plt.savefig(filename_summary, dpi=150)
    plt.close()
    print(f"Сохранено: {filename_summary}")

    # Dependence Plot
    continuous_features = [c for c in X.columns if X[c].nunique() > 15]
    if continuous_features:
        top_cont_feature = next((f for f in top_10['col_name'] if f in continuous_features), continuous_features[0])
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(shap_values[:, top_cont_feature], color=shap_values, show=False)
        plt.gca().set_title(f'Нелинейность признака: {top_cont_feature}\nДЛЯ ЦЕЛЕВОЙ МЕТРИКИ: {target_col}',
                            fontsize=14, pad=20, fontweight='bold', color='darkred')
        plt.gca().set_ylabel(f'Влияние на {target_col} (SHAP value)', fontsize=12)
        plt.tight_layout()
        filename_dep = f"{OUTPUT_DIR}/shap_dependence_{target_col}_{seg_name}_{top_cont_feature}.png"
        plt.savefig(filename_dep, dpi=150)
        plt.close()
        print(f"Сохранено: {filename_dep}")
    else:
        print("\n[ИНФО] Нет непрерывных признаков для Dependence Plot.")

# =============================================================================
# A/B-СИМУЛЯЦИИ ДЛЯ ПРОВЕРКИ ЧЕСТНОСТИ CUPED
# =============================================================================
def generate_synthetic_data(n_samples, effect=0.0, rho=0.7, metric_type='normal'):
    """
    Генерация синтетических данных для A/B-теста.
    Возвращает target, ковариату, группу (0/1).
    Ковариата – предпериодный признак (коррелирует с базовым значением target).
    """
    group = np.random.binomial(1, 0.5, n_samples)  # 0 контроль, 1 лечение
    if metric_type == 'binary':
        base_p = 0.1
        p_treat = np.clip(base_p + effect, 0, 1)
        target = np.where(group == 0,
                          np.random.binomial(1, base_p, n_samples),
                          np.random.binomial(1, p_treat, n_samples))
        # Ковариата – слабо коррелирует с таргетом (можно добавить корреляцию, но для простоты шум)
        cov = np.random.normal(0, 1, n_samples)
        return target, cov, group
    else:
        target_base = np.random.normal(100, 15, n_samples)
        target = target_base + effect * group
        # Ковариата генерируется из target_base (без эффекта!) с заданной корреляцией
        noise = np.random.normal(0, 15, n_samples)
        cov = rho * target_base + np.sqrt(1 - rho**2) * noise
        return target, cov, group

def run_power_analysis(method='none', rho=0.7, n_simulations=500, sample_sizes=[500, 1000, 2000, 4000], mde=2.0):
    """Оценивает мощность (1-beta) для заданного метода."""
    from scipy.stats import ttest_ind
    power_results = []
    for n in sample_sizes:
        pvals = []
        for _ in range(n_simulations):
            y, cov, group = generate_synthetic_data(n, effect=mde, rho=rho, metric_type='normal')
            if method == 'single':
                theta = np.cov(y, cov, ddof=1)[0,1] / np.var(cov, ddof=1) if np.var(cov, ddof=1) > 0 else 0
                y_cuped = y - theta * (cov - np.mean(cov))
                y_control = y_cuped[group == 0]
                y_treatment = y_cuped[group == 1]
            else:
                y_control = y[group == 0]
                y_treatment = y[group == 1]
            _, p = ttest_ind(y_control, y_treatment, equal_var=False)
            pvals.append(p)
        power = np.mean(np.array(pvals) < 0.05)
        power_results.append(power)
    return sample_sizes, power_results

def run_ab_validation():
    """Запускает A/B-симуляции и выводит сравнение мощности."""
    print("\n" + "="*70)
    print("A/B-СИМУЛЯЦИИ ДЛЯ ПРОВЕРКИ ЧЕСТНОСТИ CUPED")
    print("="*70)
    sizes = [500, 1000, 2000, 4000]
    mde = 2.0
    _, power_none = run_power_analysis('none', rho=0.7, sample_sizes=sizes, mde=mde)
    _, power_cuped = run_power_analysis('single', rho=0.7, sample_sizes=sizes, mde=mde)
    print("Мощность (вероятность обнаружить эффект) при alpha=0.05:")
    print("Размер выборки | Без CUPED | C CUPED (rho=0.7)")
    for n, pn, pc in zip(sizes, power_none, power_cuped):
        print(f"{n:12d} | {pn:8.2f} | {pc:8.2f}")
    print("\nОжидается, что CUPED увеличивает мощность (или снижает требуемый размер выборки).")
    print("Если мощность с CUPED значительно выше, чем без него, то снижение дисперсии реально.")
    print("="*70)

# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА РЕАЛЬНЫХ ДАННЫХ (БЕЗ УТЕЧЕК)
# =============================================================================
def run_full_segmented_analysis(file_path, pre_period_cols=None, use_ml=True):
    """
    Сквозной пайплайн с выбором лучшего метода CUPED (без утечек данных).
    pre_period_cols – список колонок, известных до эксперимента.
    Если не задан, автоматически исключаются все целевые метрики.
    """
    setup_output_dir()
    print(f"--- ЗАПУСК АНАЛИЗА (графики в {OUTPUT_DIR}) ---")

    # Загрузка или генерация данных
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[Инфо] Файл не найден, создаем синтетический датасет...")
        np.random.seed(42)
        n = 20000
        # Предпериодные признаки
        pre_data = {
            'cov_historical_clicks': np.random.normal(10, 2, n),
            'cov_app_opens': np.random.normal(50, 15, n),
            'past_orders': np.random.poisson(3, n),
            'nfu_flg': np.random.binomial(1, 0.3, n)
        }
        # Постпериодные метрики (таргеты) – генерируем с корреляцией от предпериода, но независимо друг от друга
        life1 = (0.1 + 0.05 * (pre_data['cov_historical_clicks'] - 10)/2 + 0.02 * pre_data['past_orders'] + np.random.normal(0, 0.1, n))
        life1 = (life1 > 0.15).astype(int)
        life32 = (0.05 + 0.03 * (pre_data['cov_historical_clicks'] - 10)/2 + 0.01 * pre_data['past_orders'] + np.random.normal(0, 0.08, n))
        life32 = (life32 > 0.08).astype(int)
        orig_cost = np.exp(5 + 0.2 * pre_data['past_orders'] + 0.1 * (pre_data['cov_app_opens'] - 50)/15 + np.random.normal(0, 0.5, n))

        df_raw = pd.DataFrame({
            **pre_data,
            'life_1_day_flg': life1,
            'life_32_day_flg': life32,
            'orig_cost_sum': orig_cost
        })

    df = preprocess_data(df_raw)
    targets = ['life_1_day_flg', 'life_32_day_flg', 'orig_cost_sum']

    # Определение предпериодных признаков
    if pre_period_cols is None:
        # По умолчанию: все числовые колонки, кроме целевых метрик
        all_cols = set(df.columns)
        target_set = set(targets)
        pre_period_cols = list(all_cols - target_set - {'nfu_flg'})  # nfu_flg тоже предпериодный, но оставим его в признаках
        # Если нужно добавить nfu_flg обратно:
        if 'nfu_flg' in df.columns:
            pre_period_cols.append('nfu_flg')
        print(f"Автоматически выбраны предпериодные признаки: {pre_period_cols}")
    else:
        # Проверка, что все указанные колонки существуют
        pre_period_cols = [c for c in pre_period_cols if c in df.columns]
        print(f"Используются заданные предпериодные признаки: {pre_period_cols}")

    # Сегменты с проверкой наличия nfu_flg
    segments = {"ВЕСЬ МАССИВ": df}
    if 'nfu_flg' in df.columns:
        segments["НОВЫЕ (NFU=1)"] = df[df['nfu_flg'] == 1]
        segments["АКТИВНЫЕ (NFU=0)"] = df[df['nfu_flg'] == 0]
    else:
        print("Колонка 'nfu_flg' отсутствует в датасете. Сегментация по NFU пропущена.")

    for target in targets:
        print(f"\n" + "█" * 75 + f"\n МЕТРИКА: {target.upper()}\n" + "█" * 75)
        for seg_name, seg_df in segments.items():
            if seg_df.empty: continue

            n_samples = len(seg_df)
            mean_init, var_init = seg_df[target].mean(), seg_df[target].var(ddof=1)

            # Визуализация дерева стратификации (только предпериодные признаки)
            if len(pre_period_cols) > 0:
                visualize_stratification_tree(seg_df, target, seg_name, pre_period_cols,
                                               max_depth=4, min_samples_leaf=2000)

            # --- 1. Single CUPED (лучшая ковариата из предпериодных) ---
            corrs = []
            for c in pre_period_cols:
                if c in seg_df.columns and seg_df[c].nunique() > 1:
                    corr_val = seg_df[target].corr(seg_df[c])
                    corrs.append((c, abs(corr_val)))
            if corrs:
                best_cov = max(corrs, key=lambda x: x[1])[0]
                cov_m = np.cov(seg_df[target], seg_df[best_cov], ddof=1)
                theta_single = cov_m[0,1] / cov_m[1,1] if cov_m[1,1] > 0 else 0.0
                y_single = seg_df[target] - theta_single * (seg_df[best_cov] - seg_df[best_cov].mean())
                var_single = np.var(y_single, ddof=1)
                single_red = max(0.0, (1 - var_single / var_init) * 100)
                plot_cuped_variance_reduction(seg_df[target].values, y_single.values,
                                              target, seg_name, covariate_name=f"Single CUPED ({best_cov})")
                print(f"Single CUPED: снижение = {single_red:.2f}%")
            else:
                single_red = 0.0
                print("Single CUPED: нет подходящих предпериодных ковариат")

            # --- 2. Vanilla Multi-CUPED (только предпериодные признаки) ---
            y_multi, _, multi_red = apply_vanilla_multi_cuped(seg_df, target, pre_period_cols, top_k=10, n_splits=5)
            plot_cuped_variance_reduction(seg_df[target].values, y_multi, target, seg_name,
                                          covariate_name="Vanilla Multi-CUPED (OLS top-10)")

            # --- 3. ML-CUPED (CatBoost на предпериодных признаках) ---
            ml_red = 0.0
            if use_ml and _CATBOOST_AVAILABLE and len(pre_period_cols) > 0:
                X_pre = seg_df[pre_period_cols].copy()
                # Убираем колонки, которые могут быть константными или содержать NaN
                X_pre = X_pre.dropna(axis=1, how='all')
                if X_pre.shape[1] > 0 and X_pre.shape[0] > 5:
                    y_real, y_hat, is_binary = get_oof_predictions_catboost(X_pre, seg_df[target], n_splits=5)
                    plot_sanity_checks(y_real, y_hat, target, seg_name)
                    y_ml, ml_red = apply_ml_cuped(y_real, y_hat, target)
                    plot_cuped_variance_reduction(y_real, y_ml, target, seg_name,
                                                  covariate_name="ML-CUPED (CatBoost OOF)")
                    if seg_name == "ВЕСЬ МАССИВ" and _SHAP_AVAILABLE:
                        run_shap_analysis_full(X_pre, seg_df[target], is_binary, target, seg_name)
                else:
                    print("Недостаточно предпериодных признаков для ML-CUPED")
            elif use_ml and not _CATBOOST_AVAILABLE:
                print("CatBoost не установлен, ML-CUPED пропущен.")

            # --- Выбор лучшего ---
            results = [("Single CUPED", single_red), ("Vanilla Multi-CUPED", multi_red)]
            if ml_red > 0:
                results.append(("ML-CUPED", ml_red))
            best_method, best_reduction = max(results, key=lambda x: x[1])
            print(f"\n🏆 ЛУЧШИЙ МЕТОД ДЛЯ {target} ({seg_name}): {best_method} со снижением {best_reduction:.2f}%")

            # --- Отчёт ---
            print(f"\n[{seg_name}] (n={n_samples})")
            print(f"  • Исходные параметры  : var = {var_init:.6f} | MDE = {estimate_mde(n_samples, var_init, mean_init):.2f}%")
            print(f"  • Single CUPED (-{single_red:.2f}%): лучшая ковариата = {best_cov if corrs else 'нет'}")
            print(f"  • Vanilla Multi-CUPED (-{multi_red:.2f}%): OLS топ-10 предпериодных признаков")
            if ml_red > 0:
                print(f"  • ML-CUPED (-{ml_red:.2f}%): CatBoost OOF на предпериодных признаках")
            print(f"  • Рекомендуемый метод: {best_method} (снижение {best_reduction:.2f}%)")
            print("-" * 55)

    print(f"\n✅ Все графики сохранены в папку: {os.path.abspath(OUTPUT_DIR)}")

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    # Для реального файла можно задать предпериодные колонки вручную:
    # pre_cols = ['cov_historical_clicks', 'cov_app_opens', 'past_orders', 'nfu_flg']
    # run_full_segmented_analysis('dataset_mb.csv', pre_period_cols=pre_cols, use_ml=True)

    # Либо положиться на автоматическое исключение целевых метрик:
    run_full_segmented_analysis('dataset_back.csv', use_ml=True)

    # Валидация через A/B-симуляции
    run_ab_validation()