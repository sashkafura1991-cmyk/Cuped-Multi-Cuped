import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cuped_variance_reduction(df, target_col, dict_of_series, output_dir):
    """Строит графики (KDE + Boxplot) сравнения оригинальной метрики и CUPED."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.kdeplot(df[target_col], label='Original', fill=True, ax=axes[0], color='#ff9999', alpha=0.5)
    colors = ['#66b3ff', '#99ff99', '#ffcc99']

    for idx, (name, series) in enumerate(dict_of_series.items()):
        sns.kdeplot(series, label=name, fill=True, ax=axes[0], color=colors[idx % len(colors)], alpha=0.5)

    axes[0].set_title(f'Сжатие плотности: {target_col}', fontsize=12)
    axes[0].legend()

    data_to_plot = pd.DataFrame({'Original': df[target_col]})
    for name, series in dict_of_series.items():
        data_to_plot[name] = series

    sns.boxplot(data=data_to_plot, ax=axes[1])
    axes[1].set_title('Изменение разброса', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_col}_variance_reduction.png"))
    plt.close()
