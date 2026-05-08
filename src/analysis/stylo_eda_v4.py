import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────
TRAIN_PATH  = Path('data/05_embeddings_v4/train.parquet')
OUTPUT_DIR  = Path('src/analysis/output/stylo_v4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / 'stylo_eda_report_v4.txt'

# 11 честных фичей!
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]

# Топ репрезентативных классов для графиков
TARGET_CLASSES =['human', 'sberai_large', 'facebook_xglm', 'rut5-base', 'gpt-3.5-turbo', 'gpt-4']

# ─────────────────────────── Утилиты ─────────────────────────────────────────
def save_fig(name: str):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ Сохранено: {path}')

def log(msg: str, file=None):
    print(msg)
    if file:
        file.write(msg + '\n')

# ─────────────────────────── Анализ ──────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    print('\n📊 График 1: Матрица корреляций (Heatmap)...')
    corr = df[STYLO_FEATURES].corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Корреляция Спирмена (Очищенная стилометрия V4)', fontweight='bold')
    plt.tight_layout()
    save_fig('01_correlation_heatmap.png')

def plot_boxplots_outliers(df: pd.DataFrame):
    print('\n📊 График 2: Boxplots...')
    df_subset = df[df['label'].isin(TARGET_CLASSES)]
    
    # Сетка 4x3 для 11 фичей
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for i, feature in enumerate(STYLO_FEATURES):
        sns.boxplot(data=df_subset, x='label', y=feature, ax=axes[i], 
                    order=TARGET_CLASSES, showfliers=False, hue='label', palette='muted', legend=False)
        axes[i].set_title(feature, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=30)
        
    axes[-1].axis('off') # Отключаем 12-й пустой график
    plt.suptitle('Распределение признаков по классам (без артефактов длины)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('02_feature_distributions.png')

def plot_radar_chart(df: pd.DataFrame):
    print('\n📊 График 3: Radar Chart...')
    df_subset = df[df['label'].isin(TARGET_CLASSES)]
    mean_profiles = df_subset.groupby('label')[STYLO_FEATURES].mean()
    
    scaler = MinMaxScaler()
    scaled_profiles = pd.DataFrame(scaler.fit_transform(mean_profiles), 
                                   index=mean_profiles.index, columns=mean_profiles.columns)
    
    categories = list(scaled_profiles.columns)
    N = len(categories)
    angles =[n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for cls_name, color in zip(TARGET_CLASSES, colors):
        values = scaled_profiles.loc[cls_name].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cls_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.title('Лепестковая диаграмма: Честный "почерк" моделей', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    save_fig('03_radar_chart_profiles.png')

def plot_pca_clusters(df: pd.DataFrame):
    print('\n📊 График 4: PCA...')
    df_sample = df[df['label'].isin(TARGET_CLASSES)].groupby('label').sample(n=2000, random_state=42, replace=True)
    X = StandardScaler().fit_transform(df_sample[STYLO_FEATURES])
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df_sample['pca_1'], df_sample['pca_2'] = X_pca[:, 0], X_pca[:, 1]
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_sample, x='pca_1', y='pca_2', hue='label', palette='tab10', alpha=0.6, s=15, edgecolor=None)
    plt.title(f'PCA проекция (Дисперсия: {pca.explained_variance_ratio_.sum():.2%})', fontweight='bold', fontsize=14)
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title='Автор', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig('04_pca_clusters.png')

def generate_report(df: pd.DataFrame):
    print('\n📝 Генерируем выводы...')
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        log('=' * 65, f)
        log('ОТЧЕТ ПО СТИЛОМЕТРИИ V4 (БЕЗ УТЕЧЕК ДЛИНЫ)', f)
        log('=' * 65, f)
        
        corr = df[STYLO_FEATURES].corr(method='spearman')
        high_corr =[]
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.85:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        log('\n[1] АНАЛИЗ КОРРЕЛЯЦИЙ (Мультиколлинеарность)', f)
        if high_corr:
            log('  ⚠️ Найдены сильно скоррелированные признаки:', f)
            for c1, c2, val in high_corr: log(f'    - {c1} и {c2}: r = {val:.2f}', f)
        else:
            log('  ✅ Сильных корреляций (>0.85) нет. Признаковое пространство оптимально!', f)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_PATH.exists():
        print(f"❌ Ошибка: {TRAIN_PATH} не найден.")
        return
    df = pd.read_parquet(TRAIN_PATH)
    plot_correlation_heatmap(df)
    plot_boxplots_outliers(df)
    plot_radar_chart(df)
    plot_pca_clusters(df)
    generate_report(df)
    print("\n🚀 ИССЛЕДОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()