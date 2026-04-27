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

TRAIN_PATH  = Path('data/04_stylo_features/train.parquet')
OUTPUT_DIR  = Path('src/analysis/output/stylo')
REPORT_PATH = OUTPUT_DIR / 'stylo_eda_report.txt'

STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count', 'sentence_count'
]

# Топ классов для красивых графиков (чтобы не было каши из 19 штук)
TARGET_CLASSES =['human', 'sberai_large', 'facebook_xglm', 'rugpt3-large', 'gpt-3.5-turbo', 'gpt-4']

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
    corr = df[STYLO_FEATURES].corr(method='spearman') # Spearman лучше для ненормальных распределений
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Корреляция Спирмена между стилометрическими признаками', fontweight='bold')
    plt.tight_layout()
    save_fig('01_correlation_heatmap.png')

def plot_boxplots_outliers(df: pd.DataFrame):
    print('\n📊 График 2: Boxplots (Поиск выбросов и распределение)...')
    
    df_subset = df[df['label'].isin(TARGET_CLASSES)]
    
    # Строим сетку графиков (3x3) для 9 фичей
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(STYLO_FEATURES):
        sns.boxplot(data=df_subset, x='label', y=feature, ax=axes[i], 
                    order=TARGET_CLASSES, showfliers=False, palette='muted')
        axes[i].set_title(feature, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=30)
        
    plt.suptitle('Распределение стилометрических признаков по классам (без экстремальных выбросов)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('02_feature_distributions.png')

def plot_radar_chart(df: pd.DataFrame):
    print('\n📊 График 3: Radar Chart (Профили стиля LLM)...')
    
    df_subset = df[df['label'].isin(TARGET_CLASSES)]
    
    # Считаем средние значения фичей для каждого класса
    mean_profiles = df_subset.groupby('label')[STYLO_FEATURES].mean()
    
    # Нормализуем от 0 до 1, чтобы можно было нарисовать на одной шкале радара
    scaler = MinMaxScaler()
    scaled_profiles = pd.DataFrame(scaler.fit_transform(mean_profiles), 
                                   index=mean_profiles.index, columns=mean_profiles.columns)
    
    # Настройка радара
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
    
    # Рисуем линии для ключевых классов
    colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (cls_name, color) in enumerate(zip(TARGET_CLASSES, colors)):
        values = scaled_profiles.loc[cls_name].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cls_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.title('Лепестковая диаграмма: Усредненный "почерк" моделей', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    save_fig('03_radar_chart_profiles.png')

def plot_pca_clusters(df: pd.DataFrame):
    print('\n📊 График 4: PCA (2D Проекция стилистических пространств)...')
    
    # Для PCA берем сэмпл, иначе точки сольются в сплошное пятно (по 2000 текстов на класс)
    df_sample = df[df['label'].isin(TARGET_CLASSES)].groupby('label').sample(n=2000, random_state=42, replace=True)
    
    # Масштабируем данные
    X = StandardScaler().fit_transform(df_sample[STYLO_FEATURES])
    
    # Сжимаем 9 измерений в 2
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    df_sample['pca_1'] = X_pca[:, 0]
    df_sample['pca_2'] = X_pca[:, 1]
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_sample, x='pca_1', y='pca_2', hue='label', 
                    palette='tab10', alpha=0.6, s=15, edgecolor=None)
    
    plt.title(f'PCA проекция стилометрии (Дисперсия: {pca.explained_variance_ratio_.sum():.2%})', 
              fontweight='bold', fontsize=14)
    plt.xlabel(f'Главная компонента 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'Главная компонента 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title='Автор', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_fig('04_pca_clusters.png')

# ─────────────────────────── Отчет и выводы по предобработке (Шаг 4) ──────────

def generate_report(df: pd.DataFrame):
    print('\n📝 Генерируем выводы для предобработки...')
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        log('=' * 65, f)
        log('ОТЧЕТ ПО СТИЛОМЕТРИИ И ПЛАН ПРЕДОБРАБОТКИ (ЛАБ №3, ШАГИ 2 И 4)', f)
        log('=' * 65, f)
        
        # 1. Анализ пропусков
        log('\n[1] АНАЛИЗ ПРОПУСКОВ И ИСКЛЮЧЕНИЙ (Missing values)', f)
        nulls = df[STYLO_FEATURES].isnull().sum()
        if nulls.sum() == 0:
            log('  ✅ Пропусков (NaN) в извлеченных признаках нет.', f)
        else:
            log('  ❌ Обнаружены пропуски:', f)
            log(str(nulls[nulls > 0]), f)
            
        # 2. Анализ выбросов (для Шага 4 лабы)
        log('\n[2] АНАЛИЗ ВЫБРОСОВ (Outliers)', f)
        log('  В стилометрии естественного языка всегда присутствуют "тяжелые хвосты" распределений.', f)
        for feat in['avg_sentence_len', 'ttr', 'word_count']:
            q99 = df[feat].quantile(0.99)
            mx = df[feat].max()
            log(f'  - {feat}: 99-й перцентиль = {q99:.2f}, Максимум = {mx:.2f}', f)
        log('  > Решение для Пайплайна: Будет применена Винзоризация (Winsorization) — обрезка 99-го перцентиля, '
            'чтобы одиночные тексты без точек не ломали веса линейных алгоритмов.', f)

        # 3. Анализ корреляций
        log('\n[3] КОРРЕЛЯЦИЯ И СОКРАЩЕНИЕ ПРИЗНАКОВ (Feature Selection)', f)
        corr = df[STYLO_FEATURES].corr(method='spearman')
        high_corr =[]
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.85:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        if high_corr:
            log('  ⚠️ Обнаружены сильно скоррелированные признаки (|r| > 0.85):', f)
            for c1, c2, val in high_corr:
                log(f'    - {c1} и {c2}: r = {val:.2f}', f)
            log('  > Решение: В конвейере (Пункт 7) следует рассмотреть удаление `sentence_count`, '
                'так как он полностью дублирует `word_count` и не несет новой информации о стиле.', f)
        
        # 4. Выводы для масштабирования
        log('\n[4] МАСШТАБИРОВАНИЕ (Scaling)', f)
        log('  Признаки имеют совершенно разный масштаб (ttr ~ 0.5, word_count ~ 500).', f)
        log('  > Решение для Пайплайна: Обязательное использование StandardScaler '
            'на числовых колонках. Обучение скейлера (fit) будет производиться ТОЛЬКО на train.parquet.', f)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Загрузка обучающей выборки: {TRAIN_PATH}")
    if not TRAIN_PATH.exists():
        print("❌ Ошибка: файл не найден. Выполни стилометрию (stylo_extractor.py) перед EDA!")
        return
        
    df = pd.read_parquet(TRAIN_PATH)
    
    plot_correlation_heatmap(df)
    plot_boxplots_outliers(df)
    plot_radar_chart(df)
    plot_pca_clusters(df)
    
    generate_report(df)
    print("\n🚀 ИССЛЕДОВАНИЕ СТИЛОМЕТРИИ УСПЕШНО ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()