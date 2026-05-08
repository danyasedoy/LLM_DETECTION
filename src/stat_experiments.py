"""
Статистическая проверка значимости результатов (Лабораторная работа №5)
Множественные запуски (5 seeds), расчет 95% CI, p-value и Boxplots.

Запуск: python src/stat_experiments.py
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from time import time
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────
TRAIN_PATH = 'data/05_embeddings_v4/train.parquet'
TEST_PATH  = 'data/05_embeddings_v4/test.parquet'
OUTPUT_DIR = 'src/analysis/experiments'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 11 честных стилометрических фичей (без word_count, с mattr)
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]
TARGET_COL = 'label'

# 5 случайных сидов для множественных запусков
SEEDS =[42, 100, 200, 300, 400]

# ─────────────────────────── Кастомный трансформер ─────────────────────────
class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lb_ = np.quantile(X, 0.0, axis=0)
        self.ub_ = np.quantile(X, 0.99, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

def unpack_embeddings(df):
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    return pd.concat([df.drop(columns=['bert_embedding']), emb_df], axis=1), emb_cols

# ─────────────────────────── Главная логика ──────────────────────────────────
def main():
    print("📥 1. Загрузка и однократная предобработка данных (чтобы не терять время в цикле)...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    df_train, bert_cols = unpack_embeddings(df_train)
    df_test, _ = unpack_embeddings(df_test)
    
    y_train, y_test = df_train[TARGET_COL], df_test[TARGET_COL]
    
    # ⚙️ Применяем Винзоризацию и Скейлер один раз!
    prep = Pipeline([('win', Winsorizer()), ('scl', StandardScaler())])
    X_train_stylo = pd.DataFrame(prep.fit_transform(df_train[STYLO_FEATURES]), columns=STYLO_FEATURES)
    X_test_stylo  = pd.DataFrame(prep.transform(df_test[STYLO_FEATURES]), columns=STYLO_FEATURES)
    
    X_train_bert = df_train[bert_cols].reset_index(drop=True)
    X_test_bert  = df_test[bert_cols].reset_index(drop=True)
    
    # Гибридные наборы
    X_train_hybrid = pd.concat([X_train_stylo, X_train_bert], axis=1)
    X_test_hybrid  = pd.concat([X_test_stylo, X_test_bert], axis=1)
    
    del df_train, df_test; gc.collect()

    print("🚀 2. Старт множественных экспериментов (5 запусков x 3 модели = 15 обучений)...")
    
    results =[]
    
    # Чтобы было быстрее, используем 2500 итераций вместо 4000
    base_cb_params = {
        'iterations': 2500,
        'learning_rate': 0.178,
        'depth': 4,
        'l2_leaf_reg': 1.059,
        'task_type': 'GPU',
        'auto_class_weights': 'SqrtBalanced',
        'verbose': 0 # Отключаем вывод в консоль для каждого дерева
    }

    for seed in SEEDS:
        print(f"\n🌱 Запуск с random_seed = {seed}...")
        params = base_cb_params.copy()
        params['random_seed'] = seed
        
        # --- Модель А (Стилометрия) ---
        t0 = time()
        model_A = CatBoostClassifier(**params).fit(X_train_stylo, y_train)
        f1_A = f1_score(y_test, model_A.predict(X_test_stylo), average='macro')
        results.append({'Model': 'Stylo', 'Seed': seed, 'Macro F1': f1_A, 'Time': time()-t0})
        print(f"   ✓ Stylo: F1 = {f1_A:.4f} ({(time()-t0):.1f} сек)")
        
        # --- Модель Б (Семантика) ---
        t0 = time()
        model_B = CatBoostClassifier(**params).fit(X_train_bert, y_train)
        f1_B = f1_score(y_test, model_B.predict(X_test_bert), average='macro')
        results.append({'Model': 'BERT', 'Seed': seed, 'Macro F1': f1_B, 'Time': time()-t0})
        print(f"   ✓ BERT: F1 = {f1_B:.4f} ({(time()-t0):.1f} сек)")
        
        # --- Модель В (Гибрид) ---
        t0 = time()
        model_C = CatBoostClassifier(**params).fit(X_train_hybrid, y_train)
        f1_C = f1_score(y_test, model_C.predict(X_test_hybrid), average='macro')
        results.append({'Model': 'Hybrid', 'Seed': seed, 'Macro F1': f1_C, 'Time': time()-t0})
        print(f"   ✓ Hybrid: F1 = {f1_C:.4f} ({(time()-t0):.1f} сек)")

    # ─────────────────────────── СТАТИСТИКА ──────────────────────────────
    print("\n📊 3. Расчет статистики и построение графиков...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'raw_experiment_results.csv'), index=False)
    
    # Функция расчета 95% CI по формуле со слайда 10: CI = 1.96 * (std / sqrt(n))
    def calc_ci(x):
        return 1.96 * (np.std(x, ddof=1) / np.sqrt(len(x)))

    stats_df = df_results.groupby('Model')['Macro F1'].agg(['mean', calc_ci]).rename(columns={'calc_ci': '95% CI'})
    
    # Разделяем массивы F1 для t-теста
    f1_stylo = df_results[df_results['Model'] == 'Stylo']['Macro F1'].values
    f1_bert = df_results[df_results['Model'] == 'BERT']['Macro F1'].values
    f1_hybrid = df_results[df_results['Model'] == 'Hybrid']['Macro F1'].values

    # Парный t-тест (ttest_rel) для зависимых выборок (обучение на одних и тех же данных с разными seed)
    _, p_value_stylo = stats.ttest_rel(f1_hybrid, f1_stylo)
    _, p_value_bert = stats.ttest_rel(f1_hybrid, f1_bert)

    # Вывод в консоль
    print("\n" + "="*60)
    print("🏆 ТАБЛИЦА РЕЗУЛЬТАТОВ (По формату IMRaD):")
    for model in['Stylo', 'BERT', 'Hybrid']:
        mean = stats_df.loc[model, 'mean']
        ci = stats_df.loc[model, '95% CI']
        print(f" {model:<8} | F1 = {mean:.4f}[95% CI: {(mean-ci):.4f} – {(mean+ci):.4f}]")
    
    print("\n🔬 Статистическая значимость (t-тест):")
    print(f" Hybrid vs Stylo: p-value = {p_value_stylo:.4e} (Значимо: {p_value_stylo < 0.05})")
    print(f" Hybrid vs BERT:  p-value = {p_value_bert:.4e} (Значимо: {p_value_bert < 0.05})")
    print("="*60)

    # Построение Boxplot (Требование со слайда 13)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='Macro F1', data=df_results, palette=['#DD8452', '#4C72B0', '#55A868'], showmeans=True, 
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    plt.title('Распределение метрики Macro F1 (5 запусков с разными random_seed)', fontsize=12, fontweight='bold')
    plt.ylabel('Macro F1 Score')
    plt.xlabel('Тип признаков')
    
    plot_path = os.path.join(OUTPUT_DIR, 'f1_boxplots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ График Boxplot сохранен: {plot_path}")

if __name__ == "__main__":
    main()