"""
Анализ важности признаков (Feature Importance)
Лабораторная работа №3 / НИР (Интерпретация результатов)

Запуск: python src/feature_importance.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем импорты для нашего кастомного класса
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────── Конфигурация ────────────────────────────────────
MODEL_PATH = 'models/final_v3/hybrid_v3.pkl'
OUTPUT_DIR = 'src/analysis/output/interpretation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Наши 12 стилометрических признаков
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]

# Имена 768 колонок RuBERT-base
BERT_FEATURES =[f'bert_{i}' for i in range(768)]

# В ColumnTransformer сначала шли стило-фичи, потом passthrough (BERT)
ALL_FEATURE_NAMES = STYLO_FEATURES + BERT_FEATURES

# ─────────────────────────── Кастомный класс ─────────────────────────────────
# Нам обязательно нужно определить класс здесь, чтобы joblib смог его "вспомнить"
class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lb_ = np.quantile(X, 0.0, axis=0)
        self.ub_ = np.quantile(X, 0.99, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

# ─────────────────────────── Отрисовка графиков ─────────────────────────────

def plot_top_features(df_imp: pd.DataFrame):
    """Рисует ТОП-20 самых важных признаков"""
    print("📊 Строим график ТОП-20 признаков...")
    top_20 = df_imp.head(20)
    
    # Красим столбцы: синие для BERT, оранжевые для Стилометрии
    colors =['#4C72B0' if 'bert' in feat else '#DD8452' for feat in top_20['Feature']]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_20, palette=colors)
    
    plt.title('ТОП-20 самых важных признаков (CatBoost Hybrid V3)', fontsize=14, fontweight='bold')
    plt.xlabel('Важность признака (Feature Importance %)', fontsize=12)
    plt.ylabel('Признак', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements =[Patch(facecolor='#4C72B0', label='Семантика (BERT)'),
                       Patch(facecolor='#DD8452', label='Стилометрия (Лингвистика)')]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'top_20_features.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ График сохранен: {out_path}")

def plot_group_importance(df_imp: pd.DataFrame):
    """Считает суммарный вклад Стилометрии и Семантики"""
    print("\n📊 Строим график группового вклада...")
    
    stylo_importance = df_imp[df_imp['Feature'].isin(STYLO_FEATURES)]['Importance'].sum()
    bert_importance = df_imp[~df_imp['Feature'].isin(STYLO_FEATURES)]['Importance'].sum()
    
    labels =['Стилометрия\n(12 признаков)', 'Семантика (RuBERT)\n(768 признаков)']
    sizes =[stylo_importance, bert_importance]
    colors =['#DD8452', '#4C72B0']
    explode = (0.1, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 14})
    plt.title('Вклад групп признаков в итоговое решение модели', fontsize=16, fontweight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, 'group_importance.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ График сохранен: {out_path}")

def main():
    print(f"📥 Загрузка обученной модели из {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    # Достаем CatBoost
    catboost_model = pipeline.named_steps['clf']
    
    print("⚙️ Извлечение весов признаков...")
    importances = catboost_model.get_feature_importance()
    
    df_imp = pd.DataFrame({
        'Feature': ALL_FEATURE_NAMES,
        'Importance': importances
    })
    
    df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plot_top_features(df_imp)
    plot_group_importance(df_imp)
    
    print("\n" + "="*40)
    print("🏆 ТОП-10 САМЫХ СИЛЬНЫХ ПРИЗНАКОВ:")
    print("="*40)
    print(df_imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()