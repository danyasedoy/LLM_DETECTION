"""
Базовый классификатор (Baseline) на основе стилометрии
Лабораторная работа №3 / НИР (Шаги 4, 6, 7)

Запуск: python src/baseline_pipeline.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────
TRAIN_PATH = 'data/04_stylo_features/train.parquet'
TEST_PATH  = 'data/04_stylo_features/test.parquet'
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ПУНКТ 6: Отбор признаков (Feature Selection)
# Исключаем 'sentence_count', так как EDA показал корреляцию 0.92 с 'word_count'
SELECTED_FEATURES =[
    'avg_word_len', 
    'avg_sentence_len', 
    'ttr', 
    'punct_ratio', 
    'upper_ratio', 
    'digit_ratio', 
    'stopword_ratio', 
    'word_count'
]
TARGET_COL = 'label'
RANDOM_SEED = 42

# ─────────────────────────── Кастомные Трансформеры ──────────────────────────

# ПУНКТ 4: Обработка выбросов (Винзоризация) без утечки данных!
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Обрезает аномальные значения по заданным квантилям (например, 99%).
    Вычисляет пороги ТОЛЬКО на обучающей выборке (fit), 
    и применяет их к тестовой (transform), чтобы избежать протечки.
    """
    def __init__(self, lower_quantile=0.0, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def fit(self, X, y=None):
        # Запоминаем пороги на основе Train
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self
        
    def transform(self, X):
        # Применяем запомненные пороги
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

# ─────────────────────────── Основная логика ───────────────────────────────

def main():
    print("📥 Загрузка данных...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # Отделяем признаки (X) от целевой переменной (y)
    X_train = df_train[SELECTED_FEATURES]
    y_train = df_train[TARGET_COL]
    
    X_test = df_test[SELECTED_FEATURES]
    y_test = df_test[TARGET_COL]
    
    print(f"📊 Размер Train: {len(X_train):,} строк")
    print(f"📊 Размер Test:  {len(X_test):,} строк")
    print(f"🔍 Используемые признаки ({len(SELECTED_FEATURES)} шт.): {SELECTED_FEATURES}")

    # ПУНКТ 7: Разработка воспроизводимого конвейера (Pipeline)
    print("\n⚙️ Сборка ML-конвейера (Pipeline)...")
    
    # 1. Блок предобработки (только для выбранных числовых колонок)
    numeric_transformer = Pipeline(steps=[
        ('winsorizer', Winsorizer(lower_quantile=0.0, upper_quantile=0.99)), # Убираем выбросы
        ('scaler', StandardScaler())                                         # Масштабируем (mean=0, std=1)
    ])
    
    # Обертка ColumnTransformer гарантирует, что мы применяем шаги только к нужным колонкам
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, SELECTED_FEATURES)
        ]
    )
    
    # 2. Финальный пайплайн: Предобработка + Классификатор
    # Используем Логистическую Регрессию как быстрый интерпретируемый Baseline
    # class_weight='balanced' решает проблему экстремального дисбаланса (Пункт 4/5)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', # Борьба с дисбалансом (1.3k GPT-4 vs 535k Human)
            random_state=RANDOM_SEED, # Фиксация seed для воспроизводимости (Требование №3)
            max_iter=1000,
            n_jobs=-1 # Используем все ядра процессора
        ))
    ])

    print("🧠 Обучение конвейера на Train-выборке (fit)...")
    # ВАЖНО: Скейлер и Винзоризатор вычислят параметры ТОЛЬКО здесь!
    pipeline.fit(X_train, y_train)

    print("🔮 Применение конвейера к Test-выборке (predict)...")
    # ВАЖНО: Скейлер и Винзоризатор применят сохраненные параметры, протечки нет!
    y_pred = pipeline.predict(X_test)

    # ─────────────────────────── Оценка качества ─────────────────────────────
    print("\n" + "="*50)
    print("🏆 РЕЗУЛЬТАТЫ БАЗОВОЙ МОДЕЛИ (ТОЛЬКО СТИЛОМЕТРИЯ)")
    print("="*50)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nДетальный отчет по классам:")
    print(classification_report(y_test, y_pred))

    # Сериализация конвейера (Пункт 7: "позволяющий зафиксировать процесс")
    model_path = os.path.join(MODEL_DIR, 'stylo_baseline_pipeline.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Обученный конвейер сохранен (сериализован) в: {model_path}")

if __name__ == "__main__":
    main()