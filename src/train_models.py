"""
Финальное A/B/C тестирование моделей (CatBoost) - CPU Версия
Лабораторная работа №3 / НИР (Этап 2)

Запуск: python src/train_models.py
"""

import os
import gc
import warnings
import pandas as pd
import numpy as np
import joblib
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────
TRAIN_PATH = 'data/05_embeddings/train.parquet'
TEST_PATH  = 'data/05_embeddings/test.parquet'
MODEL_DIR  = 'models/final'
os.makedirs(MODEL_DIR, exist_ok=True)

STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count'
]
TARGET_COL = 'label'
RANDOM_SEED = 42

# ─────────────────────────── Кастомный Трансформер ─────────────────────────
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.0, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def fit(self, X, y=None):
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self
        
    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

# ─────────────────────────── Утилиты ─────────────────────────────────────────

def unpack_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Распаковывает колонку-список в 312 отдельных float32 колонок"""
    print("  📦 Распаковка BERT эмбеддингов...")
    # Превращаем серию списков в 2D numpy массив
    emb_matrix = np.vstack(df['bert_embedding'].values)
    
    # Создаем имена колонок
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    
    # Создаем новый датафрейм и приклеиваем к исходному (используем float32 для экономии RAM)
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    
    # Удаляем тяжелую колонку-список и склеиваем
    df = df.drop(columns=['bert_embedding'])
    return pd.concat([df, emb_df], axis=1), emb_cols

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, model_name):
    print(f"\n" + "="*50)
    print(f"🚀 ОБУЧЕНИЕ: {model_name.upper()}")
    print("="*50)
    
    start_time = time()
    pipeline.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"⏱️ Время обучения: {train_time/60:.1f} минут")
    
    print("🔮 Оценка на тестовой выборке...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"📈 Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    
    # Сохраняем модель
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    joblib.dump(pipeline, model_path)
    print(f"💾 Модель сохранена в {model_path}")
    
    return acc, macro_f1

# ─────────────────────────── Главная логика ──────────────────────────────────

def main():
    print("📥 Загрузка обогащенных данных...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    df_train, bert_features = unpack_embeddings(df_train)
    df_test, _ = unpack_embeddings(df_test)
    
    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]
    
    # Общие параметры CatBoost для CPU
    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'thread_count': -1, # 🔥 Используем все ядра процессора ноута!
        'auto_class_weights': 'Balanced', # Решает проблему дисбаланса
        'random_seed': RANDOM_SEED,
        'verbose': 100 # Печатать лог каждые 100 итераций, чтобы не спамить в консоль
    }

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # МОДЕЛЬ А: Только Стилометрия
    # ─────────────────────────────────────────────────────────────────────────
    print("\nПодготовка данных для Модели А...")
    X_train_A = df_train[STYLO_FEATURES]
    X_test_A = df_test[STYLO_FEATURES]
    
    pipe_A = Pipeline([
        ('winsorizer', Winsorizer()),
        ('scaler', StandardScaler()),
        ('classifier', CatBoostClassifier(**cb_params))
    ])
    
    results['Модель А (Стилометрия)'] = train_and_evaluate(pipe_A, X_train_A, y_train, X_test_A, y_test, 'model_A_stylo')
    del X_train_A, X_test_A; gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # МОДЕЛЬ Б: Только Семантика (BERT)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nПодготовка данных для Модели Б...")
    X_train_B = df_train[bert_features]
    X_test_B = df_test[bert_features]
    
    # Для BERT скейлер не нужен, передаем напрямую
    pipe_B = Pipeline([
        ('classifier', CatBoostClassifier(**cb_params))
    ])
    
    results['Модель Б (Семантика)'] = train_and_evaluate(pipe_B, X_train_B, y_train, X_test_B, y_test, 'model_B_bert')
    del X_train_B, X_test_B; gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # МОДЕЛЬ В: ГИБРИД (Стилометрия + Семантика)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nПодготовка данных для Модели В (Гибрид)...")
    all_features = STYLO_FEATURES + bert_features
    X_train_C = df_train[all_features]
    X_test_C = df_test[all_features]
    
    # ColumnTransformer: применяем препроцессинг только к СТИЛОМЕТРИИ.
    # remainder='passthrough' означает "остальные колонки (BERT) пропустить без изменений"
    preprocessor_hybrid = ColumnTransformer(
        transformers=[
            ('stylo_prep', Pipeline([
                ('winsorizer', Winsorizer()),
                ('scaler', StandardScaler())
            ]), STYLO_FEATURES)
        ],
        remainder='passthrough'
    )
    
    pipe_C = Pipeline([
        ('preprocessor', preprocessor_hybrid),
        ('classifier', CatBoostClassifier(**cb_params))
    ])
    
    results['Модель В (Гибрид)'] = train_and_evaluate(pipe_C, X_train_C, y_train, X_test_C, y_test, 'model_C_hybrid')

    # ─────────────────────────── ФИНАЛЬНЫЙ ОТЧЕТ ─────────────────────────────
    print("\n" + "🏆"*20)
    print("ИТОГОВОЕ СРАВНЕНИЕ ГИПОТЕЗ ИЗ ТЗ:")
    for name, (acc, f1) in results.items():
        print(f"{name:.<30} Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print("🏆"*20)

if __name__ == "__main__":
    main()