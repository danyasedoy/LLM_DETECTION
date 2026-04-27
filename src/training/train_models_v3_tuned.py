"""
Финальное обучение Гибридной Модели V3
Лабораторная работа №4 (Итоговый пайплайн с лучшими гиперпараметрами)

Запуск: python src/train_final.py
"""

import os
import gc
import json
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
TRAIN_PATH = 'data/05_embeddings_v3/train.parquet'
TEST_PATH  = 'data/05_embeddings_v3/test.parquet' # Строго TEST!
PARAMS_PATH= 'models/final_v3/best_hyperparams.json'
MODEL_DIR  = 'models/final_v3'

STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]

class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lb_ = np.quantile(X, 0.0, axis=0)
        self.ub_ = np.quantile(X, 0.99, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

def unpack_embeddings(df):
    print("  📦 Распаковка 768D BERT эмбеддингов...")
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    df = df.drop(columns=['bert_embedding'])
    return pd.concat([df, emb_df], axis=1), emb_cols

def main():
    print("📥 Загрузка обучающей и тестовой выборки...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    df_train, bert_cols = unpack_embeddings(df_train)
    df_test, _ = unpack_embeddings(df_test)
    
    y_train, y_test = df_train['label'], df_test['label']
    all_features = STYLO_FEATURES + bert_cols
    X_train, X_test = df_train[all_features], df_test[all_features]
    del df_train, df_test; gc.collect()

    print(f"⚙️ Чтение лучших гиперпараметров из {PARAMS_PATH}...")
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    
    # Добавляем базовые параметры, которые мы не тюнили
    best_params.update({
        'iterations': 4000,           # Неглубокие деревья (depth=5) нужно строить дольше!
        'task_type': 'GPU',
        'auto_class_weights': 'SqrtBalanced',
        'random_seed': 42,
        'verbose': 500
    })

    print("\n🛠️ Сборка финального Пайплайна...")
    preprocessor = ColumnTransformer([
        ('stylo', Pipeline([('win', Winsorizer()), ('scl', StandardScaler())]), STYLO_FEATURES)
    ], remainder='passthrough')
    
    final_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', CatBoostClassifier(**best_params))
    ])

    print("\n🚀 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ (Train -> Test)")
    t0 = time()
    final_pipeline.fit(X_train, y_train)
    print(f"⏱️ Обучение заняло: {(time()-t0)/60:.1f} мин")

    print("\n🔮 Валидация на отложенном TEST датасете...")
    y_pred = final_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n" + "🏆"*20)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (19 классов, Optuna params):")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print("🏆"*20)
    
    model_path = os.path.join(MODEL_DIR, 'Ultimate_Hybrid_V3.pkl')
    joblib.dump(final_pipeline, model_path)
    print(f"💾 Ультимативная модель сохранена в {model_path}")

if __name__ == "__main__":
    main()