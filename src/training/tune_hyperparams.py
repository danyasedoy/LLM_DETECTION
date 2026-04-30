import os
import gc
import json
import warnings
import pandas as pd
import numpy as np
import optuna
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────
TRAIN_PATH = 'data/05_embeddings_v4/train.parquet'
VAL_PATH   = 'data/05_embeddings_v4/val.parquet'  # Для тюнинга используем VAL, а не TEST!
OUTPUT_DIR = 'models/final_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio',
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
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    df = df.drop(columns=['bert_embedding'])
    return pd.concat([df, emb_df], axis=1), emb_cols

def prepare_data():
    """Загружает и предварительно трансформирует данные для ускорения Optuna"""
    print("📥 Загрузка Train и Val выборок...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)
    
    # Возьмем стратифицированный сэмпл (150к строк) из Train, чтобы Optuna летала быстро.
    print("✂️ Создание стратифицированного сэмпла для быстрого тюнинга...")
    df_train = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
    
    df_train, bert_cols = unpack_embeddings(df_train)
    df_val, _ = unpack_embeddings(df_val)
    
    y_train, y_val = df_train['label'], df_val['label']
    all_features = STYLO_FEATURES + bert_cols
    
    print("⚙️ Предварительная обработка (Скейлер и Винзоризация)...")
    preprocessor = ColumnTransformer([
        ('stylo', Pipeline([('win', Winsorizer()), ('scl', StandardScaler())]), STYLO_FEATURES)
    ], remainder='passthrough')
    
    # Делаем fit на train и transform на обоих
    X_train_prep = preprocessor.fit_transform(df_train[all_features])
    X_val_prep = preprocessor.transform(df_val[all_features])
    
    del df_train, df_val; gc.collect()
    
    return X_train_prep, y_train, X_val_prep, y_val

# ─────────────────────────── Optuna Objective ────────────────────────────────

def objective(trial, X_train, y_train, X_val, y_val):
    """Целевая функция, которую Optuna будет пытаться максимизировать"""
    
    # 1. Задаем диапазоны поиска гиперпараметров (Байесовское пространство)
    params = {
        'iterations': 1000, # Держим 1000 для скорости поиска. В финале обучим на 3000
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'task_type': 'GPU',
        'auto_class_weights': 'SqrtBalanced',
        'random_seed': 42,
        'verbose': 0 # Отключаем спам в консоль
    }
    
    # 2. Обучаем модель с текущими параметрами
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    # 3. Оцениваем на Val выборке
    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    return macro_f1

# ─────────────────────────── Главная логика ──────────────────────────────────

def main():
    X_train, y_train, X_val, y_val = prepare_data()
    
    print("\n" + "🔥"*20)
    print("СТАРТ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ (OPTUNA)")
    print("🔥"*20)
    
    # Создаем "исследование" (study), цель - максимизировать F1-score
    study = optuna.create_study(direction="maximize", study_name="CatBoost_Hybrid_Tuning")
    
    # Запускаем 20 попыток (trials).
    # lambda-функция нужна, чтобы прокинуть данные в objective
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    
    print("\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print("Лучший Macro F1:", study.best_value)
    print("Лучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Сохраняем лучшие параметры в JSON, чтобы потом вставить их в финальный пайплайн
    params_path = os.path.join(OUTPUT_DIR, 'best_hyperparams.json')
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"💾 Гиперпараметры сохранены в {params_path}")

if __name__ == "__main__":
    main()