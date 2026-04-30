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

TRAIN_PATH = 'data/05_embeddings_v3/train.parquet'
TEST_PATH  = 'data/05_embeddings_v3/test.parquet'
MODEL_DIR  = 'models/final_v3'
os.makedirs(MODEL_DIR, exist_ok=True)

# Теперь у нас 12 фичей!
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
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
    print("  📦 Распаковка 768D BERT эмбеддингов...")
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    df = df.drop(columns=['bert_embedding'])
    return pd.concat([df, emb_df], axis=1), emb_cols

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, model_name):
    print(f"\n🚀 ОБУЧЕНИЕ: {model_name}")
    t0 = time()
    pipeline.fit(X_train, y_train)
    print(f"⏱️ Обучение: {(time()-t0)/60:.1f} мин")
    y_pred = pipeline.predict(X_test)
    acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')
    print(f"📈 Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    joblib.dump(pipeline, os.path.join(MODEL_DIR, f'{model_name}.pkl'))
    return acc, f1

def main():
    print("📥 Загрузка данных V3 (19 честных классов)...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    df_train, bert_cols = unpack_embeddings(df_train)
    df_test, _ = unpack_embeddings(df_test)
    
    y_train, y_test = df_train['label'], df_test['label']
  
    cb_params = {
        'iterations': 3000,
        'learning_rate': 0.05,
        'task_type': 'GPU',
        'auto_class_weights': 'SqrtBalanced',
        'random_seed': 42,
        'verbose': 500
    }
    results = {}

    print("\n--- Модель В (ГИБРИД V3) ---")
    all_features = STYLO_FEATURES + bert_cols
    preprocessor = ColumnTransformer([('stylo', Pipeline([('win', Winsorizer()), ('scl', StandardScaler())]), STYLO_FEATURES)],
        remainder='passthrough'
    )
    
    pipe_C = Pipeline([
        ('prep', preprocessor),
        ('clf', CatBoostClassifier(**cb_params))
    ])
    
    results['Гибрид V3 (19 классов)'] = train_and_evaluate(pipe_C, df_train[all_features], y_train, df_test[all_features], y_test, 'hybrid_v3')

    print("\n🏆 РЕЗУЛЬТАТ ИТЕРАЦИИ 3:")
    for n, (a, f) in results.items(): print(f"{n} -> Acc: {a:.4f} | F1: {f:.4f}")

if __name__ == "__main__":
    main()