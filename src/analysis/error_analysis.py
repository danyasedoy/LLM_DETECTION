"""
Анализ ошибок модели (Confusion Matrix)
Помогает понять, где именно "застревает" алгоритм.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = 'models/final_v3/Ultimate_Hybrid_V3.pkl'
TEST_PATH  = 'data/05_embeddings_v3/test.parquet'
OUTPUT_DIR = 'src/analysis/output/interpretation'

# Кастомный трансформер (нужен для загрузки pickle)
from sklearn.base import BaseEstimator, TransformerMixin
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

def main():
    print(f"📥 Загрузка модели из {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    print(f"📥 Загрузка тестовых данных...")
    df_test = pd.read_parquet(TEST_PATH)
    df_test, bert_cols = unpack_embeddings(df_test)
    
    stylo_features =['avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
                     'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count',
                     'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio']
    
    X_test = df_test[stylo_features + bert_cols]
    y_true = df_test['label']
    
    print("🔮 Выполнение предсказаний...")
    y_pred = pipeline.predict(X_test)
    
    print("📊 Построение матрицы ошибок...")
    classes = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Нормализуем по строкам (чтобы видеть проценты вместо абсолютных чисел)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Нормализованная матрица ошибок (Hybrid V3)", fontsize=16, fontweight='bold')
    plt.ylabel("Истинный класс", fontsize=12)
    plt.xlabel("Предсказанный класс", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    out_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"✅ Матрица ошибок сохранена: {out_path}")

if __name__ == "__main__":
    main()