"""
Построение PR-кривых (Precision-Recall) для отчета
Лабораторная работа №5
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = 'models/final_v3/Ultimate_Hybrid_V3.pkl'
TEST_PATH  = 'data/05_embeddings_v4/test.parquet'
OUTPUT_DIR = 'src/analysis/output/interpretation'

# Кастомный трансформер
class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): pass
    def transform(self, X): return np.clip(X, self.lb_, self.ub_)

def unpack_embeddings(df):
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    return pd.concat([df.drop(columns=['bert_embedding']), emb_df], axis=1), emb_cols

def main():
    print(f"📥 Загрузка модели и данных...")
    pipeline = joblib.load(MODEL_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    df_test, bert_cols = unpack_embeddings(df_test)
    
    stylo_features =['avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
                     'upper_ratio', 'digit_ratio', 'stopword_ratio',
                     'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio']
    
    X_test = df_test[stylo_features + bert_cols]
    y_true = df_test['label']
    
    print("🔮 Получение вероятностей (predict_proba)...")
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_
    
    # Биннаризуем метки для расчета кривых
    y_test_bin = label_binarize(y_true, classes=classes)
    
    # Выбираем самые интересные классы для графика
    target_classes =['human', 'gpt-4', 'facebook_xglm', 'sberai_large']
    colors =['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    plt.figure(figsize=(10, 8))
    
    print("📊 Отрисовка PR-кривых...")
    for target_class, color in zip(target_classes, colors):
        class_idx = list(classes).index(target_class)
        
        # Считаем Precision и Recall
        precision, recall, _ = precision_recall_curve(y_test_bin[:, class_idx], y_proba[:, class_idx])
        # Считаем площадь под кривой (Average Precision)
        ap = average_precision_score(y_test_bin[:, class_idx], y_proba[:, class_idx])
        
        plt.plot(recall, precision, color=color, lw=2, 
                 label=f'{target_class} (AUC-PR = {ap:.2f})')
                 
    plt.xlabel('Полнота (Recall)', fontsize=12)
    plt.ylabel('Точность (Precision)', fontsize=12)
    plt.title('PR-кривые (Precision-Recall Curve) для ключевых классов', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = os.path.join(OUTPUT_DIR, 'pr_curves.png')
    plt.savefig(out_path, dpi=150)
    print(f"✅ График PR-кривых сохранен: {out_path}")

if __name__ == "__main__":
    main()