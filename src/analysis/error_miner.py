import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = 'models/final_v3/Ultimate_Hybrid_V3.pkl'
TEST_PATH  = 'data/05_embeddings_v4/test.parquet'
OUTPUT_DIR = 'src/analysis/output/interpretation'

STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]

class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): pass
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

def unpack_embeddings(df):
    emb_matrix = np.vstack(df['bert_embedding'].values)
    emb_cols =[f'bert_{i}' for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index, dtype=np.float32)
    return pd.concat([df.drop(columns=['bert_embedding']), emb_df], axis=1), emb_cols

def main():
    print(f"📥 Загрузка данных и модели...")
    pipeline = joblib.load(MODEL_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    df_test_unpacked, bert_cols = unpack_embeddings(df_test.copy())
    
    X_test = df_test_unpacked[STYLO_FEATURES + bert_cols]
    y_true = df_test_unpacked['label']
    
    print("🔮 Выполнение предсказаний...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # 1. МАТРИЦА ОШИБОК
    classes = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Нормализованная матрица ошибок (Очищенный Hybrid V3)", fontsize=16, fontweight='bold')
    plt.ylabel("Истинный класс", fontsize=12)
    plt.xlabel("Предсказанный класс", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_clean.png'), dpi=150)
    plt.close()
    
    # 2. ПОИСК АНОМАЛЬНЫХ ОШИБОК (High Confidence Errors)
    print("\n" + "🕵️‍♂️"*15)
    print("ПОИСК ИНТЕРЕСНЫХ ОШИБОК (ДЛЯ ОТЧЕТА)")
    print("🕵️‍♂️"*15)
    
    df_test['pred'] = y_pred.flatten()
    df_test['confidence'] = np.max(y_proba, axis=1)
    
    # Ищем, где ошиблась, но была уверена больше чем на 80%
    errors = df_test[(df_test['label'] != df_test['pred']) & (df_test['confidence'] > 0.80)]
    
    # Берем топ-5 самых "уверенных" ошибок
    top_errors = errors.sort_values(by='confidence', ascending=False).head(5)
    
    for idx, row in top_errors.iterrows():
        print(f"\n❌ Истинный автор: {row['label'].upper()}")
        print(f"🤖 Предсказано как: {row['pred'].upper()} (Уверенность: {row['confidence']:.1%})")
        print(f"📄 Текст (первые 300 символов):\n{row['text'][:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()