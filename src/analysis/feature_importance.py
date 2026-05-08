import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = 'models/final_v3/Ultimate_Hybrid_V3.pkl'
OUTPUT_DIR = 'src/analysis/output/interpretation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 11 честных признаков!
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio',
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]
BERT_FEATURES =[f'bert_{i}' for i in range(768)]
ALL_FEATURE_NAMES = STYLO_FEATURES + BERT_FEATURES

class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lb_ = np.quantile(X, 0.0, axis=0)
        self.ub_ = np.quantile(X, 0.99, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

def plot_top_features(df_imp):
    top_20 = df_imp.head(20)
    colors =['#4C72B0' if 'bert' in feat else '#DD8452' for feat in top_20['Feature']]
    plt.figure(figsize=(12, 8))
    # Обновленный синтаксис Seaborn (без ворнинга)
    sns.barplot(x='Importance', y='Feature', data=top_20, hue='Feature', palette=colors, legend=False)
    plt.title('ТОП-20 самых важных признаков (Очищенный Hybrid)', fontsize=14, fontweight='bold')
    plt.xlabel('Важность признака (%)', fontsize=12)
    plt.ylabel('Признак', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements =[Patch(facecolor='#4C72B0', label='Семантика (BERT)'), Patch(facecolor='#DD8452', label='Стилометрия')]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_20_features_clean.png'), dpi=150)
    plt.close()

def plot_group_importance(df_imp):
    stylo_importance = df_imp[df_imp['Feature'].isin(STYLO_FEATURES)]['Importance'].sum()
    bert_importance = df_imp[~df_imp['Feature'].isin(STYLO_FEATURES)]['Importance'].sum()
    
    plt.figure(figsize=(8, 8))
    plt.pie([stylo_importance, bert_importance], explode=(0.1, 0), 
            labels=['Стилометрия\n(11 признаков)', 'Семантика\n(768 признаков)'], 
            colors=['#DD8452', '#4C72B0'], autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 14})
    plt.title('Вклад групп признаков (Очищенная модель)', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'group_importance_clean.png'), dpi=150)
    plt.close()

def main():
    pipeline = joblib.load(MODEL_PATH)
    catboost_model = pipeline.named_steps['clf']
    importances = catboost_model.get_feature_importance()
    
    df_imp = pd.DataFrame({'Feature': ALL_FEATURE_NAMES, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plot_top_features(df_imp)
    plot_group_importance(df_imp)
    
    print("\n🏆 ТОП-10 САМЫХ СИЛЬНЫХ ПРИЗНАКОВ (БЕЗ ЧИТЕРСТВА):")
    print(df_imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()