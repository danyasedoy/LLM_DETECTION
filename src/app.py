import os
import torch
import string
import joblib
import pandas as pd
import numpy as np
import gradio as gr

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3
from transformers import AutoTokenizer, AutoModel

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── ИНИЦИАЛИЗАЦИЯ ───────────────────────────────────
print("⏳ Загрузка библиотек и моделей (это займет около 10 секунд)...")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 1. Загружаем стилометрию
STOP_WORDS = set(stopwords.words('russian'))
MORPH = pymorphy3.MorphAnalyzer()
STYLO_FEATURES =[
    'avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 
    'upper_ratio', 'digit_ratio', 'stopword_ratio', 
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio'
]

# 2. Загружаем семантику (RuBERT)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
BERT_MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
BERT_MODEL.eval()
BERT_FEATURES =[f'bert_{i}' for i in range(768)]

# 3. Загружаем наш CatBoost (Hybrid V3)
MODEL_PATH = 'models/final_v3/Ultimate_Hybrid_V3.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

# Класс Winsorizer нужен для успешной загрузки pickle
from sklearn.base import BaseEstimator, TransformerMixin
class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): pass
    def transform(self, X):
        return np.clip(X, self.lb_, self.ub_)

PIPELINE = joblib.load(MODEL_PATH)
print("✅ Система готова к работе!")

# ─────────────────────────── БИЗНЕС-ЛОГИКА ───────────────────────────────────

def extract_stylometry(text: str) -> dict:
    """Извлекает 12 стилометрических признаков"""
    chars_count = max(len(text), 1)
    upper_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    punct_count = sum(1 for c in text if c in string.punctuation or c in '«»—')
    
    sentences = sent_tokenize(text, language='russian')
    words =[w.lower() for w in word_tokenize(text, language='russian') if w.isalpha()]
    
    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)
    unique_words = set(words)
    stop_words_count = sum(1 for w in words if w in STOP_WORDS)
    total_word_length = sum(len(w) for w in words)

    nouns = verbs = adjs = conjs = 0
    for w in words:
        pos = MORPH.parse(w)[0].tag.POS
        if pos == 'NOUN': nouns += 1
        elif pos in ('VERB', 'INFN'): verbs += 1
        elif pos in ('ADJF', 'ADJS'): adjs += 1
        elif pos in ('CONJ', 'PREP'): conjs += 1

    return {
        'avg_word_len': total_word_length / word_count,
        'avg_sentence_len': word_count / sentence_count,
        'ttr': len(unique_words) / word_count,
        'punct_ratio': punct_count / chars_count,
        'upper_ratio': upper_count / chars_count,
        'digit_ratio': digit_count / chars_count,
        'stopword_ratio': stop_words_count / word_count,
        'word_count': word_count,
        'noun_ratio': nouns / word_count,
        'verb_ratio': verbs / word_count,
        'adj_ratio': adjs / word_count,
        'conj_ratio': conjs / word_count
    }

def get_bert_embedding(text: str) -> list:
    """Извлекает вектор из 768 чисел"""
    encoded_input = TOKENIZER(
        [text], padding=True, truncation=True, max_length=512, return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        output = BERT_MODEL(**encoded_input)
    
    embedding = output.last_hidden_state[:, 0, :]
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()[0].tolist()

def analyze_text(text: str):
    """Главная функция для Gradio-интерфейса"""
    if len(text.strip()) < 50:
        return {"Ошибка": 1.0}, "❌ Текст слишком короткий для анализа (минимум 50 символов)."

    # 1. Извлечение признаков
    stylo_dict = extract_stylometry(text)
    bert_list = get_bert_embedding(text)
    
    # Собираем DataFrame (1 строка, 780 колонок)
    row_data = stylo_dict.copy()
    for i, val in enumerate(bert_list):
        row_data[f'bert_{i}'] = val
        
    df_input = pd.DataFrame([row_data], columns=STYLO_FEATURES + BERT_FEATURES)
    
    # 2. Предсказание вероятностей (predict_proba)
    probabilities = PIPELINE.predict_proba(df_input)[0]
    classes = PIPELINE.classes_
    
    # Формируем словарь для красивого графика в Gradio
    result_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    
    # 3. Блок ИНТЕРПРЕТАЦИИ (Объяснение результата аналитику)
    explanation = f"""
    ### 🕵️‍♂️ Анализ стилометрического профиля текста
    * **Лексическое разнообразие (TTR):** {stylo_dict['ttr']:.2f} *(У ИИ часто < 0.6)*
    * **Средняя длина предложения:** {stylo_dict['avg_sentence_len']:.1f} слов
    * **Доля знаков препинания:** {stylo_dict['punct_ratio']:.2%}
    * **Доля существительных:** {stylo_dict['noun_ratio']:.2%} *(У ИИ часто избыток)*
    * **Доля глаголов:** {stylo_dict['verb_ratio']:.2%}
    * **Количество слов:** {stylo_dict['word_count']}
    
    *Модель проанализировала эти 12 лингвистических параметров совместно с глубоким семантическим смыслом текста (768 признаков RuBERT) для вынесения итогового вердикта.*
    """
    
    return result_dict, explanation

# ─────────────────────────── ИНТЕРФЕЙС GRADIO ────────────────────────────────

# Настраиваем визуальную часть
with gr.Blocks(title="LLM Detector", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🤖 Гибридный детектор сгенерированных текстов (LLM Attribution)")
    gr.Markdown("Вставьте текст (от 50 символов), чтобы определить вероятность его генерации нейросетью или человеком.")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(lines=10, label="Текст для анализа", placeholder="Введите текст сюда...")
            analyze_btn = gr.Button("Анализировать текст", variant="primary")
            
        with gr.Column(scale=1):
            label_output = gr.Label(label="Вердикт модели", num_top_classes=6)
            
    with gr.Row():
        explanation_output = gr.Markdown(label="Интерпретация признаков")

    # Связываем кнопку с логикой
    analyze_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[label_output, explanation_output]
    )

if __name__ == "__main__":
    print("\n🚀 Запуск веб-интерфейса! Перейдите по ссылке ниже:")
    interface.launch(inbrowser=True)