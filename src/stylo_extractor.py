import os
import re
import string
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True) 

class StylometryExtractor:
    def __init__(self):
        # Загружаем русские стоп-слова
        self.stop_words = set(stopwords.words('russian'))
        # Регулярка для поиска эмодзи/спецсимволов 
        self.emoji_pattern = re.compile(r'[^\w\s,\.\?!;:\'"\(\)\-«»]')

    def extract_features(self, text: str) -> dict:
        """Извлекает стилометрический вектор из одного текста."""
        
        # Если текст пустой или это NaN
        if not isinstance(text, str) or len(text.strip()) == 0:
            return self._empty_features()

        # 1. Базовые счетчики
        chars_count = len(text)
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in string.punctuation or c in '«»—')
        
        # Токенизация (разбивка на предложения и слова)
        sentences = sent_tokenize(text, language='russian')
        words =[w.lower() for w in word_tokenize(text, language='russian') if w.isalpha()]
        
        sentence_count = max(len(sentences), 1)
        word_count = max(len(words), 1)
        
        # 2. Сложные метрики
        unique_words = set(words)
        stop_words_count = sum(1 for w in words if w in self.stop_words)
        
        # Считаем суммарную длину всех слов для среднего
        total_word_length = sum(len(w) for w in words)

        # 3. Формируем словарь нормализованных (!) признаков
        features = {
            # Лингвистические (по статье Федотовой)
            'avg_word_len': total_word_length / word_count,
            'avg_sentence_len': word_count / sentence_count,
            'ttr': len(unique_words) / word_count, # Коэффициент лексического разнообразия
            
            # Нормированные частоты (статистические)
            'punct_ratio': punct_count / chars_count,
            'upper_ratio': upper_count / chars_count,
            'digit_ratio': digit_count / chars_count,
            'stopword_ratio': stop_words_count / word_count,
            
            # Абсолютные значения (оставим, но на Шаге 6, возможно, выкинем)
            'word_count': word_count,
            'sentence_count': sentence_count
        }
        
        return features

    def _empty_features(self) -> dict:
        """Возвращает нули, если текст битый"""
        return {
            'avg_word_len': 0.0, 'avg_sentence_len': 0.0, 'ttr': 0.0,
            'punct_ratio': 0.0, 'upper_ratio': 0.0, 'digit_ratio': 0.0, 
            'stopword_ratio': 0.0, 'word_count': 0, 'sentence_count': 0
        }

def process_file(input_path: str, output_path: str):
    print(f"\n📂 Обработка файла: {input_path}")
    df = pd.read_parquet(input_path)
    
    extractor = StylometryExtractor()
    
    tqdm.pandas(desc="Извлечение признаков")
    
    print("⏳ Считаем стилометрию...")
    features_df = df['text'].progress_apply(lambda x: pd.Series(extractor.extract_features(x)))
    
    result_df = pd.concat([df, features_df], axis=1)
    
    result_df.to_parquet(output_path)
    print(f"✅ Готово! Сохранено в: {output_path}")

def main():
    splits_dir = os.path.join('data', '03_splits')
    output_dir = os.path.join('data', '04_stylo_features')
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in['test.parquet', 'val.parquet', 'train.parquet']:
        input_file = os.path.join(splits_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"❌ Файл {input_file} не найден. Сначала запустите split_data.py!")

if __name__ == "__main__":
    main()