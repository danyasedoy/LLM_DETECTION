import os
import gc
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3
import multiprocessing

# ─────────────────────────── Конфигурация ────────────────────────────────────
INPUT_DIR  = 'data/05_embeddings_v3'
OUTPUT_DIR = 'data/05_embeddings_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Оставляем 1 ядро для операционной системы, чтобы комп не завис
NUM_CORES = multiprocessing.cpu_count() - 1

# --- (Здесь оставь класс MattrStylometryExtractor и функцию _empty_features без изменений) ---
class MattrStylometryExtractor:
    def __init__(self, window_size=50):
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy3.MorphAnalyzer()
        self.window_size = window_size
    def _calculate_mattr(self, words: list) -> float:
        n = len(words)
        if n < self.window_size:
            return len(set(words)) / n if n > 0 else 0.0
        ttr_sum, num_windows = 0.0, n - self.window_size + 1
        for i in range(num_windows):
            ttr_sum += len(set(words[i:i+self.window_size])) / self.window_size
        return ttr_sum / num_windows
    def extract_features(self, text: str) -> dict:
        if not isinstance(text, str) or len(text.strip()) == 0: return self._empty_features()
        chars_count = len(text)
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in string.punctuation or '«»—')
        sentences = sent_tokenize(text, language='russian')
        words =[w.lower() for w in word_tokenize(text, language='russian') if w.isalpha()]
        sentence_count = max(len(sentences), 1)
        word_count = max(len(words), 1)
        stop_words_count = sum(1 for w in words if w in self.stop_words)
        total_word_length = sum(len(w) for w in words)
        nouns = verbs = adjs = conjs = 0
        for w in words:
            pos = self.morph.parse(w)[0].tag.POS
            if pos == 'NOUN': nouns += 1
            elif pos in ('VERB', 'INFN'): verbs += 1
            elif pos in ('ADJF', 'ADJS'): adjs += 1
            elif pos in ('CONJ', 'PREP'): conjs += 1
        return {'avg_word_len': total_word_length/word_count, 'avg_sentence_len': word_count/sentence_count,
                'mattr': self._calculate_mattr(words), 'punct_ratio': punct_count/chars_count,
                'upper_ratio': upper_count/chars_count, 'digit_ratio': digit_count/chars_count,
                'stopword_ratio': stop_words_count/word_count, 'noun_ratio': nouns/word_count,
                'verb_ratio': verbs/word_count, 'adj_ratio': adjs/word_count, 'conj_ratio': conjs/word_count}
    def _empty_features(self) -> dict: return {k: 0.0 for k in ['avg_word_len', 'avg_sentence_len', 'mattr', 'punct_ratio', 'upper_ratio', 'digit_ratio', 'stopword_ratio', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio']}

# ─────────────────────────── ЛОГИКА ДЛЯ ОДНОГО ЯДРА ────────────────────────────
def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Функция, которую будет выполнять каждый отдельный процессор"""
    extractor = MattrStylometryExtractor(window_size=50)
    features_df = chunk_df['text'].apply(lambda x: pd.Series(extractor.extract_features(x)))
    return pd.concat([chunk_df, features_df], axis=1)

# ─────────────────────────── ГЛАВНАЯ ЛОГИКА ──────────────────────────────────
def main():
    print(f"🚀 Старт МНОГОПОТОЧНОЙ очистки на {NUM_CORES} ядрах...")
    
    for file_name in['test.parquet', 'val.parquet', 'train.parquet']:
        in_path = os.path.join(INPUT_DIR, file_name)
        out_path = os.path.join(OUTPUT_DIR, file_name)
        
        if os.path.exists(out_path):
            print(f"\n✅ Файл {out_path} уже готов! Пропускаем.")
            continue
            
        print(f"\n📂 Обработка файла: {in_path}")
        df = pd.read_parquet(in_path)
        
        # Удаляем старые, ненужные стило-колонки
        old_cols =[
            'avg_word_len', 'avg_sentence_len', 'ttr', 'punct_ratio', 
            'upper_ratio', 'digit_ratio', 'stopword_ratio', 'word_count',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'conj_ratio', 'sentence_count'
        ]
        cols_to_drop =[c for c in old_cols if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Делим датафрейм на куски для каждого ядра
        df_chunks = np.array_split(df, NUM_CORES)
        
        print(f"⏳ Распределяем {len(df):,} строк на {NUM_CORES} потоков...")
        with multiprocessing.Pool(NUM_CORES) as pool:
            # pool.map отправляет по одному чанку в каждый процесс
            # tqdm здесь покажет прогресс по чанкам, а не по строкам
            results = list(tqdm(pool.imap(process_chunk, df_chunks), total=len(df_chunks), desc=f"Обработка {file_name}"))
            
        print("🧩 Сборка результатов...")
        final_df = pd.concat(results)
        
        final_df.to_parquet(out_path)
        print(f"✅ Сохранено в: {out_path}")
        del df, df_chunks, results, final_df; gc.collect()

    print("\n🎉 ГОТОВО! Новые данные лежат в data/05_embeddings_v4/")


if __name__ == "__main__":
    # NLTK и PyMorphy нужно инициализировать один раз до запуска потоков
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    main()