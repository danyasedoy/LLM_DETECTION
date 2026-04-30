import os
import string
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True) 

class AdvancedStylometryExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy3.MorphAnalyzer()

    def extract_features(self, text: str) -> dict:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return self._empty_features()

        chars_count = len(text)
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in string.punctuation or c in '«»—')
        
        sentences = sent_tokenize(text, language='russian')
        words =[w.lower() for w in word_tokenize(text, language='russian') if w.isalpha()]
        
        sentence_count = max(len(sentences), 1)
        word_count = max(len(words), 1)
        unique_words = set(words)
        stop_words_count = sum(1 for w in words if w in self.stop_words)
        total_word_length = sum(len(w) for w in words)

        # МОРФОЛОГИЯ 
        nouns = verbs = adjs = conjs = 0
        for w in words:
            pos = self.morph.parse(w)[0].tag.POS
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

    def _empty_features(self) -> dict:
        return {
            'avg_word_len': 0.0, 'avg_sentence_len': 0.0, 'ttr': 0.0, 'punct_ratio': 0.0,
            'upper_ratio': 0.0, 'digit_ratio': 0.0, 'stopword_ratio': 0.0, 'word_count': 0,
            'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0, 'conj_ratio': 0.0
        }

def process_file(input_path: str, output_path: str):
    print(f"\n📂 Считаем расширенную стилометрию для: {input_path}")
    df = pd.read_parquet(input_path)
    extractor = AdvancedStylometryExtractor()
    tqdm.pandas(desc="Извлечение")
    features_df = df['text'].progress_apply(lambda x: pd.Series(extractor.extract_features(x)))
    result_df = pd.concat([df, features_df], axis=1)
    result_df.to_parquet(output_path)
    print(f"✅ Сохранено в: {output_path}")

if __name__ == "__main__":
    out_dir = 'data/04_stylo_v3'
    os.makedirs(out_dir, exist_ok=True)
    for f in ['test.parquet', 'val.parquet', 'train.parquet']:
        process_file(os.path.join('data/03_splits', f), os.path.join(out_dir, f))