import os
import string
import pandas as pd
from tqdm import tqdm

def count_correct_punct(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    # ВОТ ОНА ПРАВИЛЬНАЯ СТРОЧКА: добавлено "c in"
    punct_count = sum(1 for c in text if c in string.punctuation or c in '«»—')
    return punct_count / len(text)

def main():
    folder = 'data/05_embeddings_v4'
    for file_name in['test.parquet', 'val.parquet', 'train.parquet']:
        path = os.path.join(folder, file_name)
        print(f"🔧 Чиним пунктуацию в {file_name}...")
        
        df = pd.read_parquet(path)
        tqdm.pandas(desc="Пересчет")
        df['punct_ratio'] = df['text'].progress_apply(count_correct_punct)
        
        df.to_parquet(path, engine='pyarrow')
        print(f"✅ Готово: {file_name}")

if __name__ == "__main__":
    main()