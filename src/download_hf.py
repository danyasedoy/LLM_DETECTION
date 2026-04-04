import os
import pandas as pd
from datasets import load_dataset

def download_and_save():
    raw_data_dir = os.path.join('data', '01_raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    # 1. Скачиваем CoAT (он без скриптов, качается штатно)
    coat_path = os.path.join(raw_data_dir, 'coat.parquet')
    if not os.path.exists(coat_path):
        print("📥 Скачиваем датасет CoAT...")
        coat_dataset = load_dataset("RussianNLP/coat", "authorship", split="train") 
        coat_df = coat_dataset.to_pandas()
        coat_df.to_parquet(coat_path, engine='pyarrow')
        print(f"✅ CoAT сохранен! ({len(coat_df)} строк)")
    else:
        print("✅ CoAT уже скачан, пропускаем.")

    # 2. Скачиваем Saiga (в обход запретов Hugging Face напрямую по URL)
    saiga_path = os.path.join(raw_data_dir, 'saiga.parquet')
    print("\n📥 Скачиваем датасет Saiga по прямой ссылке...")
    
    # Прямая ссылка на сырой сжатый файл в репозитории
    url = "https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga/resolve/main/ru_turbo_saiga.jsonl.zst"
    
    # Pandas сам скачает, распакует zstd-архив и прочитает JSON по строкам
    saiga_df = pd.read_json(url, lines=True, compression='zstd')
    
    saiga_df.to_parquet(saiga_path, engine='pyarrow')
    print(f"✅ Saiga сохранен! ({len(saiga_df)} строк) -> {saiga_path}")

if __name__ == "__main__":
    download_and_save()
    print("\n🚀 Все данные успешно собраны!")