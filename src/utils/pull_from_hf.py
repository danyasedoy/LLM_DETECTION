import os
from datasets import load_dataset

def main():
    REPO_ID = "danyasedoy/llm_attribution_hybrid"
    OUTPUT_DIR = os.path.join("data", "05_embeddings")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📥 Подключаемся к Hugging Face и ищем датасет {REPO_ID}...")
    
    try:
        dataset = load_dataset(REPO_ID)
    except Exception as e:
        print(f"❌ Ошибка загрузки! Убедись, что ты залогинился (huggingface-cli login) на этом ПК.")
        print(f"Текст ошибки: {e}")
        return

    print("✅ Датасет скачан в кэш! Экспортируем в локальные Parquet-файлы...")
    
    for split in ['train', 'val', 'test']:
        if split in dataset:
            output_path = os.path.join(OUTPUT_DIR, f"{split}.parquet")
            print(f"  💾 Сохраняем {split}.parquet ({len(dataset[split]):,} строк)...")
            dataset[split].to_parquet(output_path)
            
    print(f"\n🚀 Все данные успешно развернуты локально в: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()