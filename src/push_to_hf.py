from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def main():
    print("📥 Загружаем локальные данные...")
    dataset = load_dataset(
        "parquet", 
        data_files={
            "train": "data/05_embeddings/train.parquet",
            "val": "data/05_embeddings/val.parquet",
            "test": "data/05_embeddings/test.parquet"
        }
    )

    print("🚀 Отправляем в космос (на Hugging Face)...")
    # Вставь свой никнейм!
    dataset.push_to_hub("danyasedoy/llm_attribution_hybrid", private=True)
    print("✅ Датасет успешно улетел в облако!")

if __name__ == "__main__":
    main()