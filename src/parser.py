import os
import pandas as pd
import ijson
from tqdm import tqdm

def parse_mendeley_data(base_path):
    print("🧠 Начинаем парсинг Mendeley (режим 'скальпель' с ijson)...")
    mendeley_path = os.path.join(base_path, 'individual_parts', 'individual_parts')
    
    all_dataframes = []

    # 1. Загружаем человеческие тексты с помощью ijson, чтобы избежать MemoryError
    print("  - Загружаем человеческие тексты потоково...")
    human_texts_path = os.path.join(mendeley_path, 'real_texts.json')
    
    records_buffer = []
    chunksize = 10000

    # Открываем файл и начинаем итеративно парсить JSON-массив
    with open(human_texts_path, 'rb') as f:
        # ijson.items(f, 'item') читает массив ([...]) и отдает каждый элемент ('item')
        for record in tqdm(ijson.items(f, 'item'), desc="    Processing human texts"):
            # 'record' - это один словарь, например {'text': '...', 'id': 1}
            records_buffer.append({'text': record['text'], 'label': 'human'})
            
            # Когда буфер наполняется, мы превращаем его в DataFrame и очищаем
            if len(records_buffer) == chunksize:
                all_dataframes.append(pd.DataFrame(records_buffer))
                records_buffer = [] # Сбрасываем буфер

    # Добавляем остатки из буфера, если они есть
    if records_buffer:
        all_dataframes.append(pd.DataFrame(records_buffer))

    # 2. Загружаем сгенерированные тексты (они маленькие, можно без ijson)
    ai_models = ['SberAI-small', 'SberAI-large', 'Facebook-xglm']
    for model_name in ai_models:
        print(f"  - Загружаем тексты модели: {model_name}...")
        model_path = os.path.join(mendeley_path, model_name)
        json_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
        
        model_chunks = []
        for file_name in json_files:
            file_path = os.path.join(model_path, file_name)
            try:
                temp_df = pd.read_json(file_path)
                temp_df['label'] = model_name.lower()
                model_chunks.append(temp_df[['text', 'label']])
            except Exception as e:
                print(f"    Предупреждение: не удалось прочитать {file_name}. Ошибка: {e}")

        if model_chunks:
            all_dataframes.append(pd.concat(model_chunks, ignore_index=True))

    final_mendeley_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"✅ Mendeley собран! Получено {len(final_mendeley_df)} строк.")
    return final_mendeley_df

# --- main() остается без изменений ---

def main():
    raw_data_path = os.path.join('data', '01_raw')
    output_path = os.path.join('data', '02_features')
    os.makedirs(output_path, exist_ok=True)

    mendeley_df = parse_mendeley_data(raw_data_path)

    print("\n🧥 Загружаем CoAT...")
    coat_df = pd.read_parquet(os.path.join(raw_data_path, 'coat.parquet'))
    coat_df = coat_df.rename(columns={'model': 'label'})
    coat_df = coat_df[['text', 'label']]
    print(f"✅ CoAT загружен! ({len(coat_df)} строк)")

    print("\n🦌 Загружаем Saiga...")
    saiga_df = pd.read_parquet(os.path.join(raw_data_path, 'saiga.parquet'))
    saiga_df = saiga_df.rename(columns={'output': 'text'})
    saiga_df['label'] = 'saiga'
    saiga_df = saiga_df[['text', 'label']]
    print(f"✅ Saiga загружен! ({len(saiga_df)} строк)")

    print("\n🤝 Объединяем все датасеты в один...")
    final_df = pd.concat([mendeley_df, coat_df, saiga_df], ignore_index=True)

    print("\n🧹 Проводим финальную очистку...")
    final_df.dropna(subset=['text'], inplace=True)
    final_df = final_df[final_df['text'] != '']
    final_df['text'] = final_df['text'].astype(str) # На всякий случай
    
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n📊 Итоговый баланс классов в объединенном датасете:")
    print(final_df['label'].value_counts())

    unified_data_path = os.path.join(output_path, 'unified_data.parquet')
    final_df.to_parquet(unified_data_path, engine='pyarrow')
    print(f"\n🚀 Все готово! Единый датасет сохранен в:\n{unified_data_path}")


if __name__ == "__main__":
    main()