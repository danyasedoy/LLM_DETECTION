import os
import gc
import pandas as pd
import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
import numpy as np

def write_chunk(df, writer):
    """Очищает кусочек данных и сразу пишет его в Parquet-файл на диске."""
    df.dropna(subset=['text'], inplace=True)

    df = df[df['text'] != ''].copy()
    df = df[df['text'].str.len() >= 50].copy()
    if len(df) == 0:
        return
        
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str).str.lower()
    
    table = pa.Table.from_pandas(df[['text', 'label', 'source']], schema=writer.schema, preserve_index=False)
    writer.write_table(table)

def extract_bot_replies(messages):
    """Вытаскивает ответы бота из диалогов Saiga"""
    target_roles = {'bot', 'assistant', 'model', 'gpt'}
    
    try:
        if isinstance(messages, dict) and 'role' in messages and 'content' in messages:
            roles = messages['role']
            contents = messages['content']
            
            if isinstance(roles, np.ndarray): roles = roles.tolist()
            if isinstance(contents, np.ndarray): contents = contents.tolist()
                
            return "\n".join([str(c) for r, c in zip(roles, contents) if str(r).lower() in target_roles])
            
        elif isinstance(messages, list) or isinstance(messages, np.ndarray):
            return "\n".join([str(msg.get('content', '')) for msg in messages if str(msg.get('role', '')).lower() in target_roles])
            
    except Exception as e:
        return ""
        
    return ""

def main():
    raw_data_path = os.path.join('data', '01_raw')
    output_path = os.path.join('data', '02_features')
    os.makedirs(output_path, exist_ok=True)
    
    temp_parquet_path = os.path.join(output_path, 'unified_temp.parquet')
    final_parquet_path = os.path.join(output_path, 'unified_data.parquet')

    schema = pa.schema([
        ('text', pa.string()),
        ('label', pa.string()),
        ('source', pa.string()) 
    ])
    
    print("🚀 Начинаем потоковую сборку (Out-of-Core)...")
    with pq.ParquetWriter(temp_parquet_path, schema) as writer:
        
        # --- ШАГ 1: Менделеев (Сборка по кускам) ---
        mendeley_path = os.path.join(raw_data_path, 'individual_parts')
        
        # 1.1 Человеческие тексты
        print("\n🧠 Парсим Mendeley: Human...")
        human_texts_path = os.path.join(mendeley_path, 'real_texts.json')
        records =[]
        with open(human_texts_path, 'rb') as f:
            for item in tqdm(ijson.items(f, 'item'), desc="  Human"):
                records.append({'text': item, 'label': 'human', 'source': 'mendeley'})
                if len(records) >= 10000:
                    write_chunk(pd.DataFrame(records), writer)
                    records =[]
        if records: write_chunk(pd.DataFrame(records), writer)
        
        # 1.2 Тексты нейросетей
        ai_models =['SberAI-small', 'SberAI-large', 'Facebook-xglm']
        for model in ai_models:
            print(f"🧠 Парсим Mendeley: {model}...")
            model_folder = os.path.join(mendeley_path, model)
            if not os.path.exists(model_folder): continue
            
            for file_name in os.listdir(model_folder):
                if not file_name.endswith('.json'): continue
                
                records =[]
                with open(os.path.join(model_folder, file_name), 'rb') as f:
                    label = model.lower().replace('-', '_')
                    for item in tqdm(ijson.items(f, 'item'), desc=f"  {file_name}"):
                        records.append({'text': item, 'label': label, 'source': 'mendeley'})
                        if len(records) >= 10000:
                            write_chunk(pd.DataFrame(records), writer)
                            records =[]
                if records: write_chunk(pd.DataFrame(records), writer)

        # --- ШАГ 2: CoAT ---
        print("\n🧥 Добавляем CoAT...")
        coat_df = pd.read_parquet(os.path.join(raw_data_path, 'coat.parquet'))
        coat_df = coat_df.rename(columns={'model': 'label'})
        coat_df['source'] = 'coat'
        write_chunk(coat_df, writer)
        del coat_df
        gc.collect()

        # --- ШАГ 3: Saiga ---
        print("\n🦌 Добавляем Saiga...")
        saiga_df = pd.read_parquet(os.path.join(raw_data_path, 'saiga.parquet'))
        saiga_df['text'] = saiga_df['messages'].apply(extract_bot_replies)
        saiga_df['source'] = 'saiga'
        saiga_df['label'] = saiga_df['model_name'].fillna('gpt-3.5-turbo')
        
        write_chunk(saiga_df, writer)
        del saiga_df
        gc.collect()

    print(f"\n✅ Сырой датасет собран на диске: {temp_parquet_path}")

    # --- ШАГ 4: Out-of-Core Перемешивание ---
    print("\n🔀 Начинаем перемешивание 1.1М строк без загрузки в RAM...")
    
    dataset = load_dataset("parquet", data_files=temp_parquet_path, split="train")
    shuffled_dataset = dataset.shuffle(seed=42)
    
    shuffled_dataset.to_parquet(final_parquet_path)
    
    os.remove(temp_parquet_path)
    
    labels = shuffled_dataset['label']
    print("\n📊 Итоговый баланс классов в объединенном датасете:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

    print(f"\n🚀 ВСЕ ГОТОВО! датасет сохранен в:\n{final_parquet_path}")

if __name__ == "__main__":
    main()