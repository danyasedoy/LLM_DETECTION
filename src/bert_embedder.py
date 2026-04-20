import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────── Конфигурация ────────────────────────────────────
INPUT_DIR  = 'data/04_stylo_features'
OUTPUT_DIR = 'data/05_embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Модель от cointegrated: выдает вектор 312 размерностей. Идеально для 6GB VRAM.
MODEL_NAME = 'cointegrated/rubert-tiny2' 

# Размер пачки текстов. Для 6GB VRAM и tiny2 можно смело ставить 128 или 256.
BATCH_SIZE = 128 
MAX_LENGTH = 512 # Максимальная длина контекста для BERT

def get_embeddings(text_batch: list, tokenizer, model, device) -> np.ndarray:
    """Прогоняет пачку текстов через BERT и возвращает numpy-матрицу эмбеддингов"""
    
    # Токенизация с динамическим паддингом (дополняем нулями до самого длинного В ЭТОМ батче)
    encoded_input = tokenizer(
        text_batch, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        return_tensors='pt'
    ).to(device) # Переносим токены на видеокарту

    # Отключаем расчет градиентов! (Критически важно для экономии видеопамяти)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Для rubert-tiny2 эмбеддингом текста считается токен [CLS] (самый первый токен с индексом 0)
    embeddings = model_output.last_hidden_state[:, 0, :]
    
    # L2-нормализация векторов (улучшает работу классификаторов)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Переносим результат обратно на процессор и превращаем в numpy
    return embeddings.cpu().numpy()

def process_file(file_name: str, tokenizer, model, device):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    if not os.path.exists(input_path):
        print(f"❌ Файл {input_path} не найден. Пропускаем.")
        return

    print(f"\n📂 Читаем данные из {input_path}...")
    df = pd.read_parquet(input_path)
    texts = df['text'].tolist()
    
    all_embeddings =[]
    
    print(f"🧠 Генерируем эмбеддинги на {device.type.upper()} (Батч: {BATCH_SIZE})...")
    # tqdm для красивого прогресс-бара
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Обработка {file_name}"):
        batch = texts[i : i + BATCH_SIZE]
        
        # Получаем векторы
        emb_matrix = get_embeddings(batch, tokenizer, model, device)
        
        # Разбиваем матрицу на список отдельных векторов и добавляем в общий список
        all_embeddings.extend(list(emb_matrix))
    
    # 🔗 СВЯЗЫВАНИЕ ДАННЫХ: 
    # Просто добавляем список векторов как новую колонку к исходному датафрейму!
    # Порядок строк строго сохранен.
    df['bert_embedding'] = all_embeddings
    
    print(f"💾 Сохраняем обогащенный датасет в {output_path}...")
    df.to_parquet(output_path, engine='pyarrow')
    
    # Очистка памяти после каждого файла
    del df, texts, all_embeddings
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    # Проверяем наличие видеокарты
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Инициализация. Устройство: {device}")
    if device.type == 'cuda':
        print(f"🎮 Видеокарта: {torch.cuda.get_device_name(0)}")
        print(f"💾 Доступно VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\n📦 Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Переводим модель в режим предсказания (отключаем Dropout и т.д.)

    # Обрабатываем файлы по очереди
    for file_name in ['test.parquet', 'val.parquet', 'train.parquet']:
        process_file(file_name, tokenizer, model, device)
        
    print("\n🎉 ЭТАП 1 ПОЛНОСТЬЮ ЗАВЕРШЕН! Все данные готовы к A/B/C тестированию моделей!")

if __name__ == "__main__":
    main()