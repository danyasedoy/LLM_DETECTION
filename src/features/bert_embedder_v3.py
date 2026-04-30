import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

INPUT_DIR  = 'data/04_stylo_v3'
OUTPUT_DIR = 'data/05_embeddings_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = 'DeepPavlov/rubert-base-cased'
BATCH_SIZE = 64 
MAX_LENGTH = 512

def get_embeddings(text_batch, tokenizer, model, device):
    encoded_input = tokenizer(
        text_batch, padding=True, truncation=True, 
        max_length=MAX_LENGTH, return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def process_file(file_name, tokenizer, model, device):
    in_path, out_path = os.path.join(INPUT_DIR, file_name), os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(out_path):
        print(f"\n✅ Файл {out_path} уже готов! Пропускаем, экономим время.")
        return
    print(f"\n📂 Извлечение BERT-base для: {in_path}")
    df = pd.read_parquet(in_path)
    texts = df['text'].tolist()
    
    all_embeddings =[]
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        all_embeddings.extend(list(get_embeddings(batch, tokenizer, model, device)))
    
    df['bert_embedding'] = all_embeddings
    df.to_parquet(out_path, engine='pyarrow')
    del df, texts, all_embeddings; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Устройство: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    for f in['test.parquet', 'val.parquet', 'train.parquet']:
        process_file(f, tokenizer, model, device)