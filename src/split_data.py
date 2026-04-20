import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_file = os.path.join('data', '02_features', 'unified_data.parquet')
    output_dir = os.path.join('data', '03_splits')
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Загрузка данных из {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"✅ Загружено {len(df):,} строк.")

    print("\n🔀 Выполняем стратифицированное разбиение (70/15/15)...")
    
    # Шаг 1: Отделяем Train (70%) и Temp (30%)
    # stratify=df['label'] 
    df_train, df_temp = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, # Фиксация seed 
        stratify=df['label']
    )

    # Шаг 2: Делим Temp пополам -> Val (15%) и Test (15%)
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=df_temp['label']
    )

    print("\n💾 Сохраняем выборки на диск...")
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'val.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    # Вывод статистики для отчета
    print("\n📊 Итоговые размеры выборок:")
    print(f"  Train: {len(df_train):,} строк ({len(df_train)/len(df):.1%})")
    print(f"  Val:   {len(df_val):,} строк ({len(df_val)/len(df):.1%})")
    print(f"  Test:  {len(df_test):,} строк ({len(df_test)/len(df):.1%})")

    # Проверка стратификации на примере самого редкого класса
    rare_class = 'gpt-4'
    print(f"\n🔍 Проверка стратификации для редкого класса '{rare_class}':")
    print(f"  В Train: {len(df_train[df_train['label'] == rare_class])}")
    print(f"  В Val:   {len(df_val[df_val['label'] == rare_class])}")
    print(f"  В Test:  {len(df_test[df_test['label'] == rare_class])}")
    
    print("\n🚀 Разбиение успешно завершено!")

if __name__ == "__main__":
    main()