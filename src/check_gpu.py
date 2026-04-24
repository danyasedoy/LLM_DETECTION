import torch

if torch.cuda.is_available():
    print("✅ Ура! PyTorch видит твою видеокарту (GPU)!")
    print(f"   Название: {torch.cuda.get_device_name(0)}")
    print(f"   Версия CUDA: {torch.version.cuda}")
else:
    print("❌ Проблема! PyTorch не видит видеокарту и работает на CPU.")
    print("   Выполни Шаг 3 (переустановка PyTorch с поддержкой CUDA).")