# 🤖 LLM Detection — Руководство по воспроизведению эксперимента

> Полный пайплайн: от сырых данных до задеплоенного Gradio-приложения.  
> Все команды запускаются из **корня проекта** (`LLM_DETECTION/`).

---

## Структура проекта

```
LLM_DETECTION/
├── data/
│   ├── 01_raw/                  # Сырые данные (Mendeley, CoAT, Saiga)
│   ├── 02_features/             # Объединённый датасет (unified_data.parquet)
│   ├── 03_splits/               # train / val / test без признаков
│   ├── 04_stylo_features/       # Стилометрия v1 (8 признаков)
│   ├── 04_stylo_v3/             # Стилометрия v3 (12 признаков + морфология)
│   ├── 05_embeddings/           # BERT tiny2 (312 dim) + стилометрия v1
│   └── 05_embeddings_v3/        # BERT base (768 dim) + стилометрия v3  ← финал
├── models/
│   ├── final/                   # Итерация 1: CatBoost CPU, 19 классов
│   ├── final_v2/                # Итерация 2: CatBoost GPU, 6 семейств
│   └── final_v3/                # Итерация 3+4: финальная модель  ← финал
├── src/
│   ├── data/                    # Сбор и подготовка данных
│   │   ├── download_hf.py       # Скачать CoAT и Saiga с Hugging Face
│   │   ├── parser.py            # Парсинг и сборка unified_data.parquet
│   │   └── split_data.py        # Стратифицированное разбиение 70/15/15
│   ├── features/                # Инжиниринг признаков
│   │   ├── stylo_extractor.py      # Стилометрия v1 (8 признаков)
│   │   ├── stylo_extractor_v3.py   # Стилометрия v3 (12 + морфология) ← финал
│   │   ├── bert_embedder.py        # Эмбеддинги rubert-tiny2 (312 dim)
│   │   └── bert_embedder_v3.py     # Эмбеддинги rubert-base-cased (768 dim) ← финал
│   ├── training/                # Обучение моделей
│   │   ├── baseline_pipeline.py    # Baseline: LogReg на стилометрии
│   │   ├── train_models.py         # Итерация 1: A/B/C, CPU, 19 классов
│   │   ├── train_models_v2.py      # Итерация 2: GPU, 6 семейств
│   │   ├── train_models_v3.py      # Итерация 3: GPU, 19 классов, BERT-base
│   │   ├── tune_hyperparams.py     # Байесовский тюнинг (Optuna)
│   │   └── train_final.py          # Итерация 4: финальное обучение
│   ├── analysis/                # Анализ данных и интерпретация
│   │   ├── eda.py                  # EDA сырого датасета
│   │   ├── stylo_eda.py            # EDA стилометрических признаков
│   │   ├── feature_importance.py   # Важность признаков CatBoost
│   │   └── error_analysis.py       # Матрица ошибок
│   ├── utils/                   # Утилиты
│   │   ├── check_gpu.py            # Проверка видеокарты
│   │   ├── push_to_hf.py           # Выгрузка данных на Hugging Face
│   │   └── pull_from_hf.py         # Скачать данные с Hugging Face
│   └── app.py                   # Gradio-демо для инференса
├── EXPERIMENTS.md               # Журнал экспериментов и инсайты
├── PIPELINE.md                  # Этот файл
├── REPORT.md                    # Академический отчёт
└── requirements.txt
```

---

## Шаг 0. Установка окружения

```bash
# Создать и активировать виртуальное окружение
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Установить зависимости
pip install -r requirements.txt
```

**Проверить видеокарту** (рекомендуется GPU с ≥ 6 GB VRAM для шагов 5, 7, 8):

```bash
python src/utils/check_gpu.py
```

Если PyTorch не видит CUDA — переустановить:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Шаг 1. Сбор сырых данных

Датасет собирается из трёх источников: **Mendeley**, **CoAT** и **Saiga**.

### Вариант A: скачать готовые данные с Hugging Face (быстро)

Если данные уже загружены на HF-хаб:

```bash
huggingface-cli login          # Ввести токен с правом read
python src/utils/pull_from_hf.py
```

Скрипт скачает `train/val/test.parquet` сразу в `data/05_embeddings/` —  
в этом случае можно пропустить шаги 1–5 и перейти сразу к шагу 6.

### Вариант B: собрать с нуля

**1.1 Скачать CoAT и Saiga:**

```bash
python src/data/download_hf.py
```

Сохраняет `coat.parquet` и `saiga.parquet` в `data/01_raw/`.

**1.2 Скачать Mendeley (вручную):**

Датасет Mendeley распространяется отдельно и не автоматизирован.  
Разместить файлы в `data/01_raw/individual_parts/` по структуре:

```
data/01_raw/individual_parts/
├── real_texts.json              # Тексты людей
├── SberAI-small/
│   └── *.json
├── SberAI-large/
│   └── *.json
└── Facebook-xglm/
    └── *.json
```

**1.3 Собрать объединённый датасет:**

```bash
python src/data/parser.py
```

Результат: `data/02_features/unified_data.parquet` (~1.1 млн строк, ~2-4 GB).  
Скрипт использует потоковую запись (out-of-core), RAM-требование: ~4 GB.

---

## Шаг 2. EDA сырого датасета

```bash
python src/analysis/eda.py
```

Артефакты в `src/analysis/output/`:
- `01_class_balance.png` — баланс 19 классов
- `02_text_length_by_source.png` — длины текстов по источникам
- `03_text_length_kde.png` — KDE топ-7 классов
- `04_missing_values.png` — пропуски
- `05_short_text_outliers.png` — выбросы
- `eda_report.txt` — текстовый отчёт со статистикой

---

## Шаг 3. Разбиение на train / val / test

```bash
python src/data/split_data.py
```

Стратифицированное разбиение **70 / 15 / 15** по `label`, `random_state=42`.  
Результат: `data/03_splits/{train,val,test}.parquet`.

---

## Шаг 4. Извлечение стилометрических признаков

Используем **финальную версию** с морфологическим профилем (12 признаков).

```bash
python src/features/stylo_extractor_v3.py
```

> ⚠️ Требуется `pymorphy3`. Обработка ~770k строк занимает 2–4 часа на CPU  
> из-за POS-теггинга каждого слова.

Результат: `data/04_stylo_v3/{train,val,test}.parquet` — исходные колонки  
плюс 12 новых признаков:

| Признак | Описание |
|---|---|
| `avg_word_len` | Средняя длина слова |
| `avg_sentence_len` | Среднее число слов в предложении |
| `ttr` | Type-Token Ratio (лексическое разнообразие) |
| `punct_ratio` | Доля знаков препинания |
| `upper_ratio` | Доля заглавных букв |
| `digit_ratio` | Доля цифр |
| `stopword_ratio` | Доля стоп-слов |
| `word_count` | Число слов |
| `noun_ratio` | Доля существительных |
| `verb_ratio` | Доля глаголов |
| `adj_ratio` | Доля прилагательных |
| `conj_ratio` | Доля союзов и предлогов |

**EDA признаков** (опционально, но полезно перед обучением):

```bash
python src/analysis/stylo_eda.py
```

Артефакты в `src/analysis/output/stylo/`: корреляционная матрица, boxplots,  
radar chart и PCA-проекция.

---

## Шаг 5. Генерация BERT-эмбеддингов

Используем **финальную версию** — `DeepPavlov/rubert-base-cased` (768 dim).

```bash
python src/features/bert_embedder_v3.py
```

> ⚠️ Настоятельно рекомендуется GPU. На RTX 3060 (6 GB) уходит ~3-5 часов  
> на весь датасет. На CPU — несколько суток.  
> Файлы обрабатываются последовательно: `test` → `val` → `train`.  
> Если упало — скрипт пропустит уже готовые файлы при повторном запуске.

Результат: `data/05_embeddings_v3/{train,val,test}.parquet` — стилометрия  
плюс колонка `bert_embedding` (список из 768 float32).

---

## Шаг 6. Обучение финальной модели

### 6.1 Быстрый Baseline (опционально)

Логистическая регрессия только на стилометрии — для понимания нижней планки:

```bash
python src/training/baseline_pipeline.py
```

Ожидаемый результат: Accuracy ~0.55–0.65, Macro F1 ~0.15–0.25.

### 6.2 Финальное обучение — Итерация 3 (рекомендуется)

CatBoost на **19 честных классах** с полным признаковым вектором (780 dim):

```bash
python src/training/train_models_v3.py
```

Ожидаемый результат (Итерация 3):

| Метрика | Значение |
|---|---|
| Accuracy | 0.8630 |
| Macro F1 | 0.4952 |

Сохраняет `models/final_v3/hybrid_v3.pkl`.

### 6.3 Тюнинг гиперпараметров (опционально, ~2-4 часа)

Байесовская оптимизация через Optuna (20 trials на 20% сэмпла):

```bash
python src/training/tune_hyperparams.py
```

Сохраняет лучшие параметры в `models/final_v3/best_hyperparams.json`.

### 6.4 Финальное обучение с оптимальными параметрами — Итерация 4

```bash
python src/training/train_final.py
```

Ожидаемый результат (Итерация 4):

| Метрика | Значение |
|---|---|
| Accuracy | ≥ 0.87 |
| Macro F1 | ≥ 0.52 |

Сохраняет **финальную модель** `models/final_v3/Ultimate_Hybrid_V3.pkl`.

---

## Шаг 7. Анализ и интерпретация модели

```bash
# Важность признаков (какой вклад BERT vs Стилометрия)
python src/analysis/feature_importance.py

# Матрица ошибок (где путается модель)
python src/analysis/error_analysis.py
```

Артефакты в `src/analysis/output/interpretation/`:
- `top_20_features.png` — ТОП-20 признаков по важности
- `group_importance.png` — круговая диаграмма BERT vs Стилометрия
- `confusion_matrix.png` — нормализованная матрица ошибок

---

## Шаг 8. Запуск Gradio-демо

```bash
python src/app.py
```

Перейти по ссылке из консоли (обычно `http://127.0.0.1:7860`).

> ⚠️ `app.py` ищет модель по пути `models/final_v3/Ultimate_Hybrid_V3.pkl`.  
> При первом запуске скачает `DeepPavlov/rubert-base-cased` (~0.7 GB).

---

## Сводная таблица: что куда ведёт

| Скрипт | Вход | Выход |
|---|---|---|
| `src/data/download_hf.py` | HF Hub | `data/01_raw/*.parquet` |
| `src/data/parser.py` | `data/01_raw/` | `data/02_features/unified_data.parquet` |
| `src/analysis/eda.py` | `data/02_features/` | `src/analysis/output/*.png` |
| `src/data/split_data.py` | `data/02_features/` | `data/03_splits/*.parquet` |
| `src/features/stylo_extractor_v3.py` | `data/03_splits/` | `data/04_stylo_v3/*.parquet` |
| `src/analysis/stylo_eda.py` | `data/04_stylo_features/` | `src/analysis/output/stylo/*.png` |
| `src/features/bert_embedder_v3.py` | `data/04_stylo_v3/` | `data/05_embeddings_v3/*.parquet` |
| `src/training/baseline_pipeline.py` | `data/04_stylo_features/` | `models/stylo_baseline_pipeline.pkl` |
| `src/training/train_models_v3.py` | `data/05_embeddings_v3/` | `models/final_v3/hybrid_v3.pkl` |
| `src/training/tune_hyperparams.py` | `data/05_embeddings_v3/` | `models/final_v3/best_hyperparams.json` |
| `src/training/train_final.py` | `data/05_embeddings_v3/` | `models/final_v3/Ultimate_Hybrid_V3.pkl` |
| `src/analysis/feature_importance.py` | `models/final_v3/` | `src/analysis/output/interpretation/*.png` |
| `src/analysis/error_analysis.py` | `models/final_v3/` + `data/05_embeddings_v3/` | `src/analysis/output/interpretation/confusion_matrix.png` |
| `src/app.py` | `models/final_v3/` | Gradio-интерфейс |

---

## История итераций (краткая)

| Итерация | Семантика | Стилометрия | Классов | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| 1 — Baseline | rubert-tiny2 (312D) | 8 признаков | 19 | 0.7613 | 0.4021 |
| 2 — Укрупнение | rubert-tiny2 (312D) | 8 признаков | 6 | 0.9038 | 0.7552 |
| **3 — Финал** | **rubert-base (768D)** | **12 признаков** | **19** | **0.8630** | **0.4952** |
| 4 — Optuna | rubert-base (768D) | 12 признаков | 19 | ≥ 0.87 | ≥ 0.52 |

> Итерация 2 выглядит лучше, но задача была упрощена искусственно.  
> **Итерация 3/4 — честный результат** на всех 19 оригинальных классах.

---

## Частые проблемы

**`FileNotFoundError` на любом шаге**  
→ Убедитесь, что запускаете скрипты из корня `LLM_DETECTION/`, а не из папки `src/`.

**OOM (нехватка памяти GPU) при генерации эмбеддингов**  
→ Уменьшить `BATCH_SIZE` в `bert_embedder_v3.py` с 64 до 32 или 16.

**Долгая стилометрия (stylo_extractor_v3.py)**  
→ Можно запустить в несколько потоков, разбив `train.parquet` на части вручную,  
или убрать POS-теггинг (закомментировать блок `pymorphy3`) — это уберёт 4 морфологических признака.

**`ModuleNotFoundError: catboost`**  
→ `pip install catboost`

**Pickle не загружается (class Winsorizer)**  
→ Убедитесь, что в файле, где вызывается `joblib.load()`, определён класс `Winsorizer`  
(он есть в `app.py`, `error_analysis.py` и `feature_importance.py`).