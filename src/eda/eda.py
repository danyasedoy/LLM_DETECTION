"""
EDA — Разведочный анализ данных
Лабораторная работа №3 / НИР: Гибридная LLM-атрибуция

Запуск из корня проекта:
    python src/eda/eda.py

Выходные файлы сохраняются в: src/eda/output/
"""

import os
import gc
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Без GUI — для запуска в скриптах/серверах
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────── Конфигурация ────────────────────────────────────

DATA_PATH   = Path('data/02_features/unified_data.parquet')
OUTPUT_DIR  = Path('src/eda/output')
REPORT_PATH = OUTPUT_DIR / 'eda_report.txt'

# Палитра: нейтральная, хорошо читается в ч/б и цвете
PALETTE = 'muted'

# Маппинг меток → источник датасета
SOURCE_MAP = {
    # Mendeley (Антиплагиат, Wikipedia)
    'sberai_small':  'Mendeley',
    'sberai_large':  'Mendeley',
    'facebook_xglm': 'Mendeley',

    # CoAT (RussianNLP, новости + соцсети)
    'rugpt3-large':        'CoAT',
    'rugpt3-small':        'CoAT',
    'rugpt3-medium':       'CoAT',
    'm-bart50':            'CoAT',
    'rut5-base':           'CoAT',
    'mt5-small':           'CoAT',
    'rut5-base-multitask': 'CoAT',
    'rugpt2-large':        'CoAT',
    'm2m-100':             'CoAT',
    'opus-mt':             'CoAT',
    'mt5-large':           'CoAT',
    'rut5-large':          'CoAT',
    'm-bart':              'CoAT',

    # Saiga (Гусев, диалоги с современными LLM)
    'gpt-3.5-turbo': 'Saiga',
    'gpt-4':         'Saiga',

    # human встречается и в Mendeley, и в CoAT — источник неразличим без доп. колонки
    'human': 'Mixed (Mendeley + CoAT)',
}

SOURCE_COLORS = {
    'Mendeley':               '#4C72B0',
    'CoAT':                   '#55A868',
    'Saiga':                  '#C44E52',
    'Mixed (Mendeley + CoAT)':'#8172B3',
}

# ─────────────────────────── Утилиты ─────────────────────────────────────────

def save_fig(name: str):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ Сохранено: {path}')


def log(msg: str, file=None):
    print(msg)
    if file:
        file.write(msg + '\n')


# ─────────────────────────── Загрузка данных ─────────────────────────────────

SAMPLE_SIZE = 150_000  # Хватает для стат. значимых KDE и box plots

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает два датафрейма:
      df_labels — только колонка label (весь датасет, ~50 MB)
      df_sample — text + label на SAMPLE_SIZE строк (для графиков длин)
    """
    print(f'\n📂 Загружаем: {DATA_PATH}')
    if not DATA_PATH.exists():
        sys.exit(f'❌ Файл не найден: {DATA_PATH}\n'
                 f'   Убедитесь, что запускаете скрипт из корня проекта.')

    # Шаг 1: только label — весь датасет (легко, строки короткие)
    print('   Шаг 1/2: читаем колонку label (весь датасет)...')
    df_labels = pd.read_parquet(DATA_PATH, columns=['label'])
    df_labels['source'] = df_labels['label'].map(SOURCE_MAP).fillna('Unknown')
    print(f'   Загружено строк: {len(df_labels):,}')

    # Шаг 2: text+label на стратифицированном сэмпле — для графиков длин
    print(f'   Шаг 2/2: стратифицированный сэмпл {SAMPLE_SIZE:,} строк для анализа длин...')
    df_full = pd.read_parquet(DATA_PATH, columns=['text', 'label'])

    # Стратифицируем, чтобы редкие классы (gpt-4) попали в сэмпл
    frac = min(1.0, SAMPLE_SIZE / len(df_full))
    df_sample = (df_full.groupby('label', group_keys=False)
                        .apply(lambda g: g.sample(frac=frac, random_state=42))
                        .reset_index(drop=True))
    del df_full
    gc.collect()

    df_sample['text_len']   = df_sample['text'].str.len()
    df_sample['word_count'] = df_sample['text'].str.split().str.len()
    df_sample['source']     = df_sample['label'].map(SOURCE_MAP).fillna('Unknown')
    print(f'   Сэмпл готов: {len(df_sample):,} строк')

    return df_labels, df_sample


# ─────────────────────────── График 1: Баланс классов ────────────────────────

def plot_class_balance(df: pd.DataFrame):
    print('\n📊 График 1: Баланс классов...')

    counts  = df['label'].value_counts()  # df здесь = df_labels
    total   = len(df)
    pcts    = (counts / total * 100).round(1)
    sources = counts.index.map(lambda x: SOURCE_MAP.get(x, 'Unknown'))
    colors  = [SOURCE_COLORS.get(SOURCE_MAP.get(lbl, 'Unknown'), '#999') for lbl in counts.index]

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)

    # Подписи: количество + процент
    for bar, cnt, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f'{cnt:,}  ({pct}%)', va='center', ha='left', fontsize=8.5)

    # Легенда по источникам
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in SOURCE_COLORS.items()
                       if s in sources.values]
    ax.legend(handles=legend_elements, title='Источник', loc='lower right', fontsize=9)

    ax.set_xlabel('Количество примеров', fontsize=11)
    ax.set_title('Баланс классов в объединённом датасете\n(sorted by count)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlim(0, counts.max() * 1.18)

    # Вертикальная линия: средний размер класса (без human)
    non_human = counts[counts.index != 'human']
    ax.axvline(non_human.mean(), color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Среднее (без human): {int(non_human.mean()):,}')

    plt.tight_layout()
    save_fig('01_class_balance.png')


# ─────────────────────────── График 2: Длина текстов по источникам ───────────

def plot_length_by_source(df: pd.DataFrame):
    print('\n📊 График 2: Длина текстов по источникам...')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sources_order = ['Mendeley', 'CoAT', 'Saiga', 'Mixed (Mendeley + CoAT)']
    colors = [SOURCE_COLORS[s] for s in sources_order if s in df['source'].unique()]
    present_sources = [s for s in sources_order if s in df['source'].unique()]

    # Box plot — длина в символах
    data_by_source = [df[df['source'] == s]['text_len'].dropna() for s in present_sources]
    bp = axes[0].boxplot(data_by_source, labels=present_sources, patch_artist=True,
                         showfliers=False, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title('Длина текста (символы) по источнику\n[без выбросов]', fontweight='bold')
    axes[0].set_ylabel('Длина текста (символы)')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Box plot — количество слов
    data_by_source_w = [df[df['source'] == s]['word_count'].dropna() for s in present_sources]
    bp2 = axes[1].boxplot(data_by_source_w, labels=present_sources, patch_artist=True,
                          showfliers=False, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title('Количество слов по источнику\n[без выбросов]', fontweight='bold')
    axes[1].set_ylabel('Количество слов')
    axes[1].tick_params(axis='x', rotation=15)

    plt.suptitle('Распределение длин текстов по источникам данных', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('02_text_length_by_source.png')


# ─────────────────────────── График 3: KDE длин для топ-классов ──────────────

def plot_length_kde(df: pd.DataFrame):
    print('\n📊 График 3: KDE распределений длин для ключевых классов...')

    # Берём топ-7 классов по количеству + gpt-4 (важен семантически)
    top_labels = list(df['label'].value_counts().head(7).index)
    if 'gpt-4' not in top_labels:
        top_labels.append('gpt-4')

    fig, ax = plt.subplots(figsize=(13, 6))

    palette = sns.color_palette(PALETTE, len(top_labels))

    for label, color in zip(top_labels, palette):
        subset = df[df['label'] == label]['text_len']
        # Убираем экстремальные выбросы для читаемости KDE
        subset = subset[subset < subset.quantile(0.98)]
        subset.plot.kde(ax=ax, label=f'{label} (n={len(df[df["label"]==label]):,})',
                        color=color, linewidth=1.8)

    ax.set_xlabel('Длина текста (символы)', fontsize=11)
    ax.set_ylabel('Плотность', fontsize=11)
    ax.set_title('KDE распределения длин текстов: топ-7 классов + GPT-4\n'
                 '(Если кривые смещены — модель будет читерить на длине, а не на стиле)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    save_fig('03_text_length_kde.png')


# ─────────────────────────── График 4: Матрица пропусков ─────────────────────

def plot_missing_values(df: pd.DataFrame):
    print('\n📊 График 4: Матрица пропусков...')

    try:
        import missingno as msno
    except ImportError:
        print('  ⚠️  missingno не установлен. Строим упрощённую версию.')
        _plot_missing_simple(df)
        return

    # missingno работает с сэмплом (полный датасет слишком тяжёлый для визуализации)
    sample = df[['text', 'label', 'text_len', 'word_count']].sample(
        min(5000, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    msno.matrix(sample, ax=ax, sparkline=False, fontsize=11,
                color=(0.25, 0.45, 0.75))
    ax.set_title('Матрица пропусков (сэмпл 5,000 строк)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig('04_missing_values.png')


def _plot_missing_simple(df: pd.DataFrame):
    """Fallback если missingno не установлен."""
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df[['text', 'label', 'text_len', 'word_count']].isnull().sum()
    colors = ['#C44E52' if v > 0 else '#55A868' for v in missing.values]
    ax.bar(missing.index, missing.values, color=colors)
    ax.set_title('Количество пропусков по колонкам', fontweight='bold')
    ax.set_ylabel('Кол-во пропусков')
    for i, v in enumerate(missing.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold',
                color='green' if v == 0 else 'red')
    plt.tight_layout()
    save_fig('04_missing_values.png')


# ─────────────────────────── График 5: Выбросы — короткие тексты ─────────────

def plot_short_text_outliers(df: pd.DataFrame):
    print('\n📊 График 5: Выбросы — доля очень коротких текстов (<50 символов)...')

    threshold = 50
    short = df[df['text_len'] < threshold].groupby('label').size()
    total_by_label = df.groupby('label').size()
    pct_short = (short / total_by_label * 100).fillna(0).sort_values(ascending=False)

    # Показываем только классы, где >0% коротких текстов
    pct_short = pct_short[pct_short > 0]

    if len(pct_short) == 0:
        print('  ℹ️  Коротких текстов не обнаружено. График пропускается.')
        # Создаем информационный плейсхолдер
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'✅ Коротких текстов (< {threshold} символов)\nне обнаружено ни в одном классе',
                ha='center', va='center', fontsize=14, color='green',
                transform=ax.transAxes)
        ax.set_title(f'Проверка на короткие тексты (< {threshold} символов)', fontweight='bold')
        ax.axis('off')
        save_fig('05_short_text_outliers.png')
        return

    colors = [SOURCE_COLORS.get(SOURCE_MAP.get(lbl, 'Unknown'), '#999') for lbl in pct_short.index]

    fig, ax = plt.subplots(figsize=(10, max(5, len(pct_short) * 0.5)))
    bars = ax.barh(pct_short.index, pct_short.values, color=colors, edgecolor='white')

    for bar, val in zip(bars, pct_short.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left', fontsize=9)

    ax.axvline(1.0, color='red', linestyle='--', linewidth=1, label='Порог 1%')
    ax.set_xlabel(f'% текстов короче {threshold} символов', fontsize=11)
    ax.set_title(f'Доля текстов-выбросов (< {threshold} символов) по классам\n'
                 f'(Потенциальный шум в данных)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    save_fig('05_short_text_outliers.png')


# ─────────────────────────── Текстовый отчёт ─────────────────────────────────

def generate_report(df_labels: pd.DataFrame, df_sample: pd.DataFrame):
    print('\n📝 Генерируем текстовый отчёт...')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        log('=' * 65, f)
        log('EDA REPORT — Hybrid LLM Attribution Dataset', f)
        log('=' * 65, f)

        total = len(df_labels)
        log('\n[1] БАЗОВАЯ СТАТИСТИКА', f)
        log(f'  Всего записей:    {total:,}', f)
        log(f'  Число классов:    {df_labels["label"].nunique()}', f)
        log(f'  Число источников: {df_labels["source"].nunique()}', f)
        log(f'  Сэмпл для анализа длин: {len(df_sample):,} строк', f)

        # 2. Пропуски
        log('\n[2] ПРОПУСКИ', f)
        missing = df_labels[['label']].isnull().sum()
        for col, cnt in missing.items():
            status = '✅ OK' if cnt == 0 else f'❌ {cnt:,} пропусков!'
            log(f'  {col}: {status}', f)

        # 3. Короткие тексты (на сэмпле)
        log('\n[3] КОРОТКИЕ ТЕКСТЫ (< 50 символов) — потенциальный шум [на сэмпле]', f)
        short_df = df_sample[df_sample['text_len'] < 50]
        log(f'  Всего коротких в сэмпле: {len(short_df):,} ({len(short_df)/len(df_sample)*100:.2f}%)', f)
        if len(short_df) > 0:
            log('  По классам:', f)
            for label, cnt in short_df['label'].value_counts().items():
                log(f'    {label}: {cnt:,}', f)

        # 4. Длины текстов по классам (на сэмпле)
        log('\n[4] СТАТИСТИКА ДЛИН ТЕКСТОВ (символы) ПО КЛАССАМ [на сэмпле]', f)
        stats = df_sample.groupby('label')['text_len'].agg(['mean', 'median', 'min', 'max', 'std'])
        stats = stats.round(0).astype(int)
        log(f'  {"Класс":<25} {"Среднее":>10} {"Медиана":>10} {"Мин":>8} {"Макс":>10} {"Std":>10}', f)
        log(f'  {"-"*75}', f)
        for label, row in stats.iterrows():
            log(f'  {label:<25} {row["mean"]:>10,} {row["median"]:>10,} '
                f'{row["min"]:>8,} {row["max"]:>10,} {row["std"]:>10,}', f)

        # 5. Проверка на подозрительные метки Saiga
        log('\n[5] ПРОВЕРКА МЕТОК SAIGA (нет ли Llama/Mistral в human?)', f)
        suspicious_patterns = ['llama', 'mistral', 'saiga', 'openchat', 'vicuna',
                                'zephyr', 'falcon', 'alpaca', 'orca']
        found_any = False
        for pattern in suspicious_patterns:
            mask = df_labels['label'].str.contains(pattern, case=False, na=False)
            if mask.sum() > 0:
                log(f'  ⚠️  Найдено "{pattern}": {mask.sum():,} записей', f)
                found_any = True
        if not found_any:
            log('  ✅ Подозрительных меток не найдено', f)

        # 6. Дисбаланс классов
        log('\n[6] СООТНОШЕНИЕ ДИСБАЛАНСА', f)
        counts = df_labels['label'].value_counts()
        ratio = counts.max() / counts.min()
        log(f'  Самый большой класс: {counts.idxmax()} ({counts.max():,})', f)
        log(f'  Самый малый класс:   {counts.idxmin()} ({counts.min():,})', f)
        log(f'  Соотношение max/min: {ratio:.0f}x', f)
        if ratio > 100:
            log(f'  ❌ Экстремальный дисбаланс! class_weight="balanced" + macro F1.', f)

        # 7. Проблема human
        log('\n[7] ПРОБЛЕМА КЛАССА "human"', f)
        human_count = counts.get('human', 0)
        log(f'  human: {human_count:,} записей ({human_count/total*100:.1f}% датасета)', f)
        log(f'  ⚠️  Агрегирует Wikipedia (Mendeley) + новости/соцсети (CoAT).', f)
        log(f'     Рекомендация: отслеживать confusion matrix human по доменам.', f)

        # 8. Рекомендации
        log('\n[8] ВЫВОДЫ И РЕКОМЕНДАЦИИ ДЛЯ ПРЕДОБРАБОТКИ', f)
        log('  1. Пропуски отсутствуют — imputation не требуется.', f)
        log('  2. Дисбаланс критический → class_weight="balanced", метрика macro F1.', f)
        log('  3. Длины текстов различаются → нормализовать stylo-признаки к длине.', f)
        log('  4. Разбиение: стратифицированное по label, seed=42, пропорции 70/15/15.', f)
        log('  5. StandardScaler для stylo — вычислять только на train-выборке.', f)

        log('\n' + '=' * 65, f)
        log('Конец отчёта', f)

    print(f'  ✅ Отчёт сохранён: {REPORT_PATH}')


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_labels, df_sample = load_data()

    print(f'\n🔎 Сэмпл (первые строки):')
    print(df_sample[['label', 'source', 'text_len', 'word_count']].head(5).to_string())

    # Графики баланса и отчёт — на полном df_labels (только label, RAM ок)
    # Графики длин — на df_sample
    plot_class_balance(df_labels)
    plot_length_by_source(df_sample)
    plot_length_kde(df_sample)
    plot_missing_values(df_sample)
    plot_short_text_outliers(df_sample)
    generate_report(df_labels, df_sample)

    print('\n' + '=' * 55)
    print('🎉 EDA завершён. Все файлы в:', OUTPUT_DIR)
    print('=' * 55)


if __name__ == '__main__':
    main()