# ruRagEvaluation

Адаптация фреймворка [TRACe (RAGBench)](https://arxiv.org/abs/2407.11005) для оценки RAG-систем на русском языке.

## О проекте

RAG (Retrieval-Augmented Generation) стал дефакто стандартом для создания QA систем с интеграцией внутренних "знаний". Однако существующие методы оценки RAG (BLEU, ROUGE, BERTScore) непрозрачны и не позволяют понять, где именно система ошибается — в поиске контекста или в генерации ответа. Фреймворк TRACe решает эту проблему через токен-уровневую классификацию, но работает только для английского языка.

В данном проекте:

1. **Переведён RAGBench** (7 доменных датасетов, ~30k примеров) на русский язык и опубликован на [HuggingFace](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru)
2. **Обучены 4 модели-классификатора** (DeBERTa-v3, BGE-M3, RuModernBERT, RoPEBert) для токен-уровневой оценки relevance, utilization и adherence
3. **Воспроизведена LLM-based оценка** (Qwen2.5-72B) по промпту из оригинальной статьи
4. **Проведено сравнение** влияние длины контекста, весов лосса, RU vs EN
5. **Создан Streamlit-сервис** для интерактивной оценки RAG-систем

## Структура репозитория

```
ruRagEvaluation/
├── model_utils/                # Пайплайн обучения и оценки
│   ├── config.py               # Конфигурация (DataConfig, ModelConfig, TrainingConfig)
│   ├── data.py                 # Загрузка RAGBench-RU, токенизация, построение примеров
│   ├── models.py               # DebertaTraceSimple / DebertaTraceComplex (backbone + 3 головы)
│   ├── metrics.py              # Token-level P/R/F1, подбор порогов
│   ├── pipeline.py             # Полный пайплайн: данные -> обучение -> eval -> сохранение
│   ├── train_loop.py           # Цикл обучения (train/val эпохи)
│   ├── runner.py               # Grid sweep по моделям и длинам контекста
│   ├── run_experiments.py      # Точка входа для запуска экспериментов
│   ├── results.py              # Сериализация результатов (result.json)
│   └── inference.py            # Инференс на сохранённых чекпоинтах
│
├── prompt_test/                # LLM-based оценка
│   ├── data.py                 # Загрузка RAGBench, обёртка для LLM
│   ├── scoring.py              # Асинхронный запуск оценки через Qwen2.5-72B
│   ├── metrics.py              # Overlap-метрики (sentence-level)
│   └── utils.py                # Async-утилиты для API-запросов
│
├── new_streamlit.py            # Streamlit-сервис (инференс, визуализация, batch-оценка)
├── analysis.ipynb              # Сводный анализ результатов всех экспериментов
├── torch_model.ipynb           # Начальные эксперименты на английском датасете
├── tranclate.ipynb             # Перевод RAGBench на русский
├── test_llm_razm.ipynb         # Эксперименты с LLM-based оценкой
└── dissertation_plan.md        # Структура диссертации
```

## Модели

| Модель | Тип | Макс. длина контекста | Языки |
|--------|-----|----------------------|-------|
| [MoritzLaurer/DeBERTa-v3-large](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | DeBERTa | 128, 256, 512 | Multi |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | BGE | 512 -- 8192 | Multi |
| [deepvk/RuModernBERT-base](https://huggingface.co/deepvk/RuModernBERT-base) | ModernBERT | 512 -- 8192 | RU |
| [Tochka-AI/ruRoPEBert-e5-base-2k](https://huggingface.co/Tochka-AI/ruRoPEBert-e5-base-2k) | RoPE-BERT | 256 -- 2048 | RU |

Архитектура: backbone (трансформер) + 3 линейных головы для token-level классификации.

## Датасет

**RAGBench-RU**: [CMCenjoyer/ragbench-ru](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru)

Перевод выполнен с помощью Qwen2.5-72B-Instruct. Содержит 7 доменных поддатасетов:

| Датасет | Домен |
|---------|-------|
| CovidQA | Медицина (COVID-19) |
| CUAD | Юриспруденция (контракты) |
| DelucionQA | Детекция галлюцинаций |
| ExpertQA | Экспертные вопросы |
| FinQA | Финансы |
| HagRID | Общие знания |
| HotpotQA | Многошаговые рассуждения |

Каждый пример содержит: вопрос, документы (с разбивкой на предложения), ответ, разметку relevance/utilization/adherence.

## Метрики TRACe

Три таргета оцениваются на уровне токенов:

- **Relevance** — какие предложения документов релевантны вопросу
- **Utilization** — какие из релевантных предложений фактически использованы в ответе
- **Adherence** — насколько токены ответа подкреплены документами

Для каждого таргета вычисляются Precision, Recall и F1. Оптимальные пороги классификации подбираются на validation set (сетка 0.1 -- 0.7).

## Быстрый старт

### Установка

```bash
pip install torch transformers datasets pandas streamlit tqdm
```

### Обучение моделей

```bash
cd model_utils

# Полный sweep по всем моделям и длинам контекста
python run_experiments.py

# Один эксперимент
python run_experiments.py --models "BAAI/bge-m3" --lengths 1024 --epochs 3

# Smoke test
python run_experiments.py --single
```

### Streamlit-сервис

```bash
streamlit run new_streamlit.py
```

### Анализ результатов

Откройте `analysis.ipynb`, укажите пути к папкам с результатами в ячейке конфигурации и запустите все ячейки.

## Ссылки

- **Оригинальная статья**: [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005)
- **Датасет (RU)**: [CMCenjoyer/ragbench-ru](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru)
- **Оригинальный RAGBench (EN)**: [rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench)
