# ruRagEvaluation

Адаптация фреймворка [TRACe (RAGBench)](https://arxiv.org/abs/2407.11005) для оценки RAG-систем на русском языке.

Магисетерская диссертация, 2024–2026. Датасет: [CMCenjoyer/ragbench-ru](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru). Опубликованная модель: [CMCenjoyer/ru-trace-modernbert-2048](https://huggingface.co/CMCenjoyer/ru-trace-modernbert-2048). Текст работы: [text/thesis.pdf](text/thesis.pdf).

## О проекте

RAG (Retrieval-Augmented Generation) стал дефакто стандартом для создания QA систем с интеграцией внутренних "знаний". Однако существующие методы оценки RAG (BLEU, ROUGE, BERTScore) непрозрачны и не позволяют понять, где именно система ошибается — в поиске контекста или в генерации ответа. Фреймворк TRACe решает эту проблему через токен-уровневую классификацию, но работает только для английского языка.

В данном проекте:

1. **Переведён RAGBench** (7 доменных датасетов, ~30k примеров) на русский язык и опубликован на [HuggingFace](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru).
2. **Обучены token-level оценщики** для трёх таргетов: relevance, utilization, adherence.
3. **Использовались backbone-модели**: DeBERTa-v3, BGE-M3, RuModernBERT, RoPEBert.
4. **Реализованы две архитектуры классификационных голов**:
   - **simple** — 3 линейные головы поверх backbone (paper-style, дефолт);
   - **complex** — shared transform + residual + LayerNorm + MLP-головы.
5. **Эксперименты с весами лосса** в двух режимах: 1:1:1 (равные) и 3:3:1 (акцент на relevance/utilization).
6. **Воспроизведена LLM-based оценка** (Qwen2.5-72B) по промпту из оригинальной статьи; есть сопоставление с обученными классификаторами.
7. **OOD(Out-of-Distribution)-валидация** обученных оценщиков на `bearberry/sberquadqa` — см. [external_eval/sberquadqa/](external_eval/sberquadqa/).
8. **Streamlit-сервис** для интерактивной оценки RAG-систем — [new_streamlit.py](new_streamlit.py).
9. **LaTeX-текст работы** — в [text/](text/), собранный PDF — [text/thesis.pdf](text/thesis.pdf).

## Структура репозитория

```
ruRagEvaluation/
├── model_utils/                      # Пайплайн обучения и оценки
│   ├── config.py                     # DataConfig / ModelConfig / TrainingConfig / ProjectConfig
│   ├── data.py                       # Загрузка RAGBench-RU, токенизация, маски
│   ├── models.py                     # DebertaTraceSimple / DebertaTraceComplex (backbone + 3 головы)
│   ├── metrics.py                    # Token-level P/R/F1, подбор порогов на val
│   ├── pipeline.py                   # Полный пайплайн: данные -> обучение -> eval -> result.json
│   ├── train_loop.py                 # Цикл обучения (train/val эпохи)
│   ├── runner.py                     # Grid sweep по моделям и длинам контекста
│   ├── run_experiments.py            # Точка входа (CLI и run_single)
│   ├── results.py                    # Сериализация результатов
│   ├── inference.py                  # Инференс на сохранённых чекпоинтах
│   └── sweeps3/                      # Локальная директория с чекпоинтами; в репозиторий не коммитится
│
├── prompt_test/                      # LLM-based оценка
│   ├── data.py                       # Загрузка RAGBench, обёртка для LLM
│   ├── scoring.py                    # Async-запуск Qwen2.5-72B
│   ├── metrics.py                    # Overlap-метрики (sentence-level)
│   └── utils.py                      # Async-утилиты для API-запросов
│
├── external_eval/sberquadqa/         # OOD-валидация обученных оценщиков
│   ├── _common.py                    # MODEL_REGISTRY, загрузка моделей, агрегация скоров
│   ├── prepare_dataset.py            # Подготовка SberQuadQA в chunk-level формате
│   ├── score_dataset.py              # Инференс выбранной моделью
│   ├── analyze_results.py            # Метрики и визуализации одного прогона
│   ├── run_all.py                    # Прогон по всем доступным чекпоинтам + leaderboard
│   ├── summarize_outputs.py          # Сводная агрегация результатов
│   └── outputs/                      # Generated artifacts; крупные .parquet могут отсутствовать в git
│
├── analysis_output/                  # Сводный анализ для текста диплома
│   ├── tables/                       # CSV/TeX (full_grid, leaderboard_top10, ru_vs_en_*, ...)
│   └── plots/                        # PNG (best_models_bar_*, f1_vs_length_*, ablation_*, ...)
│
├── text/                             # LaTeX-диссертация
│   ├── thesis.tex                    # Корневой файл
│   ├── chapters/                     # 01_intro ... 08_conclusion + appendix
│   ├── references.bib
│   ├── Makefile                      # Сборка через latexmk
│   └── thesis.pdf                    # Собранный текст (перегенерируется make)
│
├── new_streamlit.py                  # Актуальный Streamlit-сервис
├── old_streamlit.py                  # Deprecated, оставлен для истории
├── analysis.ipynb                    # Сводный анализ всех экспериментов
├── tranclate.ipynb                   # Перевод RAGBench на русский
├── torch_model.ipynb                 # Воспроизведение пайплайна на EN
├── test_llm_razm.ipynb / llm_judje.ipynb  # Эксперименты с LLM-оценкой
├── ... (и др. вспомогательные ноутбуки)
│
├── dissertation_plan.md              # Постановка задачи и статус
└── CLAUDE.md                         # Конвенции и типичные команды
```

## Модели

| Модель | Тип | Макс. длина контекста | Языки |
|--------|-----|----------------------|-------|
| [MoritzLaurer/DeBERTa-v3-large](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | DeBERTa | 128, 256, 512 | Multi |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | BGE | 512 -- 8192 | Multi |
| [deepvk/RuModernBERT-base](https://huggingface.co/deepvk/RuModernBERT-base) | ModernBERT | 512 -- 8192 | RU |
| [Tochka-AI/ruRoPEBert-e5-base-2k](https://huggingface.co/Tochka-AI/ruRoPEBert-e5-base-2k) | RoPE-BERT | 256 -- 2048 | RU |

**Архитектуры моделей**: backbone (трансформер) + либо 3 линейные головы (**simple**, paper-style, дефолт), либо shared transform + residual + LayerNorm + MLP-головы (**complex**). Переключение — флагом `use_complex_model` в `ModelConfig` (Python-конфиг). CLI-флага для этого режима нет: при программном запуске нужно собрать `ProjectConfig` и вызвать `pipeline.train(...)` напрямую.

**Веса лосса**: задаются CLI-флагом `--loss-weights REL UTIL ADH` в [model_utils/run_experiments.py](model_utils/run_experiments.py), например `--loss-weights 3 3 1`. Веса нормируются по сумме. Эксперименты проведены для двух режимов: 1:1:1 и 3:3:1.

**Чекпоинты**: артефакты обучения сохраняются в `model_utils/sweeps3/<model>/len_<L>/` (`model.pt` + `result.json` с подобранными на val порогами). В git не коммитятся.

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

Для каждого таргета вычисляются Precision, Recall и F1. Оптимальные пороги классификации подбираются на validation set (сетка 0.1 -- 0.9).

## Результаты

Лучшая RU-конфигурация на test (из [analysis_output/tables/leaderboard_top10.csv](analysis_output/tables/leaderboard_top10.csv)) — **ModernBERT, max_length=2048, веса 3:3:1, simple-head**: mean F1 ≈ 0.858 (relevance ≈ 0.798, utilization ≈ 0.841, adherence ≈ 0.935). В топ-3 — также BGE-M3 на длине 2048 (1:1:1 и 3:3:1) с близким качеством. Длина 2048 устойчиво лидирует у ModernBERT и BGE-M3; DeBERTa архитектурно ограничена ~512 токенами и проигрывает по utilization, но конкурентна по relevance на коротких контекстах.

Сравнение с **Qwen2.5-72B**: LLM измерялась через sentence-level overlap, а классификаторы — через token-level F1, поэтому прямое сопоставление чисел некорректно. Детали — в [analysis_output/tables/classifier_vs_llm.csv](analysis_output/tables/classifier_vs_llm.csv) и в тексте диплома.

Ключевые артефакты:
- [analysis_output/tables/leaderboard_top10.csv](analysis_output/tables/leaderboard_top10.csv) — топ-10 конфигураций по mean F1.
- [analysis_output/tables/best_config_per_model.csv](analysis_output/tables/best_config_per_model.csv) — лучшая длина для каждой модели.
- [analysis_output/tables/ru_vs_en_331.csv](analysis_output/tables/ru_vs_en_331.csv) — сравнение RU vs EN (3:3:1).
- [analysis_output/tables/classifier_vs_llm.csv](analysis_output/tables/classifier_vs_llm.csv) — классификаторы vs Qwen2.5-72B.
- [analysis_output/plots/](analysis_output/plots/) — графики (`best_models_bar_*`, `f1_vs_length_*`, `ablation_weights_*`, `ru_vs_en_*`, `simple_vs_complex_ru_331*` и др.).

Воспроизведение всех таблиц и графиков — [analysis.ipynb](analysis.ipynb).

## OOD-валидация на SberQuadQA

Внешний out-of-distribution датасет [bearberry/sberquadqa](https://huggingface.co/datasets/bearberry/sberquadqa) с чанк-уровневой разметкой `is_relevant`. Принципы валидации:

- модели применяются **без дообучения** и **без подбора порогов** на SberQuadQA;
- token-level предикты агрегируется до уровня чанков;
- пороги берутся из `result.json` рядом с чекпоинтом (если файла нет — fallback 0.5);
- Примеры выходящие за допустимую длину контекста по умолчанию пропускаются (флаг `--truncate` опционален).

Метрики: F1, ROC-AUC, PR-AUC, top1_acc, top-k recall.

Кратко по [external_eval/sberquadqa/outputs/leaderboard.csv](external_eval/sberquadqa/outputs/leaderboard.csv): лучшая на OOD — **ModernBERT (len=2048)**: F1 ≈ 0.757, ROC-AUC ≈ 0.927, PR-AUC ≈ 0.813, top1_acc ≈ 0.897. BGE-M3 (len=1024) и DeBERTa-v3-large (len=512) — заметно ниже по F1 при сопоставимом ROC-AUC.

Артефакты (все *generated*, могут перегенерироваться):
- [external_eval/sberquadqa/outputs/leaderboard.csv](external_eval/sberquadqa/outputs/leaderboard.csv)
- [external_eval/sberquadqa/outputs/combined_bar_metrics.png](external_eval/sberquadqa/outputs/combined_bar_metrics.png)
- [external_eval/sberquadqa/outputs/combined_pr_curve.png](external_eval/sberquadqa/outputs/combined_pr_curve.png)
- [external_eval/sberquadqa/outputs/combined_roc_curve.png](external_eval/sberquadqa/outputs/combined_roc_curve.png)
- per-model папки с `chunks_scored.parquet`, `metrics_summary.json`, `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `failure_cases.json`, `run_meta.json`.

## Быстрый старт

### Установка

- Python 3.9+ (рекомендуется 3.10+).
- Минимальные зависимости:

```bash
pip install torch transformers datasets pandas numpy scikit-learn streamlit tqdm matplotlib pyarrow
```

- Для LLM-based оценки нужен openai-совместимый SDK (`pip install openai`): [prompt_test/utils.py](prompt_test/utils.py) использует `client.chat.completions.create` (можно подключаться к локальному серверу через `base_url`).
- При загрузке RuModernBERT и RoPEBert требуется `trust_remote_code=True`.

### Чекпоинты

Готовые чекпоинты экспериментов лежат по пути `model_utils/sweeps3/<model>/len_<L>/model.pt` и в репозиторий не коммитятся (хранятся локально / на удалённом сервере). Без них [new_streamlit.py](new_streamlit.py) и [external_eval/sberquadqa/score_dataset.py](external_eval/sberquadqa/score_dataset.py) инференс выполнить не смогут. Варианты: (1) обучить модели заново через [model_utils/run_experiments.py](model_utils/run_experiments.py); (2) запросить артефакты у автора.

### Обучение моделей

```bash
# Полный sweep по всем моделям и длинам контекста
python model_utils/run_experiments.py

# Один эксперимент
python model_utils/run_experiments.py --models BAAI/bge-m3 --lengths 1024 --epochs 3

# Эксперимент с акцентом на relevance/utilization (веса 3:3:1)
python model_utils/run_experiments.py --models BAAI/bge-m3 --lengths 2048 --loss-weights 3 3 1

# Smoke test (одна модель × одна длина)
python model_utils/run_experiments.py --single
```

Запуск возможен и из корня репозитория, и из `model_utils/` — внутри `run_experiments.py` подмешиваются нужные `sys.path`. Переключение на complex-head делается только программно: `ModelConfig(use_complex_model=True)` + вызов `pipeline.train(cfg)` (отдельного CLI-флага нет).

### Streamlit-сервис

```bash
streamlit run new_streamlit.py
```

Требует наличия чекпоинтов (см. выше).

### OOD-эксперимент на SberQuadQA

```bash
# 1. Подготовка датасета (chunk-level)
python -m external_eval.sberquadqa.prepare_dataset --n_examples 5000

# 2. Скоринг конкретной моделью (ключ из MODEL_REGISTRY в _common.py)
python -m external_eval.sberquadqa.score_dataset --model_key "BAAI-bge-m3 (len=1024)"

# 3. Анализ метрик одного прогона
python -m external_eval.sberquadqa.analyze_results --scored_dir external_eval/sberquadqa/outputs/<tag>

# 4. Прогон по всем доступным чекпоинтам и сборка leaderboard.csv
python -m external_eval.sberquadqa.run_all

# 5. Сводная агрегация результатов
python -m external_eval.sberquadqa.summarize_outputs
```

### LaTeX-диссертация

```bash
cd text && make    # latexmk-сборка -> text/thesis.pdf
```

### Анализ результатов

Откройте [analysis.ipynb](analysis.ipynb), укажите пути к папкам с результатами в ячейке конфигурации и запустите все ячейки.

## Документация

- [dissertation_plan.md](dissertation_plan.md) — постановка задачи и статус.
- [CLAUDE.md](CLAUDE.md) — конвенции, типичные команды, особенности кода.
- [text/thesis.pdf](text/thesis.pdf) — собранный текст работы.
- [analysis.ipynb](analysis.ipynb) — воспроизведение всех таблиц и графиков.

## Ссылки

- **Оригинальная статья**: [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005)
- **Датасет (RU)**: [CMCenjoyer/ragbench-ru](https://huggingface.co/datasets/CMCenjoyer/ragbench-ru)
- **Оригинальный RAGBench (EN)**: [rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench)
- **OOD-датасет**: [bearberry/sberquadqa](https://huggingface.co/datasets/bearberry/sberquadqa)
