# NLP HW2: генерация Python-кода

## 1. Команды для обучения итоговой модели на Kaggle

Все команды выполняются из корня проекта. Нужны локальное окружение,
авторизация Kaggle и исходный HF-кэш: подготовка описана в разделе 3.
Замените `your_kaggle_username` своим логином. Если prepared-данные уже есть,
пропустите `prepare_data.py`; если dataset с ними уже загружен, пропустите его упаковку и upload.

```bash
source .venv/bin/activate
export PATH="$(uv tool dir --bin):$PATH"
export KAGGLE_OWNER="your_kaggle_username"
export HW2_DATA_DIR="$(pwd)/../data"
export HW2_PREPARED_DIR="$HW2_DATA_DIR/prepared/code8k"

python prepare_data.py data_dir="$HW2_DATA_DIR" output_dir="$HW2_PREPARED_DIR" num_proc=4

python kaggle/package_data.py \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --output kaggle/build/repro-data --owner "$KAGGLE_OWNER" --slug nlp-hw2-data
kaggle datasets create -p kaggle/build/repro-data --keep-tabular
kaggle datasets status "$KAGGLE_OWNER/nlp-hw2-data"
```

После статуса `ready` собираем и отправляем код финальной модели:

```bash
python kaggle/prepare_pre768.py \
  --output kaggle/build/repro-train \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --owner "$KAGGLE_OWNER" \
  --code-slug nlp-hw2-code-pre768-repro \
  --inference-plan src/configs/inference_sweeps/final_beam4_128.json

kaggle datasets create -p kaggle/build/repro-train/code_dataset --keep-tabular
kaggle datasets status "$KAGGLE_OWNER/nlp-hw2-code-pre768-repro"
# После ready:
kaggle kernels push -p kaggle/build/repro-train/kernel --accelerator NvidiaTeslaT4
```

`NvidiaTeslaT4` выделяет **T4 x2**. Launcher проверяет две GPU и запускает
`torch.distributed.run --standalone --nproc_per_node=2`.
Выбран `experiments=pre768_3epochs`; обычный `train.py` без этого override
использует меньший бейзлайн. В этом варианте отправки Comet не обязателен:
`writer.mode=disabled`, метрики сохраняются в JSONL/stdout.

Сохраните фактический `owner/kernel/version` из ответа API. Например:

```bash
export TRAINING_REF="$KAGGLE_OWNER/nlp-hw2-pre768-train/1"
kaggle kernels status "$TRAINING_REF"
# После завершения — скачать outputs, включая checkpoints:
kaggle kernels output "$TRAINING_REF" -p artifacts/repro-training
```

У каждой повторной сборки должны быть новые build-каталог и code slug.
Готовые каталоги скрипты не перезаписывают. Наш лимит часов отключён;
внешний предел сессии и GPU-квота Kaggle сохраняются. Продолжение из checkpoint
описано в разделе 5.

## 2. Финальный инференс на одной T4 и замер времени

### Запуск через notebook

1. Импортируйте [inference_kaggle.ipynb](inference_kaggle.ipynb) в Kaggle.
2. Выберите **GPU T4 x2**, включите **Internet** для установки зависимостей.
3. Через **Add Input** подключите один inference-артефакт с весами и тестом:
   приложенный `final_inference_artifact.zip`, собственный экспорт либо
   доступный [dataset готовой модели](https://www.kaggle.com/datasets/andrewkhoroshilov/inference-test).
4. Запустите **Run All**. В конце выводятся PASS, число задач, параметры,
   модель GPU и фактическое время. Ответы — `/kaggle/working/predictions.jsonl`,
   отчёт — `/kaggle/working/inference_report.json`.

Ноутбук понимает ZIP, распакованную папку и структуру `code_bundle/`,
которую создаёт Kaggle. Для приватного dataset доступ предоставляется отдельно
от доступа к notebook. Для проверки приложенных финальных весов включена
сверка SHA256. При собственном переобучении поменяйте эталонные хеши или
отключите только `VERIFY_REFERENCE_ARTIFACT`.

Финальные параметры:

```text
num_beams=4  max_new_tokens=128  batch_size=32  length_penalty=0.6
amp=true  cache_type=static  sort_by_length=true  compile_step=false
```

Генерация выполняется новым процессом с `CUDA_VISIBLE_DEVICES=0`.
Проверка `require_t4=true` требует ровно одну видимую T4 и полный test из
1919 исходных вопросов. Вторая карта не участвует. Веса загружаются в GPU
внутри измеряемого процесса, test токенизируется там же.

В отчёте разделены:

| Поле | Интервал |
| --- | --- |
| `external_process_seconds` | Полный процесс: импорты, CUDA, загрузка модели, токенизация, генерация, запись JSONL и завершение |
| `preparation_seconds` | Установка пакетов, распаковка и копирование файлов до инференса |
| `notebook_elapsed_seconds` | Все ячейки вместе с подготовкой и паузами между ними |

Приведённое ниже время относится к **полному процессу инференса**.
Подготовка в него не входит. Общее время notebook также отображается,
поэтому граница измерения видна явно.

### Создать inference-артефакт после своего обучения через API

После `COMPLETE` обучения привяжите подготовленный inference job к его точной версии:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
p = Path("kaggle/build/repro-train/inference/kernel-metadata.json")
m = json.loads(p.read_text())
m["kernel_sources"] = [os.environ["TRAINING_REF"]]
p.write_text(json.dumps(m, indent=2))
PY
kaggle kernels push -p kaggle/build/repro-train/inference --accelerator NvidiaTeslaT4
```

Этот job экспортирует `best.pt`, проверяет код по manifest и выполняет два
запуска [финального плана](src/configs/inference_sweeps/final_beam4_128.json).
С фактическим ref из ответа API:

```bash
export INFERENCE_REF="$KAGGLE_OWNER/nlp-hw2-pre768-inference/1"
kaggle kernels status "$INFERENCE_REF"
# После COMPLETE:
kaggle kernels output "$INFERENCE_REF" -p artifacts/repro-inference
```

В outputs находятся `predictions-beam4-128.jsonl`, отчёты повторов,
`decoding-sweep/summary.json` и `inference_artifact/` с `model.pt`,
`tokenizer.json`, `manifest.json`, `test/`, `code_bundle.zip`.
Этот артефакт можно подключить к notebook для независимой проверки.

### Повторить benchmark отдельно

После подготовки файлов на Kaggle из каталога распакованного проекта:

```bash
python - <<'PY'
import json
from pathlib import Path
plan = {"repeats": 2, "default_tag": "beam4-128", "variants": [
    {"tag": "beam4-128", "num_beams": 4, "max_new_tokens": 128,
     "batch_size": 32, "length_penalty": 0.6}
]}
Path("/kaggle/working/final-plan.json").write_text(json.dumps(plan, indent=2))
PY
CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py \
  --checkpoint /kaggle/working/hw2_artifact/model.pt \
  --prepared-dir /kaggle/working/hw2_artifact \
  --output-dir /kaggle/working/final-benchmark \
  --plan /kaggle/working/final-plan.json
```

Каждый повтор использует новый Python-процесс. В `summary.json` должны быть
`passes_all_repeats=true` и оба `runs[*].seconds <= 180`.
В отчётах проверяются `examples=1919`, `full_test=true`,
`visible_cuda_devices=1`, `eligible_single_t4_benchmark=true`.
`generation_seconds` не заменяет полный внешний таймер.

## 3. Локальная среда и исходные данные

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv tool install --python 3.12 "kaggle==2.2.5"
export PATH="$(uv tool dir --bin):$PATH"
kaggle auth login
export HW2_DATA_DIR="$(pwd)/../data"
```

На Kaggle устанавливается только [requirements-kaggle.txt](requirements-kaggle.txt),
сохраняющий системный CUDA-PyTorch. Общий `requirements.txt` со старым PyTorch
предназначен для локального окружения.

Скачивание точной ревизии датасета:

```bash
python - <<'PY'
import os
from getpass import getpass
from pathlib import Path
from datasets import load_dataset
root = Path(os.environ["HW2_DATA_DIR"])
token = os.getenv("HF_TOKEN") or getpass("HF_TOKEN или Enter: ").strip() or None
ds = load_dataset("SlavaYus/tinysms_qa_12m",
    revision="b2724689ff079e9c22efd548fe134bfb29036dbb",
    token=token, cache_dir=str(root / "cache"))
assert len(ds["train"]) == 11838003 and len(ds["test"]) == 1919
PY
```

[prepare_data.py](prepare_data.py) работает offline с этим кэшем.
Validation выделяется по hash `question`, fraction=0.001, seed=42;
программы одного вопроса не пересекают splits. Byte-level BPE обучается
только на исходном train с исключением validation-групп, на подвыборке до
200 000 строк. Словарь 8192, min_frequency=2, полный байтовый алфавит.
Добавляется EOS; строки длиннее 256 source или target токенов отбрасываются.
Дедупликация отключена. Новые данные и таргеты не создаются.

Ожидаемые размеры: train **11 781 594**, val **11 989**, test **1919**;
отброшено по длине **44 420** строк. SHA256 `tokenizer.json`:
`27c60831ad236373af954c059f9c2f6933121efecb6e882d1909f07a8f70d152`.
Prepared содержит `train/val/test`, `tokenizer.json`, `manifest.json`.
В test сохранены вопросы и row_id, без ответов.

## 4. Модель и обучение

Все веса обучаются с нуля только на исходном датасете.
Конфиги: [code_baseline.yaml](src/configs/code_baseline.yaml),
[pre768_3epochs.yaml](src/configs/experiments/pre768_3epochs.yaml).

| Параметр | Значение |
| --- | --- |
| Модель | Encoder-decoder Transformer, **64 776 192** параметра |
| Ширина / heads / FFN | 768 / 12 / 1920 |
| Encoder / decoder | 4 / 4 независимых слоя, layer_repeats=1 |
| Нормализация / activation | Pre-Norm LayerNorm / ReLU |
| Позиции / dropout | Learned absolute, max length 256 / 0.1 |
| Embeddings | Общий входной embedding, отдельная output projection |
| Обучение | 3 эпохи, seed=42, 2 T4, DDP, FP16 autocast + GradScaler |
| Optimizer | Fused AdamW, LR=3e-4, betas=(0.9,0.98), decay=0.01 для матриц |
| Schedule / clip | Warmup 500, cosine до 10% LR / max grad norm 1.0 |
| Loss | Target-token cross-entropy, padding=-100, без label smoothing |
| Batching | До 16 384 padded source+target токенов/GPU, max 256 задач, bucket=3200/rank |
| Accumulation / workers | 2 microbatches/GPU / 2 spawn-workers на rank |
| Validation / checkpoint | Весь val каждые 500 steps / save каждые 1000, при улучшении и в конце |

Loss нормализуется по числу target-токенов на обеих GPU во всём accumulation
window. Число задач в батче переменное. План трёх эпох без overflow —
94 387 обновлений; при пропусках GradScaler успешных steps может быть меньше.
Веса и состояния optimizer хранятся в FP32.

Исходная среда Kaggle: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8.
Фактические `config.yaml`, `training_plan.json`, `environment.json`,
`metrics.jsonl` сохраняются в outputs. При другом образе и GPU побитовое
совпадение нового обучения не гарантируется.

## 5. Продолжение после остановки сессии

Kaggle ограничивает длительность сессии независимо от `max_hours=null`.
Скачайте outputs остановленной версии и создайте свой приватный Dataset
с файлами **last.pt и best.pt**. `last.pt` содержит optimizer, scheduler,
GradScaler, RNG обоих ranks и позицию в sampler; `best.pt` нужен для сохранения
лучших весов. После готовности этого dataset:

```bash
python kaggle/prepare_pre768.py \
  --output kaggle/build/repro-resume --owner "$KAGGLE_OWNER" \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --code-slug nlp-hw2-code-pre768-resume \
  --resume-dataset "$KAGGLE_OWNER/nlp-hw2-resume-checkpoint" \
  --inference-plan src/configs/inference_sweeps/final_beam4_128.json
kaggle datasets create -p kaggle/build/repro-resume/code_dataset --keep-tabular
# После ready отправляем продолжение:
kaggle kernels push -p kaggle/build/repro-resume/kernel --accelerator NvidiaTeslaT4
```

Архитектура, tokenizer/manifest, world_size=2, batching и seed должны совпадать.
Это продолжение прежних трёх эпох с восстановленным LR schedule.
Сохранение `last.pt` выполняется атомарно. Пока job работает, файлы могут
не публиковаться API; после Cancel наличие checkpoints проверяется отдельно.

## Результат итоговой конфигурации

| Показатель | Значение |
| --- | --- |
| Public pass@1, результат Telegram-бота | **73,54%** |
| Полный test | 1919 задач |
| Внешнее время двух запусков на одной T4 | 58,234157 / 57,838530 с |
| Максимум, округлённый вверх | **58,24 с < 180 с** |
| Посылка | `predictions-final-beam4-128.jsonl` |

Public score относится только к публичной части. Проверки формата и
синтаксиса не заменяют pass@1. [Отчёт 1](docs/benchmarks/final-beam4-128/repeat-0.json),
[отчёт 2](docs/benchmarks/final-beam4-128/repeat-1.json),
[эталонные хеши](docs/benchmarks/final-beam4-128/reference.json).
Каждая строка посылки содержит только объект `{"code": "..."}` в исходном порядке.
Веса, prepared, ZIP и ответы исключены из Git; для запуска готовой модели
нужен приложенный артефакт или доступ к dataset с ним.

Основа проекта: [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).
Лицензия: [MIT](LICENSE). Исследования и эксперименты сохранены в
[RESEARCH.md](RESEARCH.md) и [EXPERIMENTS.md](EXPERIMENTS.md).
