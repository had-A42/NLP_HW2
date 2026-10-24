# NLP HW2: генерация Python-кода по условию задачи

Проект обучает encoder-decoder Transformer с нуля на
[SlavaYus/tinysms_qa_12m](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m).
На вход подаётся `question`, модель генерирует `code`. Обучение выполняется
на двух NVIDIA Tesla T4 в Kaggle, финальный инференс использует одну T4.
Дополнительные обучающие данные, решения других моделей и pretrained weights
в обучении не используются.

Все команды ниже выполняются из корня проекта, где находятся `train.py`
и `inference.py`. Локальные команды рассчитаны на bash/zsh в macOS или Linux.
Локальный компьютер готовит и отправляет задания; сами вычисления идут на Kaggle.

## Итоговое решение

Финальная конфигурация декодирования: **beam=4, максимум 128 новых токенов,
batch=32, length penalty=0.6**. Используются FP16 autocast, static KV-cache,
сортировка условий по длине и восстановление исходного порядка задач.
Компиляция decoder-step отключена.

| Показатель | Результат |
| --- | --- |
| Public pass@1 по результату Telegram-бота | **73,54%** |
| Полный тест | **1919 задач** |
| Максимум времени двух полных запусков на одной T4 | **58,24 с**, округлено вверх |
| Ограничение задания | 180 с на весь тест на одной T4 |
| Файл для отправки | `predictions-final-beam4-128.jsonl` |

Public pass@1 относится к публичной части теста. Он не является оценкой на
всех 1919 задачах. Локальные проверки синтаксиса и наличия `solve()` не заменяют
pass@1. Команды в этом README не отправляют сообщения Telegram-боту.

Измерения выполнены в двух отдельных Python-процессах:

| Повтор | Внешний процесс, с | `full_seconds`, с | `generation_seconds`, с |
| --- | ---: | ---: | ---: |
| 1 | 58,234157 | 57,312674 | 52,820446 |
| 2 | 57,838530 | 56,925355 | 52,361545 |

Для проверки лимита используется время **всего внешнего процесса**, а не
только цикла генерации. Оба запуска дали одинаковый файл ответов. Исходные
отчёты: [повтор 1](docs/benchmarks/final-beam4-128/repeat-0.json),
[повтор 2](docs/benchmarks/final-beam4-128/repeat-1.json),
[хеши модели и ответов](docs/benchmarks/final-beam4-128/reference.json).

Настройки для повторения этих замеров находятся в
[final_beam4_128.json](src/configs/inference_sweeps/final_beam4_128.json).
Веса, большие prepared-данные и сгенерированные ответы исключены из Git.
Для инференса нужен приложенный к решению inference-артефакт либо собственный
экспорт после обучения. Приватные артефакты Kaggle доступны только аккаунтам,
которым выдан доступ.

## Конфигурация обучения итоговой модели

Нужно явно выбирать `experiments=pre768_3epochs`. Обычный запуск `train.py`
без этого override использует меньшую модель исходного бейзлайна.

| Параметр | Значение |
| --- | --- |
| Архитектура | Encoder-decoder Transformer, 64 776 192 обучаемых параметра |
| `d_model`, `n_heads`, `ffn_dim` | 768, 12, 1920 |
| Encoder / decoder | 4 / 4 независимых слоя |
| Повторное применение слоёв | `layer_repeats=1` |
| Нормализация | Pre-Norm, обычный LayerNorm |
| FFN activation | ReLU |
| Позиции | Обучаемые абсолютные embeddings |
| Dropout | 0.1 |
| Embeddings | Общие для входа encoder/decoder; выходная проекция отдельная |
| Токенизатор | Byte-level BPE, словарь 8192, обучен на исходном train |
| Максимальная длина source / target при обучении | 256 / 256 токенов, включая EOS |
| Цель обучения | 3 полные эпохи |
| Лимит часов в тренере и launcher | Отключён: `max_hours=null` |
| Seed | 42 |
| GPU | 2 × NVIDIA Tesla T4, DDP |
| Вычисления | FP16 autocast + GradScaler; master weights и optimizer state в FP32 |
| Оптимизатор | Fused AdamW, LR=0.0003, betas=(0.9, 0.98) |
| Weight decay | 0.01 для матриц; 0 для одномерных параметров |
| LR schedule | 500 warmup steps, затем cosine decay до 10% исходного LR |
| Gradient clipping | Max norm 1.0 |
| Loss | Token cross-entropy, padding labels=-100, без label smoothing |
| Accumulation | 2 microbatches на каждой GPU |
| Batch budget | До 16 384 padded source+target токенов на GPU |
| Ограничение числа задач в microbatch | 256; фактическое число переменное |
| Length bucketing | `bucket_size=3200` на rank |
| DataLoader | 2 workers на rank, `spawn`, persistent workers |
| Validation | Каждые 500 успешных обновлений, весь val на rank 0 |
| Checkpoints | Каждые 1000 шагов, при улучшении val и при штатном завершении |

Loss нормализуется по числу непустых target-токенов во всём accumulation
window на обеих GPU. Это не среднее арифметическое отдельных microbatch losses.
Динамический бюджет считается как `B * (max_source_length + max_target_length)`
внутри батча; фиксированного числа задач в каждом обновлении нет.

Тренер строит детерминированный план каждой эпохи. Для исходных prepared-данных
и двух ranks планы содержат 62 911, 62 936 и 62 926 microbatches на rank.
План без FP16 overflow соответствует 94 387 optimizer steps. При пропуске
обновления GradScaler число успешных steps может быть меньше; конец обучения
определяется прохождением трёх эпох.

Основные конфиги: [code_baseline.yaml](src/configs/code_baseline.yaml),
[pre768_3epochs.yaml](src/configs/experiments/pre768_3epochs.yaml).
Фактические параметры конкретного запуска сохраняются в `config.yaml` и
`training_plan.json` в output, поэтому не нужно восстанавливать их по имени run.

В исходном GPU-запуске использовались Python 3.13.15, PyTorch 2.11.0+cu128
и CUDA 12.8. Kaggle обновляет образы. Для строгого сравнения сохраняйте
`environment.json` и используйте тот же образ окружения, если он доступен.
Seed и фиксированный датасет воспроизводят постановку; побитовое совпадение
повторного обучения на другом образе или GPU не обещается.

## 1. Локальное окружение и авторизация

[uv](https://docs.astral.sh/uv/pip/environments/) используется для локальной
подготовки данных и сборки пакетов. Локальная машина не обязана иметь CUDA.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` относится к локальному PyTorch-шаблону и закрепляет старый
PyTorch. **На Kaggle этот файл устанавливать не нужно.** GPU launchers и
ноутбук устанавливают [requirements-kaggle.txt](requirements-kaggle.txt),
который сохраняет CUDA-сборку PyTorch из окружения Kaggle.

Kaggle CLI устанавливается отдельно от проектного окружения:

```bash
uv tool install --python 3.12 "kaggle==2.2.5"
export PATH="$(uv tool dir --bin):$PATH"
export KAGGLE_TOOL_PYTHON="$(uv tool dir)/kaggle/bin/python"
kaggle auth login
kaggle kernels list --mine
```

Если команда `kaggle` не находится, проверьте `PATH` или откройте новый терминал
после `uv tool update-shell`. Переменная `KAGGLE_TOOL_PYTHON` указывает на Python
из tool-окружения, где установлен Kaggle SDK; проектный `.venv` может его не иметь.
[Документация CLI](https://github.com/Kaggle/kaggle-cli/tree/main/docs).

Задайте несекретные имена и пути; замените значения `your_...` своими:

```bash
export KAGGLE_OWNER="your_kaggle_username"
export COMET_WORKSPACE="your_comet_workspace"
export HW2_DATA_DIR="$(pwd)/../data"
export HW2_PREPARED_DIR="$HW2_DATA_DIR/prepared/code8k"
```

Автоматический запуск ниже доставляет метрики в Comet с локальной машины.
Он использует `COMET_API_KEY` из окружения либо `~/.comet.config`. При отсутствии
ключа можно один раз сохранить его через скрытый ввод:

```bash
python - <<'PY'
import configparser
from getpass import getpass
from pathlib import Path

path = Path.home() / ".comet.config"
config = configparser.ConfigParser(interpolation=None)
config.read(path)
if not config.has_section("comet"):
    config.add_section("comet")
key = getpass("COMET_API_KEY: ").strip()
assert key, "Ключ не должен быть пустым"
config.set("comet", "api_key", key)
with path.open("w") as stream:
    config.write(stream)
path.chmod(0o600)
print("Локальная конфигурация Comet сохранена")
PY
```

Ключ не вставляется в notebook, Hydra overrides или Kaggle metadata.
В автоматическом GPU-запуске `writer.mode=disabled`: метрики всегда пишутся
в JSONL/stdout, а локальный `comet_relay.py` передаёт их в Comet. Для этого пути
не требуется подключать Comet Secret к Kaggle notebook. Вариант без Comet
описан ниже как ручной запуск.

## 2. Скачать исходный датасет и подготовить данные

Используется точная ревизия HF-репозитория
`b2724689ff079e9c22efd548fe134bfb29036dbb`. Загрузите её в тот же cache layout,
который ожидает [CodeTaskDataset](src/datasets/code_task_dataset.py):

```bash
python - <<'PY'
import os
from getpass import getpass
from pathlib import Path
from datasets import load_dataset

root = Path(os.environ["HW2_DATA_DIR"])
root.mkdir(parents=True, exist_ok=True)
token = os.getenv("HF_TOKEN") or getpass("HF_TOKEN, либо Enter для публичного доступа: ").strip() or None
ds = load_dataset(
    "SlavaYus/tinysms_qa_12m",
    revision="b2724689ff079e9c22efd548fe134bfb29036dbb",
    token=token,
    cache_dir=str(root / "cache"),
)
assert len(ds["train"]) == 11838003
assert len(ds["test"]) == 1919
print(ds)
PY
```

`CodeTaskDataset` работает offline с Arrow-кэшем. Он не скачивает данные сам
и выдаёт ошибку, если нужного кэша нет. Если эта ревизия уже скачана, повторять
загрузку не нужно.

```bash
python prepare_data.py \
  data_dir="$HW2_DATA_DIR" \
  output_dir="$HW2_PREPARED_DIR" \
  num_proc=4
```

Подготовка включает следующие операции:

1. Разбиение исходного train по hash текста `question`, seed=42,
   `validation_fraction=0.001`. Все программы одного вопроса находятся
   в одном split.
2. Обучение byte-level BPE на подвыборке до 200 000 исходных train-строк
   с исключением validation-групп. В обучение токенизатора входят question и code.
   Словарь 8192, `min_frequency=2`, полный байтовый алфавит.
3. Токенизация и добавление EOS. Строки, где source или target превышает
   256 токенов, отбрасываются. Тексты и программы не переписываются;
   дополнительные таргеты не создаются. Удаление точных дублей отключено.
4. Сохранение `train`, `val`, `test`, `tokenizer.json`, `manifest.json`.
   В test остаются исходные вопросы и row_id; токенизация test выполняется
   внутри измеряемого инференса.

Для исходной конфигурации ожидаются:

| Split / проверка | Значение |
| --- | --- |
| Train после подготовки | 11 781 594 строки |
| Validation | 11 989 строк |
| Test | 1919 строк |
| Отброшено по длине | 44 420 строк |
| SHA256 `tokenizer.json` | `27c60831ad236373af954c059f9c2f6933121efecb6e882d1909f07a8f70d152` |

Проверка результата подготовки:

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["HW2_PREPARED_DIR"])
m = json.loads((root / "manifest.json").read_text())
assert [m["splits"][s]["rows"] for s in ["train", "val", "test"]] == [11781594, 11989, 1919]
sha = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
assert sha == m["tokenizer_sha256"]
assert sha == "27c60831ad236373af954c059f9c2f6933121efecb6e882d1909f07a8f70d152"
print("Данные и токенизатор проверены")
PY
```

Скрипт не перезаписывает готовый output. При изменении настроек укажите новый
каталог. Для сырых данных, prepared и ZIP одновременно требуется место на диске;
упаковка не удаляет исходные файлы.

## 3. Загрузить prepared-данные в Kaggle

```bash
python kaggle/package_data.py \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --output kaggle/build/repro-data \
  --owner "$KAGGLE_OWNER" \
  --slug nlp-hw2-data

kaggle datasets create -p kaggle/build/repro-data --keep-tabular
kaggle datasets status "$KAGGLE_OWNER/nlp-hw2-data"
```

Dataset создаётся приватным. Дождитесь `ready` до отправки обучения.
Сразу после загрузки возможна задержка публикации; устойчивый 403 требует
проверить владельца dataset и доступ к нему. Если dataset с теми же prepared
и tokenizer уже есть, загрузку можно пропустить. Для изменённых данных
используйте отдельный артефакт и соответствующие ссылки в metadata.

Launcher умеет работать как с распакованными `manifest.json/train/val/test`,
так и с одним `prepared.zip`. При нескольких подходящих datasets он завершится
с ошибкой, чтобы не выбрать вход случайно.

## 4. Обучить модель на двух T4

Соберите код, полный запрошенный конфиг, план эпох и metadata. Для этого шага
GPU не нужна; размеры модели проверяются на meta device.

```bash
python kaggle/prepare_pre768.py \
  --output kaggle/build/repro-train \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --owner "$KAGGLE_OWNER" \
  --code-slug nlp-hw2-code-pre768-repro \
  --comet-workspace "$COMET_WORKSPACE" \
  --comet-project NLP_HW2 \
  --inference-plan src/configs/inference_sweeps/final_beam4_128.json
```

Проверьте `kaggle/build/repro-train/requested-config.yaml`:
`norm_style=pre`, `d_model=768`, `n_heads=12`, `ffn_dim=1920`, слои 4/4,
`max_epochs=3`, `max_steps=null`, `max_hours=null`.
В `expected-training-plan.json` сохранится план для двух ranks.

Запуск через API:

```bash
"$KAGGLE_TOOL_PYTHON" kaggle/launch_prepared.py \
  --build kaggle/build/repro-train \
  --output artifacts/repro-train \
  --relay-python "$(pwd)/.venv/bin/python"
```

Команда загружает приватный code dataset, ждёт его готовности и отправляет
Kaggle script с accelerator `NvidiaTeslaT4` (**T4 x2**). Скрипт запускает
`torch.distributed.run --standalone --nproc_per_node=2`. Если среда не дала
две GPU, он завершится с понятной ошибкой. Перед обучением выполняется
техническая GPU-проверка; её временные веса не используются в обучении.

Сохраните `artifacts/repro-train/launch-state.json`: там фактический
`owner/kernel/version`, возвращённый API. Он может отличаться от ожидаемого
slug. Этот файл также позволяет повторным вызовом команды восстановить
мониторы без создания ещё одного обучения.

```bash
cat artifacts/repro-train/monitor-state.json
cat artifacts/repro-train/live-status.json
tail -f artifacts/repro-train/monitor.log artifacts/repro-train/live-stream.log
```

Обучение идёт на Kaggle независимо от локального терминала. Для живых метрик
Comet, автоматического запуска инференса и скачивания результатов локальная
машина должна оставаться включённой и в сети. После перезапуска сначала
проверьте статус облачной версии; не отправляйте новое обучение с нуля.
Старые PID в launch-state нужно сверить с реальными процессами перед их
ручной остановкой или удалением из state.

### Ручной запуск без Comet

После сборки пакета можно самостоятельно загрузить code dataset и notebook:

```bash
kaggle datasets create -p kaggle/build/repro-train/code_dataset --keep-tabular
kaggle kernels push -p kaggle/build/repro-train/kernel --accelerator NvidiaTeslaT4
```

Это альтернатива `launch_prepared.py`: не выполняйте обе команды отправки
для одного нового обучения. В ручном режиме launcher на GPU уже использует
`writer.mode=disabled`; Comet не обязателен, метрики сохраняются в outputs.
Статус и логи доступны в Kaggle; для CLI используйте фактический ref из ответа.

## 5. Checkpoints и продолжение после лимита Kaggle

Наш таймер часов отключён, но [платформа Kaggle](https://www.kaggle.com/docs/notebooks)
имеет собственный предел длительности сессии и GPU-квоту. Три эпохи могут
потребовать нескольких сессий. `max_hours=null` не отменяет этот предел.

В output обучения, в `artifacts/code-pre768-3epochs/`, находятся:

| Файл | Назначение |
| --- | --- |
| `last.pt` | Веса, optimizer, scheduler, GradScaler, RNG обоих ranks и позиция в sampler для resume |
| `best.pt` | Лучшие веса по validation loss для инференса |
| `model-step-*.pt` | Последние пять snapshots весов |
| `config.yaml`, `training_plan.json` | Фактическая конфигурация и план |
| `metrics.jsonl`, `environment.json` | Метрики и окружение |
| `tokenizer.json`, `data_manifest.json` | Токенизатор и данные |

Checkpoints записываются через временный файл и атомарную замену.
`last.pt` обновляется каждые 1000 шагов, при улучшении val и при штатном
завершении. Для resume нужен именно `last.pt`: `best.pt` не содержит optimizer.

Пока committed job работает, API может не публиковать его файлы.
После остановки сначала подтвердите наличие outputs и скачайте их.
Ручной Cancel не является гарантией сохранности; наличие пригодной копии
проверяется отдельно. Автомонитор запускает инференс после `COMPLETE`,
а при принудительном завершении обучение нужно продолжить отдельно.

Для скачивания подставьте фактическую версию своего запуска:

```bash
export TRAINING_REF="your_kaggle_username/your-training-slug/1"
kaggle kernels status "$TRAINING_REF"
kaggle kernels output "$TRAINING_REF" -p artifacts/recovered-training
```

Найдите в скачанном output `last.pt` и `best.pt`. Можно прикрепить outputs
предыдущей версии через `--resume-training "$TRAINING_REF"`, если Kaggle
разрешает такую зависимость. Надёжный альтернативный путь: собственный
приватный dataset с этими двумя файлами. Укажите реальные пути в блоке ниже:

```bash
python - <<'PY'
import json
import os
import shutil
from pathlib import Path

source = Path("/path/to/recovered/artifacts/code-pre768-3epochs")
out = Path("kaggle/build/resume-checkpoint-upload")
out.mkdir(parents=True, exist_ok=False)
for name in ["last.pt", "best.pt"]:
    shutil.copyfile(source / name, out / name)
(out / "dataset-metadata.json").write_text(json.dumps({
    "id": os.environ["KAGGLE_OWNER"] + "/nlp-hw2-resume-checkpoint",
    "title": "NLP HW2 resume checkpoint",
    "licenses": [{"name": "other"}],
}, indent=2))
PY
kaggle datasets create -p kaggle/build/resume-checkpoint-upload --keep-tabular
kaggle datasets status "$KAGGLE_OWNER/nlp-hw2-resume-checkpoint"
```

После готовности dataset:

```bash
python kaggle/prepare_pre768.py \
  --output kaggle/build/repro-resume \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --owner "$KAGGLE_OWNER" \
  --code-slug nlp-hw2-code-pre768-resume \
  --resume-dataset "$KAGGLE_OWNER/nlp-hw2-resume-checkpoint" \
  --comet-workspace "$COMET_WORKSPACE" \
  --inference-plan src/configs/inference_sweeps/final_beam4_128.json

"$KAGGLE_TOOL_PYTHON" kaggle/launch_prepared.py \
  --build kaggle/build/repro-resume \
  --output artifacts/repro-resume
```

Модель продолжает прежние три эпохи с сохранённой позиции. LR schedule не
начинается заново; прошедшие эпохи повторно не обучаются. Сохраняется и
предыдущий `best.pt`, если последующие веса не улучшили val. Тренер проверяет
совместимость архитектуры, manifest, batching, world_size и плана обучения.
Используйте те же две GPU, настройки batching и seed. Для каждой новой части
создавайте отдельные build/output каталоги и code slug.

## 6. Финальный инференс на одной T4

### Автоматический путь после обучения

При выборе `--inference-plan .../final_beam4_128.json` монитор после завершения
обучения отправляет отдельный inference notebook, привязанный к точной
версии training output. Он экспортирует `best.pt`, выполняет два полных
замера финальной конфигурации и скачивает результаты в `artifacts/repro-train`.
Если обучение продолжалось через resume, используйте output последней части.

В `inference/` появятся `predictions-beam4-128.jsonl`, `predictions.jsonl`,
`inference_report.json`, `decoding-sweep/summary.json`, отчёты повторов и
`inference_artifact/`. При финальном плане `predictions.jsonl` тоже содержит
beam=4. При старом плане `pre768.json` этот общий файл означал greedy-64;
не выбирайте итоговую посылку только по общему имени.

### Запуск готовых весов через notebook

[Ноутбук inference_kaggle.ipynb](inference_kaggle.ipynb) уже содержит итоговые
beam=4 / 128 / batch=32 / alpha=0.6. Создайте Kaggle Notebook, включите
**GPU T4 x2** и подключите один inference-артефакт через Add Input.
Процессу будет видна только первая T4 благодаря `CUDA_VISIBLE_DEVICES=0`;
вторая карта не участвует в вычислениях. Проверка `require_t4=true` требует
одну видимую T4 и полный test.

Артефакт должен иметь структуру:

```text
inference_artifact/
  model.pt
  tokenizer.json
  manifest.json
  code_bundle.zip
  test/
    ... Arrow-файлы и metadata Hugging Face Dataset ...
```

После автоматического инференса такой экспорт уже есть в output.
Для собственного checkpoint его можно создать локально:

```bash
python export_inference.py \
  --checkpoint /path/to/best.pt \
  --prepared-dir "$HW2_PREPARED_DIR" \
  --output kaggle/build/final-inference-artifact \
  --code-bundle kaggle/build/repro-train/code_dataset/code_bundle.zip \
  --dataset-id "$KAGGLE_OWNER/nlp-hw2-final-inference"

kaggle datasets create -p kaggle/build/final-inference-artifact \
  --dir-mode zip --keep-tabular
```

Укажите bundle именно той версии кода, с которой получен checkpoint.
После подключения в Kaggle должны быть доступны `model.pt` и папка `test`.
Если ZIP не распакован платформой, распакуйте его вручную и задайте путь
через `ARTIFACT_OVERRIDE` в первой ячейке notebook.

Запустите ячейки по порядку. Подготовка копирует данные на локальный диск
`/kaggle/working` и распаковывает проверенный bundle. Следующий отдельный
процесс загружает веса и генерирует `predictions.jsonl`.
Последняя ячейка проверяет 1919 строк, единственное строковое поле `code`,
одну T4 и лимит времени. Ответы возвращаются в исходном порядке.

Прямой эквивалент команды после подготовки артефакта и кода на Kaggle:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
python inference.py \
  checkpoint=/kaggle/working/hw2_artifact/model.pt \
  prepared_dir=/kaggle/working/hw2_artifact \
  output_path=/kaggle/working/predictions.jsonl \
  report_path=/kaggle/working/inference_report.json \
  device=cuda require_t4=true split=test limit=null \
  num_beams=4 max_new_tokens=128 batch_size=32 length_penalty=0.6 \
  amp=true cache_type=static sort_by_length=true compile_step=false
```

Эту команду выполняют из каталога распакованного проекта. В notebook это
`/kaggle/working/hw2_inference`. Команда не использует teacher-модель,
внешние API или исполнение сгенерированных программ.

## 7. Как повторить замеры времени

Внешний таймер стартует перед `subprocess.run()` и останавливается после
завершения дочернего процесса. В него входят импорты, инициализация CUDA,
загрузка весов, токенизация исходных test-вопросов, encoder/decoder,
декодирование, запись JSONL, диагностика и завершение процесса.

Установка пакетов, скачивание и предварительное копирование артефакта на
локальный диск, распаковка кода и экспорт weights выполнены до этого таймера.
Они не включены в приведённые 58,24 секунды. `generation_seconds` измеряет
только цикл генерации и не используется отдельно для подтверждения лимита.

Для двух свежих процессов используйте `benchmark_inference.py` с финальным
планом. Если запускаете код из старого экспортированного bundle, где нового
JSON ещё нет, создайте план отдельным файлом:

```bash
python - <<'PY'
import json
from pathlib import Path
plan = {
    "repeats": 2,
    "default_tag": "beam4-128",
    "variants": [{"tag": "beam4-128", "num_beams": 4,
                  "max_new_tokens": 128, "batch_size": 32, "length_penalty": 0.6}],
}
Path("/kaggle/working/final-beam-plan.json").write_text(json.dumps(plan, indent=2))
PY

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py \
  --checkpoint /kaggle/working/hw2_artifact/model.pt \
  --prepared-dir /kaggle/working/hw2_artifact \
  --output-dir /kaggle/working/final-benchmark \
  --plan /kaggle/working/final-beam-plan.json
```

Эта команда предназначена для отдельного benchmark после подготовки;
она запускает генерацию дважды. Для обычной посылки notebook делает один
запуск. Каталог benchmark должен быть новым для каждого сравнения.
Свежий Python-процесс не означает сброс дискового или драйверного кэша всей VM.

Проверьте `final-benchmark/summary.json`: `default_passes=true`,
`passes_all_repeats=true`, а оба `runs[*].seconds <= 180`.
В каждом `*.report.json` должны быть `examples=1919`, `full_test=true`,
`visible_cuda_devices=1`, `eligible_single_t4_benchmark=true`,
`passes_required_benchmark=true`. Поле `external_process_seconds` сохраняет
внешний замер; `full_seconds` и `generation_seconds` приведены отдельно.

Замер на CPU, другой GPU, неполном test или с заранее токенизированными
вместо исходных test-вопросов не подтверждает наш полный single-T4 протокол.
На другом запуске Kaggle время может отличаться, поэтому сохраняйте оба
отчёта, версии библиотек и фактические параметры decoding.

## 8. Проверить и отправить результат

Файл имеет ровно один JSON-объект на строку:

```json
{"code": "def solve():\n    return 42\n"}
```

Проверка перед отправкой, подставьте путь к своему файлу:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("/path/to/predictions.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines()]
assert len(rows) == 1919
assert all(set(r) == {"code"} and isinstance(r["code"], str) for r in rows)
print("Готово:", p, "задач:", len(rows))
PY
```

Отправьте сам `.jsonl` в [бот курса](https://t.me/hse_nlp_hw2_2026_bot).
Поля row_id, настройки генерации и отчёты времени в файл посылки не добавляются.
Все эти сведения хранятся отдельно. Полученный public pass@1 запишите вместе
с точной конфигурацией и хешем отправленного файла.

## Структура проекта и дополнительные материалы

| Файл / каталог | Назначение |
| --- | --- |
| [prepare_data.py](prepare_data.py) | Offline-подготовка и обучение токенизатора |
| [src/model/code_transformer.py](src/model/code_transformer.py) | Encoder-decoder и cached greedy/beam decoding |
| [src/trainer/code_trainer.py](src/trainer/code_trainer.py) | DDP/AMP, target-token weighting, epochs и resume |
| [inference.py](inference.py), [benchmark_inference.py](benchmark_inference.py) | Генерация и отдельные замеры |
| [export_inference.py](export_inference.py) | Экспорт weights, test, tokenizer и bundle |
| [kaggle/prepare_pre768.py](kaggle/prepare_pre768.py) | Сборка воспроизводимого запуска |
| [kaggle/launch_prepared.py](kaggle/launch_prepared.py) | Отправка и фоновые мониторы |
| [PIPELINE.md](PIPELINE.md) | Дополнительные сведения о пайплайне |
| [RESEARCH.md](RESEARCH.md), [EXPERIMENTS.md](EXPERIMENTS.md) | Исследование и журнал экспериментов |
| [INTERACTION_HISTORY.md](INTERACTION_HISTORY.md) | История разрешённых к записи взаимодействий |

Сгенерированные `kaggle/build/`, `artifacts/`, checkpoints и JSONL не входят
в Git. Сохраняйте frozen bundles, фактические configs, manifests и отчёты
вместе с результатами эксперимента.

Проект основан на [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).
Исходная документация шаблона и ссылки на его авторов сохранены в
[docs/template_README.md](docs/template_README.md). Лицензия проекта: [MIT](LICENSE).
