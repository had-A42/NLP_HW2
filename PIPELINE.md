# HW2: подготовка, обучение и инференс

Код сохраняет структуру шаблона: `src/`, Hydra, `train.py`, `inference.py`.
Новый путь включён по умолчанию. Старый пример классификации остаётся доступен
через `train.py -cn=baseline` и `inference.py -cn=inference`.

## Окружение

Для локальных проверок использованы Python 3.12.14, PyTorch 2.2.0,
NumPy 1.26.4, datasets 4.4.1, PyArrow 21.0.0, tokenizers 0.22.2,
Hydra 1.3.2. Зависимости находятся в `requirements.txt`.
В Kaggle используем `requirements-kaggle.txt`, который сохраняет имеющийся
CUDA-enabled PyTorch. Не устанавливаем поверх него macOS/CPU-окружение.

Команды ниже выполняются из корня `HW2/NLP_HW2` в активном Python-окружении.
Для Comet нужен `COMET_API_KEY` в окружении или Kaggle Secrets. Значения
workspace/project уже заданы: `had-a42` / `NLP_HW2`. Для локальных проверок без
сети используйте `writer.mode=disabled`; `offline` сохраняет Comet локально.

## Подготовка данных

```bash
python prepare_data.py
```

По умолчанию читается существующий `../data/cache`; скачивание не выполняется.
Результат — `../data/prepared/code8k/` с train, val, test, tokenizer.json,
manifest.json и полным конфигом подготовки. Существующая папка не перезаписывается.

Стабильный hash исходного `question` относит 0.1% групп вопросов в val. Все
программы одного вопроса попадают в один сплит. BPE обучается только на train,
на воспроизводимой выборке до 200 000 строк; byte-level кодирование сохраняет
отступы и Unicode. Полный train затем токенизируется батчами.

Исходные таргеты не переписываются. Пустые значения вызывают ошибку с row_id.
Примеры, превышающие заданные source/target limits, отбрасываются из train/val
с подсчётом в manifest; тест не отбрасывается и не обрезается. Если тестовое
условие не помещается, подготовка выдаёт ошибку. Слишком короткие лимиты следует
изменить до запуска большого обучения. В manifest есть статистика длин.

Другие варианты подготовки:

```bash
python prepare_data.py tokenizer.vocab_size=16384 output_dir=../data/prepared/code16k
python prepare_data.py deduplicate=true output_dir=../data/prepared/code8k-dedup
```

Дедупликация удаляет только одинаковые question/code по 128-битному отпечатку;
разные программы одного вопроса сохраняются. Она выключена по умолчанию.
`source_limit=N` берёт первые N исходных строк. Для маленькой технической
проверки можно увеличить validation_fraction; это обязательно отражается в
конфиге, такая проверка не сравнивается с полным запуском.

## Обучение

```bash
python train.py
```

На машине с двумя CUDA GPU:

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

Бейзлайн: d_model=512, 8 голов по 64 компоненты, 4 encoder / 4 decoder слоя, FFN=1280,
Post-LayerNorm, ReLU, общие encoder/decoder token embeddings и отдельный LM
head. Все веса случайные. Размер словаря, специальные токены и пределы длин
берутся из подготовленного артефакта и сохраняются в resolved config.
При словаре 8192 и пределах позиций 256 модель содержит 31 647 744 параметра:
на 20 015 104 больше исходного варианта. Прежний вариант на 11 632 640 параметров
сохранён как `experiments=small_11m` для контроля. Качество обоих вариантов
и время на одной T4 пока не измерены.

AdamW, warmup + cosine decay, clipping и FP16 AMP включены в CUDA-режиме.
Основной CPU-режим использует FP32. Для train и validation по умолчанию
выбран динамический бюджет `dataloader.token_budget=16384` padded-токенов
на GPU: `число задач * (max source length + max target length) <= 16384`.
Количество задач переменное; `max_batch_size=256` ограничивает его сверху.
В пуле до `bucket_size=3200` задач на rank примеры сортируются по сумме длин,
затем распределяются между ranks по очереди, а готовые глобальные батчи
перемешиваются. На двух GPU размер пула равен 6400 задачам.

`accumulation_steps=2` означает два микробатча на шаг оптимизатора. На двух
GPU это до 65 536 padded-токенов суммарно за шаг, число полезных target-токенов
и задач меняется. Loss и градиенты взвешиваются по общему числу непустых
target-токенов всего accumulation window, включая оба ranks. Логируются
`target_tokens_per_step`, `source_tokens_per_step`, `padded_tokens_per_step`,
`examples_per_step`, `microbatches_per_step` и `padding_fraction`.

Эпоха имеет одинаковое число микробатчей на всех ranks, без повторов строк;
при нечётном числе строк на двух GPU отбрасывается одна случайная строка.
При одном rank, включая validation, хвост сохраняется полностью. План эпохи
кэшируется в компактных массивах; resume использует epoch и start_batch.

DDP использует отдельные части глобальных батчей; Comet, валидация и запись
файлов выполняются на rank 0. Изменение seed/конфига для новой гипотезы должно
получить новое run_name. Существующий завершённый каталог не перезаписывается.

Основные ограничения задаются через `trainer.max_steps` и `trainer.max_hours=4`.
Остановка по времени проверяется после обновления; финальная валидация и
сохранение требуют дополнительного времени. Kaggle launcher отдельно
контролирует весь свой четырёхчасовой бюджет, включая setup, и оставляет запас
для завершения дочернего обучения.

В `saved/<run_name>/` сохраняются config.yaml, environment.json, metrics.jsonl,
токенизатор, manifest данных, `last.pt` с полным состоянием, `best.pt` по val NLL
и последние пять model-only snapshots. NLL и AST-метрики не являются pass@1.

Возобновление с прежними данными, архитектурой и world_size:

```bash
torchrun --standalone --nproc_per_node=2 train.py trainer.resume_from=saved/code-baseline/last.pt
```

Восстанавливаются optimizer, scheduler, scaler, позиция в последовательности
батчей и RNG каждого rank. Для resume нужен `last.pt`, а не усреднённые или
экспортированные веса. Проверяются настройки batching, seed, accumulation и
world_size; менять их при точном resume нельзя. Старые checkpoints без
`batching_config` требуют исходной версии кода. При возобновлении учитывается
уже потраченное время.

## Отдельные абляции

```bash
python train.py experiments=bucketed
python train.py experiments=fixed_batches
python train.py experiments=shallow_decoder
python train.py experiments=weight_tying
```

Остальные конфиги в `src/configs/experiments/`: `lr_low`, `lr_high`,
`warmup_short`, `pre_norm`, `rms_norm`, `swiglu`, `rope`, `gqa`, `bpe16k`,
`deduplicated`. Все варианты имеют отдельное run_name. Они реализуют переключение
одного фактора, не заявляют улучшение качества. Варианты данных сначала требуют
соответствующего подготовленного артефакта. GQA использует совместимый с
PyTorch 2.2 repeat-interleave fallback; ускорение на T4 пока не подтверждено.

`experiments=bucketed` и `experiments=fixed_batches` — контрольные варианты
с фиксированными 32 задачами на GPU, с группировкой по длинам и без неё.
Все остальные варианты наследуют динамический бюджет 16 384.

## Kaggle API

Авторизация CLI (один раз):

```bash
uv tool install kaggle
kaggle auth login
```

Также нужен доступ аккаунта к GPU и свободная квота. Авторизация и токены
описываются в [официальной инструкции](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
Значения ключей не добавляем в исходники или metadata JSON.

`andrewkhoroshilov` — предварительное имя аккаунта из обсуждения; проверьте его
перед отправкой или передайте `--owner`. Эти команды готовят локальные файлы:

```bash
python kaggle/prepare_kernel.py --output kaggle/build/dynamic-16k --owner andrewkhoroshilov
python kaggle/package_data.py --prepared-dir ../data/prepared/code8k --output kaggle/build/data --owner andrewkhoroshilov
```

Код упаковывается по списку допустимых исходников, без `.env`, ключей,
ноутбуков с пользовательскими выводами и локальных checkpoints. Сохраняется
manifest SHA256 исходников. Во входном data-артефакте находятся только
подготовленные данные из разрешённого датасета и параметры подготовки.

Ниже шаги для первой загрузки training inputs и запуска пользователем.
На 2026-10-23 пакет кода kaggle/build/dynamic-16k уже собран; повторять его
подготовку в тот же output нельзя. Полный data-пакет ещё нужно собрать.

```bash
cd /Users/hadhad/HSE/NLP/HW2/NLP_HW2
export PATH="$HOME/.local/bin:$PATH"
.venv/bin/python kaggle/package_data.py --prepared-dir ../data/prepared/code8k --output kaggle/build/data --owner andrewkhoroshilov
```

Упаковка большого архива может долго работать без вывода. Затем создайте
приватные inputs и проверьте, что оба получили статус ready:

```bash
kaggle datasets create -p kaggle/build/dynamic-16k/code_dataset
kaggle datasets create -p kaggle/build/data
kaggle datasets status andrewkhoroshilov/nlp-hw2-code
kaggle datasets status andrewkhoroshilov/nlp-hw2-data
```

После ready создайте kernel без выполнения:

```bash
kaggle kernels push -p kaggle/build/dynamic-16k/kernel --no-run
```

Перед первым обучением откройте созданный приватный kernel в редакторе Kaggle,
добавьте и разрешите ему секрет `COMET_API_KEY` через Add-ons → Secrets.
Целевой kernel: https://www.kaggle.com/code/andrewkhoroshilov/nlp-hw2-train.
Если секрет уже создан, проверьте переключатель доступа именно в этом kernel.
Официальная CLI-документация указывает, что создание Secrets не поддерживается
через CLI: [инструкция](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels).
После этого запускаем и проверяем статус через API:

```bash
kaggle kernels push -p kaggle/build/dynamic-16k/kernel --accelerator NvidiaTeslaT4 -t 14400
kaggle kernels status andrewkhoroshilov/nlp-hw2-train
kaggle kernels output andrewkhoroshilov/nlp-hw2-train -p artifacts/kaggle-dynamic-16k
```

Первый полный запуск пользователь выполняет сам. Прогресс обучения смотрите
в Comet, workspace had-a42, project NLP_HW2, run code-baseline; console logs —
в Kaggle на странице выполняющейся версии. `kernels status` показывает
состояние задачи, а не число optimizer steps. Команда output предназначена
для скачивания сохранённых файлов после завершения, включая checkpoints.
Ограничения: две T4, FP16 AMP, 16 384 padded-токена на GPU, accumulation=2,
до 20 000 обновлений или четырёхчасового бюджета launcher с setup.

Datasets создаются приватными по умолчанию; не добавляйте `--public`.
При следующих версиях inputs используйте `kaggle datasets version` с
сообщением версии. Launcher понимает входные zip и распакованные inputs,
проверяет две видимые GPU, читает `COMET_API_KEY` через Kaggle Secrets и
запускает DDP. Сам code bundle включает конфиги и не требует внешнего checkout.
Для другой гипотезы используйте новые output/kernel slug и
`--override experiments=NAME`. При необходимости задайте `--max-hours`.

## Усреднение и экспорт

```bash
python average_checkpoints.py saved/code-baseline/model-step-00001000.pt saved/code-baseline/model-step-00002000.pt saved/code-baseline/model-step-00003000.pt --output saved/code-baseline/averaged.pt
python export_inference.py --checkpoint saved/code-baseline/best.pt --prepared-dir ../data/prepared/code8k --output artifacts/inference --code-bundle kaggle/build/dynamic-16k/code_dataset/code_bundle.zip --dataset-id andrewkhoroshilov/nlp-hw2-model
```

Усредняются только совместимые веса в FP32, корректно обрабатываются tied
weights; получается одна модель. Экспорт исключает optimizer/RNG и train/val,
оставляя model.pt, токенизатор, manifest и исходные тестовые вопросы. Для
воспроизводимости используйте code bundle того запуска, на котором обучены веса.

## Инференс и посылка

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py checkpoint=artifacts/inference/model.pt prepared_dir=artifacts/inference device=cuda require_t4=true
```

По умолчанию — greedy, 64 новых токена, статический KV-cache, batch size 64,
сортировка по длине условия с восстановлением исходного порядка. Encoder и
cross-attention K/V считаются один раз на батч. Выходной слой применяется к
одной новой позиции, а не ко всему префиксу. PAD/BOS/UNK не выдаются как код.
Есть динамический cache для сравнения и экспериментальная компиляция шага;
она выключена по умолчанию и зависит от версии PyTorch/Python/CUDA.

`predictions.jsonl` содержит ровно по одному объекту `{"code": "..."}` на задачу.
В отчёте отдельно указаны холодное время до готового файла, время генерации,
длины, отсутствие EOS и AST-метрики. Только полный тест из 1 919 задач на одной
видимой T4 может получить флаг успешного обязательного замера. CPU/limit-прогоны
его не получают. Официальный предел 180 секунд не меняется настройкой
технического `max_seconds`. Реальное время на T4 пока не измерено.

```bash
python benchmark_inference.py --checkpoint artifacts/inference/model.pt --prepared-dir artifacts/inference --output-dir benchmarks/unsorted
python benchmark_inference.py --checkpoint artifacts/inference/model.pt --prepared-dir artifacts/inference --output-dir benchmarks/sorted --sort-by-length
```

Каждый вариант запускается в новом процессе, три повтора по умолчанию.
Дисковый cache компилятора автоматически не очищается; сравнивайте стоимость
первого вызова отдельно. Дополнительные флаги: `--cache-type dynamic`,
`--compile-step`, `--max-new-tokens`. Для отдельной проверки beam=2:
`python inference.py num_beams=2 ...`. Beam не включён в бейзлайн.

Для сдачи подготовлен `inference_kaggle.ipynb`; подключите экспортированный
артефакт и используйте одну видимую T4. Исходный пользовательский `run.ipynb`
сохранён. Установку пакетов notebook показывает отдельно от таймера процесса;
точные границы официального замера следует сверить с протоколом преподавателей.

## Проверки

```bash
python -m unittest discover -s tests -v
```

Проверяются cache/full-prefix parity, causal/padding masks, обратимость BPE,
сдвиг labels, группировка вопросов, DDP-срезы батчей, EOS/beam, совместимость
усреднения и точное resume с dropout. Fixtures в тестах используются только
для технических проверок и не входят в обучающие данные решения.

Квантизация, дистилляция и другие методы сжатия не реализованы.
