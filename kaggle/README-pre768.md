# Следующий запуск: Pre-LN, 768, три эпохи

Актуальная инструкция воспроизведения финального решения находится в
[основном README](../README.md). Финальный decoding — beam=4/128, batch=32,
alpha=0.6; для сборки используется `--inference-plan src/configs/inference_sweeps/final_beam4_128.json`.
Ниже сохранены сведения о первоначальном запуске и сравнении шести вариантов.

Актуальный пакет: `kaggle/build/pre768-unlimited`. Пользователь разрешил
отправку обучения и снял наш лимит времени. Старый пакет
`kaggle/build/pre768-3epochs` сохраняет прежнюю настройку 12 часов;
для нового запуска он не используется.

Фактически запущенная версия:
[nlp-hw2-pre768-three-epochs-train, API-версия 1](https://www.kaggle.com/code/andrewkhoroshilov/nlp-hw2-pre768-three-epochs-train).
Kaggle сформировал slug первого notebook из title; точная API-ссылка
хранится в artifacts/pre768-3epochs/launch-state.json.
Comet: [текущий run](https://www.comet.com/had-a42/nlp-hw2/4b69071818a6441bb60f18edebf5da0e).

Модель: encoder-decoder, 64 776 192 параметра, Pre-LN + обычный LayerNorm,
d_model=768, heads=12, FFN=1920, 4 encoder / 4 decoder слоя, ReLU,
learned absolute positions, dropout=0.1, layer_repeats=1. Веса случайные;
tokenizer и подготовленные данные прежние. Бюджет 16 384 padded-токена/GPU,
accumulation=2, AdamW lr=3e-4, warmup=500.

Цель — 3 эпохи: планы по 62 911 / 62 936 / 62 926 microbatches на rank,
всего 94 387 optimizer steps без FP16 overflow. Это цель, не обещание
завершить все эпохи в одной сессии: остаются ограничения длительности
GPU-сессии и квота Kaggle. Наши trainer.max_hours и лимит launcher отключены
через null, scheduler настроен на три эпохи. Мониторы используют --max-hours=0
(без своего срока завершения), они не ограничивают обучение.
На GPU сохраняется фактический `training_plan.json`, progress по батчам,
полный runtime config и checkpoints. Resume проверяет совместимость плана.

## Запуск готового пакета

Из корня проекта:

```bash
cd /Users/hadhad/HSE/NLP/HW2/NLP_HW2
/Users/hadhad/.local/share/uv/tools/kaggle/bin/python kaggle/launch_prepared.py
```

Эта команда загружает приватный код, отправляет новый job
и запускает фоновые процессы (фактический notebook ref берётся из ответа API):
получение stdout/метрик, доставку в Comet и монитор обучения с автоматическим
инференсом. Подготовленные данные `andrewkhoroshilov/nlp-hw2-data` используются
повторно. Существующий запущенный notebook не изменяется.

Перед основным обучением отдельный процесс на одной T4 проверит
forward/backward при [32,256,256] и скорость greedy/beam=2/4 на всех 1919
test-условиях с принудительными 128 шагами. Его случайные веса и генерации
не сохраняются как обучающие данные и не используются в основном запуске.
Это технический preflight; официальный замер выполняется после обучения
на экспортированном checkpoint в новых процессах.

Авторизация Kaggle уже должна работать в установленном CLI. Comet relay
использует существующий локальный `COMET_API_KEY` или `~/.comet.config`.
Ключ не попадает в notebook, metadata, code bundle или журналы. Для живых
логов Comet и автоматического запуска инференса Mac должен оставаться
включённым и подключённым к сети. Обучение выполняется независимо на Kaggle.

Артефакты появятся в `artifacts/pre768-3epochs`:

- `launch-state.json` — точная версия обучения и PID мониторов.
- `live-status.json`, `live-metrics.jsonl`, `kaggle-live.log` — прогресс.
- `comet-relay-state.json` — ссылка на Comet и состояние доставки метрик.
- `monitor-state.json` — обучение, инференс, скачивание, финальный статус.
- `inference/` — portable model, отчёты и отдельные файлы предсказаний.

Статус в терминале:

```bash
tail -f artifacts/pre768-3epochs/monitor.log artifacts/pre768-3epochs/live-stream.log
```

Повторный вызов `launch_prepared.py` с теми же build/output использует
сохранённую версию обучения и восстанавливает недостающие мониторы.
Не изменять frozen code bundle после отправки. Для исправленного кода
подготовить отдельный пакет/slug и отдельный каталог артефактов.

Если Kaggle прервёт сессию, продолжение использует собственный last.pt
с optimizer/scaler/scheduler/RNG и позицией в sampler, сохраняя план трёх
эпох. Для подготовки продолжения с точной версией в качестве input:

```bash
python kaggle/prepare_pre768.py \
  --output kaggle/build/pre768-resume2 \
  --code-slug nlp-hw2-code-pre768-resume2 \
  --resume-training andrewkhoroshilov/nlp-hw2-pre768-three-epochs-train/VERSION
/Users/hadhad/.local/share/uv/tools/kaggle/bin/python kaggle/launch_prepared.py \
  --build kaggle/build/pre768-resume2 \
  --output artifacts/pre768-resume2
```

VERSION берётся из launch-state.json. Сначала проверить наличие опубликованного
last.pt. Этот механизм подготовлен; автоматическое создание новой сессии
при исчерпании квоты не включено. Инференс текущего монитора запускается
после статуса COMPLETE; сбой требует проверки состояния и checkpoint.

## Варианты инференса

План: `src/configs/inference_sweeps/pre768.json`. Одна T4 видна процессу;
каждый вариант запускается дважды, каждый раз в новом Python-процессе.

| Тег | Beam | Максимум токенов | Batch задач |
| --- | --- | --- | --- |
| greedy-64 | 1 | 64 | 128 |
| greedy-128 | 1 | 128 | 128 |
| beam2-64 | 2 | 64 | 64 |
| beam2-128 | 2 | 128 | 64 |
| beam4-64 | 4 | 64 | 32 |
| beam4-128 | 4 | 128 | 32 |

Length penalty=0.6, static KV-cache, сортировка условий по длине с
восстановлением исходного порядка. Beam возвращает одну программу на задачу.
Формат — ровно 1919 строк `{"code": "..."}`. Увеличение beam компенсируется
меньшим batch по числу одновременных траекторий; это не обещает равной памяти.

Сохраняются полный wall time процесса, внутренний таймер, синтаксическая
корректность и доля без EOS. Кандидат проходит ограничение только при обоих
полных замерах <=180 секунд и корректном формате. Все файлы сохраняются
отдельно: `predictions-greedy-64.jsonl`, `predictions-beam2-128.jsonl` и т.д.
Сводка — `decoding-sweep/summary.json` в output Kaggle и скачанных артефактах.
В `predictions.jsonl` по умолчанию копируется greedy-64; это технический default,
не вывод о лучшем pass@1. Качество beam и лимитов длины нужно оценивать отдельно.

Повторить sweep для уже экспортированного checkpoint на одной T4:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py \
  --checkpoint /path/to/inference_artifact/model.pt \
  --prepared-dir /path/to/inference_artifact \
  --output-dir /path/to/new-sweep \
  --plan src/configs/inference_sweeps/pre768.json
```

Для сборки нового пакета после изменений:

```bash
python kaggle/prepare_pre768.py --output kaggle/build/pre768-unlimited-v2
```

Это только локальная сборка. Не переиспользовать старый remote code dataset
для изменившегося bundle без явного создания новой версии или нового slug.
