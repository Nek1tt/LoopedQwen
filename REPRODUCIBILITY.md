# Воспроизводимость

Документ описывает три уровня проверки: аудит опубликованных таблиц без GPU, повторную оценку лучшего checkpoint и полное обучение с нуля.

## 1. Требования

- Python 3.10 или новее;
- для обучения и полной оценки рекомендуется GPU NVIDIA с поддержкой `float16`;
- примерно 30 ГБ свободного места для окружения, данных и checkpoint;
- Git и Git LFS не требуются для клонирования этого репозитория: веса хранятся отдельно.

```bash
git clone https://github.com/Nek1tt/LoopedQwen.git
cd LoopedQwen
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Проверка установки:

```bash
python scripts/sanity_check.py
pytest -q
python scripts/verify_results.py
```

## 2. Проверка опубликованных результатов без обучения

Команда ниже читает все сохранённые JSON/CSV, проверяет ожидаемые размеры плотных оценок и ключевые числа финального сравнения:

```bash
python scripts/verify_results.py
```

Ожидаемый итог — сообщение об успешной проверке семи экспериментов. Этот режим не требует PyTorch и GPU.

## 3. Загрузка лучшего checkpoint

Модель, токенизатор, пользовательский файл архитектуры и карточка опубликованы здесь:

[Nek1tt/LoopedQwen-fixed-ma-seed43](https://huggingface.co/Nek1tt/LoopedQwen-fixed-ma-seed43)

```bash
pip install "huggingface_hub>=0.34"
hf download Nek1tt/LoopedQwen-fixed-ma-seed43 \
  --local-dir outputs/final/best_hf
```

Файл конфигурации содержит `auto_map`, поэтому модель также можно загрузить непосредственно через Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "Nek1tt/LoopedQwen-fixed-ma-seed43"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
print(model.num_parameters())
```

Ожидаемое число параметров: `5_670_402`.

## 4. Повторная оценка

Для точного совпадения с опубликованными числами нужен исходный `val.bin`: 1 000 000 токенов `uint16`, то есть 2 000 000 байт. Этот файл не опубликован в Git или модельном репозитории.

Если исходный `val.bin` доступен, поместите его в `data/val.bin` и выполните плотную оценку:

```bash
mkdir -p data
# Поместите исходный val.bin в data/val.bin.

python scripts/eval.py \
  --checkpoint outputs/final/best_hf \
  --val-file data/val.bin \
  --loops 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --batches 100 \
  --batch-size 2 \
  --operator-diagnostic-batches 5 \
  --output outputs/final/dense_eval.json
```

Если исходного `val.bin` нет, публичный checkpoint всё равно позволяет повторить протокол оценки на заново подготовленном validation-потоке FineWeb. Для этого используйте токенизатор из опубликованного checkpoint: новый BPE-токенизатор для этой проверки обучать не нужно, поскольку соответствие token ID в новом словаре может отличаться от токенизатора, на котором обучались веса модели.

Сначала подготовьте новый validation-поток:

```bash
python scripts/prepare_data.py \
  --tokenizer outputs/final/best_hf \
  --output-dir data/repro_eval \
  --train-tokens 1 \
  --val-tokens 1000000
```

`--train-tokens 1` создаёт минимальный неиспользуемый training-файл, поскольку текущий интерфейс `prepare_data.py` требует положительное число train tokens. Для повторной оценки используется только `data/repro_eval/val.bin`.

Проверьте размер нового validation-файла:

```bash
python -c "from pathlib import Path; p=Path('data/repro_eval/val.bin'); print(p.stat().st_size)"
```

Ожидаемый размер: `2_000_000` байт.

После этого выполните ту же плотную оценку:

```bash
python scripts/eval.py \
  --checkpoint outputs/final/best_hf \
  --val-file data/repro_eval/val.bin \
  --loops 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
  --batches 100 \
  --batch-size 2 \
  --operator-diagnostic-batches 5 \
  --output outputs/final/dense_eval_repro.json
```

Оценщик создаёт генератор со seed 1234 отдельно для каждой глубины, поэтому все глубины получают одни и те же пакетные выборки. Опубликованные контрольные значения для seed 43:

| Метрика | Значение |
|---|---:|
| Лучший PPL | 905,82 на глубине 11 |
| Средний PPL, глубины 8–24 | 912,40 |
| PPL@16 | 909,87 |
| PPL@32 | 944,76 |

При исходном бинарном потоке небольшие различия последнего десятичного знака возможны из-за версии CUDA и режима вычислений. На заново подготовленном потоке абсолютные значения могут отличаться из-за состояния потокового FineWeb; в этом режиме воспроизводится протокол оценки, а не точные исторические PPL.

## 5. Подготовка данных для обучения с нуля

Этот раздел нужен только для полного обучения с нуля и не требуется для повторной оценки публичного checkpoint из раздела 4.

Исходный токенизатор Qwen не используется: его словарь нарушил бы ограничение на число параметров. Для нового обучения сначала обучается байтовый BPE-словарь размером 16 000:

```bash
python scripts/train_tokenizer.py \
  --output-dir tokenizer \
  --vocab-size 16000 \
  --documents 100000
```

Затем поток FineWeb упаковывается в непересекающиеся бинарные последовательности:

```bash
python scripts/prepare_data.py \
  --tokenizer tokenizer \
  --output-dir data \
  --train-tokens 100000000 \
  --val-tokens 1000000
```

`data/metadata.json` фиксирует число токенов и параметры подготовки. Для строгой побитовой репликации используйте токенизатор и бинарные файлы из исходного пакета, потому что потоковый источник FineWeb может измениться независимо от кода репозитория. Подготовка заново воспроизводит метод и бюджет, но не гарантирует тот же порядок документов.

Проверьте размеры и хеши локальных файлов перед дорогостоящим запуском. В опубликованном пакете находится `artifact_hashes.json` с эталонными SHA-256.

## 6. Обучение финальной модели

Точная конфигурация лучшего запуска:

`experiments/007_alternating_operators/configs/fixed_ma_seed43.yaml`

```bash
python scripts/train.py \
  --config experiments/007_alternating_operators/configs/fixed_ma_seed43.yaml
```

Настройки, которые должны оставаться неизменными для прямого сравнения:

- seed 43;
- глубина, равномерно выбранная из 8–24;
- intermediate losses на глубине 8, в midpoint и на конечной глубине;
- веса 0,25 / 0,25 / 0,5;
- порядок MLP → Attention;
- пакет 2 × 32 шага накопления градиента;
- длина контекста 512;
- 305 шагов оптимизатора и 9 994 240 токенов;
- скорость обучения `3e-4`, 30 шагов разогрева, затухание до `3e-5`;
- ограничение нормы градиента 1,0.

Выходной каталог:

```text
outputs/experiments/007_multiseed/fixed_ma_seed43/
  best_hf/          лучший checkpoint
  last_state.pt     состояние для продолжения
  metrics.jsonl     журнал обучения и проверки
  run_config.json   фактически использованная конфигурация
```

Продолжение прерванного запуска:

```bash
python scripts/train.py \
  --config experiments/007_alternating_operators/configs/fixed_ma_seed43.yaml \
  --resume outputs/experiments/007_multiseed/fixed_ma_seed43/last_state.pt
```

Выбор случайной глубины зависит от глобального номера микропакета, поэтому после корректного продолжения последовательность глубин сохраняется.

## 7. Повторение парного сравнения для трёх seed

Для стандартного порядка используйте конфигурации эксперимента 005, для MLP → Attention — конфигурации эксперимента 007. Проведите запуски с seed 42, 43 и 44. Не меняйте данные, бюджет, критерий выбора checkpoint или набор глубин проверки.

После обучения выполните плотную оценку каждого checkpoint на глубинах 4–32 и по 100 пакетных выборок. Итоговые эталонные таблицы находятся в:

```text
experiments/005_anytime_depth/results/multiseed/
experiments/007_alternating_operators/results/multiseed/
```

Критерий успешного повторения — преимущество MLP → Attention у каждого seed и на каждой глубине. В исходном прогоне получено 87 побед из 87.

## 8. Запуск в Colab

Все основные сценарии поддерживают `--in-process`: обучение и оценка выполняются в активном ядре, а `tqdm` показывает ход выполнения, время до завершения, функцию потерь, скорость, норму градиента и выбранную глубину.

```python
%run experiments/007_alternating_operators/run.py \
  --variant fixed_ma \
  --resume \
  --in-process
```

Блокноты не включены в Git, потому что они дублируют команды и добавляют сохранённый вывод. Источником истины остаются конфигурации YAML и сценарии Python.

## 9. Что можно и нельзя считать воспроизведённым

- `verify_results.py` подтверждает внутреннюю согласованность опубликованных файлов.
- Оценка публичного checkpoint на исходном `val.bin`, если он сохранён у исполнителя, воспроизводит итоговые метрики.
- Оценка публичного checkpoint на заново подготовленном FineWeb validation-потоке с токенизатором из этого checkpoint воспроизводит протокол оценки, но не гарантирует точного совпадения исторических PPL.
- Обучение на исходных бинарных потоках проверяет полный запуск.
- Повторная потоковая выгрузка FineWeb для обучения с нуля проверяет метод, но не является строгой репликой данных без закреплённой версии и порядка документов.
- Процентный выигрыш на более крупной модели пока является гипотезой масштабирования, а не измеренным фактом.
