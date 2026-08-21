# Эксперимент 002: dropout на входе повторных loops

## Гипотеза

В baseline поздние loops вносят почти коллинеарные обновления, а residual stream продолжает расти. Проверяем, заставит ли случайное зануление части recurrent hidden state перед повторными проходами восстанавливать и перераспределять информацию.

Сравниваются три модели с одинаковой архитектурой (5.67M параметров, 2 общих слоя, 8 loops) и бюджетом до 10M обучающих токенов на запуск:

- `baseline_r8`: контроль без dropout;
- `dropout_p010_r8`: input dropout `p=0.10`;
- `dropout_p020_r8`: input dropout `p=0.20`.

Dropout применяется начиная со второго loop только во время обучения. Первый проход по embeddings не изменяется, а при evaluation маска отключена. Dropout не добавляет параметров.

Baseline переиспользуется из Experiment 001 и повторно не обучается.

## Результаты

Все модели обучались с 8 loops и оценивались с 1, 2, 4, 8, 12 и 16 loops.

| Вариант | Dropout | Loss@8 | PPL@8 | PPL@16 |
|---|---:|---:|---:|---:|
| `baseline_r8` | 0.00 | 7.0257 | 1125.2 | **1145.8** |
| `dropout_p010_r8` | 0.10 | **6.9908** | **1086.6** | 1308.7 |
| `dropout_p020_r8` | 0.20 | 7.1311 | 1250.3 | 1533.8 |

`p=0.10` улучшил perplexity на обученной глубине 8 loops на 3.43%, но не сделал дополнительные итерации полезными: на 12 и 16 loops он хуже baseline на 8.41% и 14.22%. `p=0.20` ухудшил качество на всех проверенных глубинах.

Умеренный dropout уменьшает коллинеарность обновлений около восьмого loop, но после обученной глубины hidden-state norm растёт быстрее baseline. Следовательно, input dropout работает как регуляризатор для фиксированной глубины, но не решает задачу устойчивого использования большего числа loops. Результат получен на одном seed и является пилотным.

Полные метрики находятся в [`results/summary.csv`](results/summary.csv).

## Запуск

Из корня репозитория:

Для повторного запуска кодовая база должна содержать поддержку параметров `loop_input_dropout` и `loop_input_dropout_start` в `LoopedQwen3Config` и recurrent forward pass.

```bash
python experiments/002_loop_input_dropout/run.py --variant dropout_p010_r8
python experiments/002_loop_input_dropout/run.py --variant dropout_p020_r8
```

Для продолжения прерванного обучения добавьте `--resume`. Только оценка существующего checkpoint запускается с `--eval-only`.

После получения двух JSON соберите итоговую таблицу:

```bash
python experiments/002_loop_input_dropout/summarize.py
```

Checkpoints сохраняются в `outputs/experiments/002/`, а JSON и CSV — в `experiments/002_loop_input_dropout/results/`. Директорию `outputs/` не следует коммитить в GitHub.
