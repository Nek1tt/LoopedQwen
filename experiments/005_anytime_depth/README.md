# Эксперимент 005: случайная глубина и промежуточные LM losses

## Гипотеза

Experiment 004 удержал hidden-state RMS и сохранил ненулевые обновления до
обученной глубины, но получил жёсткую специализацию на 16 loops. Experiment 005
проверяет, можно ли убрать привязку к одному горизонту, обучая один и тот же
projected recurrent block завершать вычисление на разных глубинах.

Гипотеза выведена только из результатов Experiment 004; внешняя литература при
выборе постановки не использовалась.

## Матрица эксперимента

Все варианты используют архитектуру `projected_learned` из Experiment 004,
seed 42, одинаковые данные и не более 10M обработанных train-токенов.

| Вариант | Train depth | Supervised depths одного microbatch |
|---|---|---|
| `fixed_r16_final` | всегда 16 | `16` |
| `random_r8_24_final` | равномерно случайная `R ∈ [8,24]` | `R` |
| `random_r8_24_intermediate` | равномерно случайная `R ∈ [8,24]` | `8`, midpoint и `R` |

Для intermediate-варианта конечный loss имеет вес 0.5. Оставшиеся 0.5 поровну
делятся между уникальными промежуточными точками. Если точки совпадают на малой
глубине, дубликаты удаляются и веса нормализуются.

$$
L = 0.5L_R + 0.25L_8 + 0.25L_{\lfloor(8+R)/2\rfloor}
$$

При совпадении depths формула автоматически сводится к двум или одному loss.
LM head и все параметры модели общие; новые матрицы не добавляются.

Случайная глубина вычисляется детерминированно из seed и глобального номера
microbatch. Поэтому resume не меняет последовательность глубин. При среднем
`R=16` random-варианты имеют примерно тот же основной recurrent compute, что и
fixed control; intermediate head добавляет только декодирование дополнительных
состояний.

## Validation и итоговая оценка

Checkpoint выбирается по среднему validation loss на depths 8, 12, 16, 20 и
24. Каждый depth видит одинаковые validation batches. Итоговый evaluation
проверяет 1, 2, 4, 8, 12, 16, 20, 24 и 32 loops и сохраняет:

- loss и perplexity;
- hidden-state norm;
- relative update;
- cosine similarity соседних состояний;
- loop-dependent gate.

Критерий поддержки гипотезы — более ровная кривая PPL внутри 8–24 без резкого
ухудшения после 16 и, отдельно, более безопасная экстраполяция к 32 loops.

## Результаты

Все три запуска завершили 305 optimizer steps и обработали 9,994,240
train-токенов. NaN и divergence не наблюдались. Максимальный записанный
gradient norm составил 1.805 у fixed control, 1.770 у random-final и 1.751 у
random-intermediate. В записанных training batches random-варианты покрыли весь
диапазон 8–24 со средней глубиной 15.78.

### Anytime checkpoint criterion

Основной checkpoint criterion одинаков для всех вариантов: средний validation
loss на depths 8, 12, 16, 20 и 24. Evaluation выбранных checkpoints даёт:

| Eval loops | Fixed R16 | Random R, final loss | Random R, intermediate losses |
|---:|---:|---:|---:|
| 8 | 1898.09 | 1393.38 | **1109.54** |
| 12 | 1899.65 | 1387.78 | **1107.61** |
| 16 | 1900.36 | 1390.59 | **1111.30** |
| 20 | 1900.72 | 1395.25 | **1116.51** |
| 24 | 1900.84 | 1401.47 | **1122.54** |
| 32 | 1900.94 | 1416.80 | **1136.32** |

Средний PPL на train range 8–24 равен 1899.93, 1393.69 и 1113.50
соответственно. Случайная глубина сама по себе снижает этот показатель на
26.64% относительно fixed control. Промежуточные losses дают ещё 20.10%
снижения относительно random-final.

У random-final разброс PPL внутри 8–24 составляет только 13.69, у
random-intermediate — 14.93. Резкого перелома после loop 16, наблюдавшегося в
Experiment 004, больше нет. Extrapolation до 32 loops также остаётся плавной.

### Почему fixed control выбрал ранний checkpoint

`fixed_r16_final/best_hf` соответствует шагу 100, а random checkpoints — шагу
304. Это не незавершённое обучение и не serialization bug. После шага 100
fixed-модель продолжала улучшать обученную глубину 16, но одновременно теряла
качество на ранних выходах:

| Step | Validation loss@8 | Validation loss@16 | Mean loss@8/12/16/20/24 |
|---:|---:|---:|---:|
| 100 | **7.5599** | 7.5610 | **7.5608** |
| 200 | 9.9662 | 7.3921 | 8.1206 |
| 304 | 9.9854 | **7.2759** | 8.0932 |

Единый anytime criterion корректно сохранил ранний checkpoint: более поздний
лучше только около жёстко обученной глубины. Сама траектория fixed control тем
самым повторяет специализацию на точном горизонте, обнаруженную в Experiment
004.

Для сравнения качества на фиксированном горизонте полезен полноценный
`projected_learned` checkpoint Experiment 004, выбранный по depth 16. Его
PPL@16 равен 1428.02, а PPL@32 — 1964.71. Относительно этого более сильного
reference:

| Вариант | PPL@16 | Изменение | PPL@32 | Изменение |
|---|---:|---:|---:|---:|
| Experiment 004 fixed-depth reference | 1428.02 | — | 1964.71 | — |
| Random R, final loss | 1390.59 | -2.62% | 1416.80 | -27.89% |
| Random R, intermediate losses | **1111.30** | **-22.18%** | **1136.32** | **-42.16%** |

Следовательно, выигрыш random-depth не является только следствием слабого
раннего checkpoint внутри Experiment 005.

### Динамика состояния

| Вариант | Norm@16 | Update@16 | Cosine@16 | Norm@32 | Update@32 | Cosine@32 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed R16 anytime checkpoint | 14.1788 | 0.00139 | 0.999999 | 14.1788 | 0.00005 | 1.000000 |
| Random R, final loss | 5.6740 | 0.01113 | 0.999913 | 5.6740 | **0.00579** | 0.999972 |
| Random R, intermediate losses | **4.1233** | **0.01233** | 0.999899 | **4.1233** | 0.00542 | 0.999975 |

RMS projection сохраняет norm постоянной. Random-depth модели поддерживают на
loop 16 обновление примерно на порядок больше fixed anytime checkpoint и не
коллапсируют к почти нулевому update на loop 32. Однако cosine всё ещё близок к
единице, а PPL достигает минимума около loop 12 и затем медленно ухудшается.

## Многосидовая проверка random-depth вариантов

Исходный запуск выше использует один seed 42. Чтобы отделить устойчивый эффект
промежуточных losses от удачного инициализационного seed, дополнительно
обучены `random_r8_24_final` и `random_r8_24_intermediate` на seeds 43 и 44.
Каждый из шести запусков завершил 305 optimizer steps и обработал 9,994,240
train-токенов. Для каждого выбранного checkpoint выполнен dense evaluation на
каждой целой глубине от 4 до 32, по 100 одинаковых validation batches на
глубину.

Таблица содержит среднее и sample standard deviation PPL по трём seeds:

| Вариант | PPL@8 | PPL@11 (минимум) | PPL@16 | PPL@24 | PPL@32 |
|---|---:|---:|---:|---:|---:|
| Random R, final loss | 1362.25 ± 56.13 | 1355.56 ± 59.36 | 1358.19 ± 61.55 | 1367.49 ± 61.82 | 1380.38 ± 61.47 |
| Random R, intermediate losses | **1111.16 ± 19.00** | **1108.46 ± 20.79** | **1112.72 ± 23.01** | **1123.97 ± 23.42** | **1137.03 ± 21.47** |

`intermediate` лучше `final` на каждом из 29 проверенных depths и на каждом
из трёх seeds. Разница составляет 18.43% на loop 8, 18.23% на loop 11, 18.07%
на loop 16 и 17.63% на loop 32. Следовательно, улучшение от промежуточного
supervision не объясняется одним удачным seed.

Dense evaluation уточнила положение optimum: для обоих random-depth вариантов
минимум среднего PPL находится на loop 11 (практически тот же уровень на loop
10–12), а не точно на loop 12. После этого кривая плавно ухудшается: от
минимума до loop 32 PPL увеличивается на 1.83% у `final` и на 2.58% у
`intermediate`. Поэтому random-depth надёжно устраняет резкий 16-step
перелом, но пока не делает каждый дополнительный loop полезным.

Многосидовый запуск не повторял `fixed_r16_final`; значит, он подтверждает
сравнение двух random-depth objectives, но сам по себе не расширяет
статистическую надёжность сравнения random-depth с fixed-depth control.
Исходные данные находятся в [`results/multiseed/`](results/multiseed/): шесть
dense evaluation JSON, per-seed таблица, aggregate mean/std и paired comparison.

## Вывод

Гипотеза подтверждена частично. Случайная глубина устраняет жёсткую 16-step
программу и создаёт модель, которую можно остановить в широком диапазоне 8–24
loops без резкой потери качества. Промежуточные LM losses дают большой
дополнительный выигрыш при том же количестве train-токенов.

При этом вычисление ещё не является строго монотонным anytime computation:
после примерно 11 loops дополнительные шаги не улучшают PPL, а медленно его
ухудшают. Многосидовая проверка на seeds 42–44 подтверждает устойчивый выигрыш
intermediate losses и широкую область безопасной остановки для random-depth
моделей. Она не заменяет многосидовое сравнение с fixed-depth control, которое
остается отдельной открытой проверкой.

## Прогресс и логи

Runner поддерживает `--in-process` и запускает training/evaluation в текущем
Python kernel через `runpy`. Поэтому `tqdm.auto` может обновляться прямо в
активной notebook-ячейке. Training bar показывает:

- optimizer step и ETA;
- общий loss, learning rate, gradient norm и tokens/s;
- среднюю случайную глубину `Rμ` текущего optimizer step;
- supervised heads последнего microbatch;
- gate в начале и на последней выбранной глубине;
- последний validation loss.

Validation и evaluation используют weighted progress в единицах `loop-batch`,
чтобы ETA учитывал разную стоимость depths. Полные записи, включая все 32
случайные глубины optimizer step и component losses, сохраняются в
`metrics.jsonl`.

## Запуск

Из корня репозитория:

```bash
python experiments/005_anytime_depth/run.py \
  --variant fixed_r16_final --resume
python experiments/005_anytime_depth/run.py \
  --variant random_r8_24_final --resume
python experiments/005_anytime_depth/run.py \
  --variant random_r8_24_intermediate --resume
python experiments/005_anytime_depth/summarize.py
```

Машиночитаемые результаты находятся в [`results/`](results/): три исходных
evaluation JSON и воспроизводимая таблица `summary.csv`. Notebook намеренно не
включён в Git-репозиторий.
