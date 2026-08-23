# Experiment 007: recurrent operator order

## Мотивация

Experiments 004–006 последовательно показали:

1. контроль hidden-state norm не устраняет специализацию на одной глубине;
2. random depth и intermediate LM losses создают устойчивые early exits;
3. простое усиление high-CE токенов не превращает поздние loops в полезные
   corrective steps.

Во всех этих вариантах один recurrent pass остаётся одним и тем же автономным
оператором: каждый физический Qwen3 block всегда выполняет attention, затем
MLP. Experiment 007 проверяет архитектурную альтернативу: может ли двухфазный
порядок подоператоров создать разные стадии вычисления без новых весов, gates,
step embeddings или дополнительного inference compute.

## Гипотеза

Обозначим стандартные pre-norm residual-подоператоры:

$$
A(h)=h+\mathrm{Attention}(\mathrm{RMSNorm}_{A}(h)),
$$

$$
M(h)=h+\mathrm{MLP}(\mathrm{RMSNorm}_{M}(h)).
$$

Обычный loop повторяет:

$$
h_{t+1}=M(A(h_t)).
$$

Предлагаемый alternating schedule использует:

$$
h_{t+1}=
\begin{cases}
M(A(h_t)), & t\text{ нечётный},\\
A(M(h_t)), & t\text{ чётный}.
\end{cases}
$$

Attention и MLP используют те же параметры на каждом loop. Поскольку эти
операторы в общем случае не коммутируют,

$$
A(M(h)) \neq M(A(h)),
$$

чередование создаёт двухфазную time-varying dynamics при неизменных parameter
count и количестве вызовов attention/MLP.

## Экспериментальная матрица

Все варианты используют настройки лучшей модели Experiment 005:

- uniform random train depth R от 8 до 24;
- LM losses на loop 8, midpoint и sampled terminal depth;
- веса 0.25/0.25/0.5;
- projected learned loop updates;
- seed 42;
- 305 optimizer steps и 9,994,240 processed tokens;
- одинаковый validation criterion на R=8/12/16/20/24.

| Вариант | Нечётные loops | Чётные loops | Новый train run |
|---|---|---|---:|
| fixed_am_control | Attention → MLP | Attention → MLP | нет, Experiment 005 |
| fixed_ma | MLP → Attention | MLP → Attention | да |
| alternating_am_ma | Attention → MLP | MLP → Attention | да |

fixed_ma необходим: без него нельзя отделить эффект чередования от простого
эффекта обратного порядка.

## Causal interventions

После обучения alternating_am_ma один и тот же checkpoint оценивается четырьмя
способами:

| Evaluation | Schedule |
|---|---|
| native | AM/MA |
| force_am | AM/AM |
| force_ma | MA/MA |
| reverse_parity | MA/AM |

Override не меняет config или веса. Если native schedule лучше принудительных
режимов, это является причинным свидетельством того, что модель использует
выученную фазовую структуру, а не получила только случайный regularization
effect во время обучения.

## Mechanistic diagnostics

Dense evaluation выполняется на каждой глубине R=4…32 по 100 одинаковых
validation batches. Помимо PPL сохраняются:

- relative full-state update;
- cosine между последовательными состояниями;
- cosine между последовательными update vectors;
- directional diversity токенов;
- относительные attention- и MLP-updates;
- operator order каждого loop.

На пяти batches каждой глубины дополнительно вычисляется order defect:

$$
D(h)=
\frac{
\left\|M(A(h))-A(M(h))\right\|
}{
\left\|h\right\|+\varepsilon
}.
$$

Этот diagnostic требует альтернативного block evaluation и поэтому намеренно
ограничен пятью batches. Он не участвует в prediction и training.

## Критерии результата

Основные показатели:

1. mean PPL на R=8…24;
2. PPL@16, PPL@24 и PPL@32;
3. best depth и regret@32;
4. количество adjacent PPL increases;
5. native-vs-intervention effect.

Сильным подтверждением будет одновременное выполнение:

- alternating лучше обоих fixed-order вариантов;
- PPL@32 regret заметно ниже 2.64% seed-42 control;
- принудительная замена schedule во время evaluation ухудшает PPL;
- operator defect остаётся ненулевым и связан с различающимися update
  directions.

Если fixed_ma выигрывает, а alternating нет, результат относится к порядку
подоператоров, но не подтверждает двухфазную гипотезу. Если intervention почти
ничего не меняет, alternating schedule нельзя считать причинно используемым.

## Результаты первого запуска

Оба новых варианта завершили 305 optimizer steps и обработали по 9,994,240
train-токенов. Parameter count одинаков: 5,670,402. Dense evaluation использует
100 одинаковых validation batches на каждой целой глубине от 4 до 32.

| Вариант | Best PPL | Mean PPL 8–24 | PPL@16 | PPL@32 |
|---|---:|---:|---:|---:|
| fixed AM control | 1107.32 @ R=10 | 1112.86 | 1111.38 | 1136.57 |
| fixed MA | **988.45 @ R=12** | **994.02** | **991.44** | **1023.59** |
| alternating AM/MA | 1112.37 @ R=12 | 1116.00 | 1114.12 | 1137.37 |

Исходная гипотеза о преимуществе чередования не подтвердилась: alternating
вариант практически повторяет AM control. Вместо этого обнаружен более сильный
эффект постоянного обратного порядка. На seed 42 fixed MA снижает mean PPL
8–24 на 10.68% и лучше control на каждой из 29 evaluation depths.

## Multiseed-подтверждение fixed MA

Fixed MA был дополнительно обучен на seeds 43 и 44. Для парного сравнения
используются уже опубликованные fixed-AM checkpoints Experiment 005 с теми же
seeds, token budget, depth schedule и intermediate objective.

| Seed | AM mean PPL 8–24 | MA mean PPL 8–24 | Изменение | MA лучше на depths |
|---:|---:|---:|---:|---:|
| 42 | 1112.86 | **994.02** | **−10.68%** | 29/29 |
| 43 | 1137.13 | **912.40** | **−19.76%** | 29/29 |
| 44 | 1092.62 | **942.87** | **−13.71%** | 29/29 |

Итоговая средняя кривая:

| Вариант, 3 seeds | Best PPL | Mean PPL 8–24 | PPL@8 | PPL@16 | PPL@24 | PPL@32 |
|---|---:|---:|---:|---:|---:|---:|
| fixed AM | 1108.46 @ R=11 | 1114.20 | 1111.16 | 1112.72 | 1123.97 | 1137.03 |
| fixed MA | **943.95 @ R=12** | **949.77** | **949.50** | **947.07** | **961.01** | **980.65** |
| relative change | **−14.84%** | **−14.76%** | **−14.55%** | **−14.89%** | **−14.50%** | **−13.75%** |

Fixed MA выигрывает во всех 87 парных точках: 3 seeds × 29 depths. Даже худший
MA-run по mean PPL 8–24 лучше лучшего AM-run. Вариативность fixed MA между
seeds выше, однако интервалы наблюдаемых результатов не пересекаются.

Лучшим индивидуальным checkpoint является seed 43: best PPL 905.82 при R=11,
mean PPL 8–24 912.40, PPL@16 909.87 и PPL@32 944.76. Он также имеет лучший
train-time validation criterion, поэтому именно этот checkpoint выбран для
публикации.

## Причинный и механистический анализ

Alternating checkpoint чувствителен к фазе. Force-AM постепенно ухудшает PPL
от +0.90% при R=8 до +9.92% при R=32; force-MA и reverse parity ухудшают PPL
примерно вдвое. Значит, нечётные и чётные passes выучили разные роли, но их
чередование не улучшило LM objective.

У alternating dynamics cosine между соседними recurrent update vectors
пересекает ноль около loop 12 и достигает примерно −0.75 при R=32. Поздние
AM- и MA-обновления начинают компенсировать друг друга. У fixed MA тот же
cosine после loop 3 выше 0.98 и приближается к 0.999, одновременно с плавным
уменьшением full-state update. Это согласуется с устойчивым однонаправленным
refinement вместо двухфазной осцилляции.

Order defect fixed MA остаётся около 0.20 на поздних loops, поэтому эффект
порядка нельзя свести к двум почти коммутирующим представлениям одного блока.
Дополнительное вмешательство в seed-42 fixed-MA checkpoint подтверждает
специализацию:

| Depth | Native MA PPL | MA checkpoint evaluated as AM | Ухудшение |
|---:|---:|---:|---:|
| 4 | 1052.57 | 1543.66 | +46.66% |
| 8 | 993.50 | 1536.82 | +54.69% |
| 16 | 991.44 | 1576.13 | +58.97% |
| 24 | 1004.91 | 1634.73 | +62.67% |
| 32 | 1023.59 | 1707.78 | +66.84% |

Обратное вмешательство AM checkpoint → MA выполнить не удалось: исходные
weights seed-42 control уже отсутствовали в активной runtime. Это ограничивает
симметрию causal matrix, но не затрагивает три полных парных train/eval
сравнения.

## Интерпретация и ограничения

Результат подтверждает не исходную alternating-гипотезу, а более простую:
порядок подоператоров внутри weight-tied recurrence является существенным
архитектурным выбором. В проверенной постановке MLP до attention даёт
существенно более качественный recurrent operator без новых параметров и без
дополнительных sublayer calls.

При этом fixed MA улучшает абсолютный уровень всей anytime-кривой, но не делает
test-time scaling монотонным. Средний минимум достигается при R=12, а PPL@32
выше него на 3.89%. Поэтому вывод ограничивается устойчивым улучшением качества
при любом tested stopping depth, а не утверждением, что каждый дополнительный
loop полезен.

## Реализация

Для attention_mlp вызывается официальный Qwen3DecoderLayer.forward.
mlp_attention переиспользует те же:

- input_layernorm;
- post_attention_layernorm;
- self_attn;
- mlp;
- residual additions.

Меняется только порядок их применения. Unit test проверяет численную
эквивалентность attention_mlp официальному Qwen3 forward, равенство parameter
count, правильную последовательность вызовов, HF round-trip и causal override.

## Запуск

Из корня репозитория:

    python experiments/007_alternating_operators/run.py \
      --variant fixed_ma \
      --resume \
      --in-process

    python experiments/007_alternating_operators/run.py \
      --variant alternating_am_ma \
      --resume \
      --in-process

После обоих запусков:

    python experiments/007_alternating_operators/summarize.py
    python experiments/007_alternating_operators/collect_results.py

Будет создан experiment_007_results.zip без checkpoint weights и notebook.

Multiseed fixed-MA runs воспроизводятся напрямую из сохранённых конфигураций:

    python scripts/train.py \
      --config experiments/007_alternating_operators/configs/fixed_ma_seed43.yaml

    python scripts/train.py \
      --config experiments/007_alternating_operators/configs/fixed_ma_seed44.yaml

Сводные multiseed-таблицы полностью перестраиваются из committed JSON:

    python experiments/007_alternating_operators/summarize_multiseed.py

## Артефакты

- `results/*.json` — исходные fixed-order, alternating и intervention curves;
- `results/summary.csv` и `results/metrics.csv` — первичная матрица seed 42;
- `results/operator_diagnostics.csv` — покомпонентная trajectory-диагностика;
- `results/multiseed/fixed_ma_seed*_dense_eval.json` — три fixed-MA curves;
- `results/multiseed/summary_aggregate.csv` — mean/std по depths;
- `results/multiseed/paired_comparison.csv` — все 87 парных сравнений;
- `results/multiseed/cross_order_matrix.csv` — доступные causal interventions.
- `results/**/training/` — полные train/validation logs, фактические run configs
  и summaries выбранных checkpoints.

Weights намеренно не хранятся в Git. Лучший checkpoint seed 43 и tokenizer
публикуются отдельным Hugging Face model repository.

## Позиционирование

Работа [Feed-Forward Steering in Transformer Residual
Dynamics](https://arxiv.org/abs/2608.02071) анализирует attention как
aggregation, FFN как directional steering и измеряет чувствительность к
последовательному порядку подоператоров. Experiment 007 использует близкую
механистическую мотивацию, но проверяет другую конструкцию: детерминированное
чередование AM/MA внутри weight-tied language-model recurrence.

Поиск публичных looped-LM работ и submission не обнаружил такую ablation, но это
не является утверждением об абсолютном отсутствии аналогов во всей литературе.
