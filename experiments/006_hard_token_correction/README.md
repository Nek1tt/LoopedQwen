# Эксперимент 006: hard-token recurrent correction

## Мотивация

Experiment 005 сделал random-depth модель устойчивой к остановке, однако dense
evaluation на seeds 42–44 показал одинаковую форму кривой: лучший средний PPL
достигается около loop 11, после чего дополнительные loops постепенно ухудшают
предсказание. При этом обычный intermediate objective одинаково сильно
супервизирует уже решённые и всё ещё трудные токены.

Experiment 006 проверяет гипотезу, что поздняя recurrent computation станет
полезнее, если обучать её в первую очередь исправлять ошибки предыдущей
supervised depth, сохраняя обычный LM loss как стабилизирующую часть objective.

## Постановка

Архитектура, depth schedule и token budget полностью совпадают с
`random_r8_24_intermediate` из Experiment 005:

- равномерная случайная train depth `R ∈ [8, 24]`;
- LM losses на loop 8, midpoint и `R`;
- веса depths 0.25, 0.25 и 0.5;
- 305 optimizer steps и 9,994,240 train-токенов;
- seed 42;
- checkpoint criterion — средний validation loss на 8/12/16/20/24 loops.

Меняется только token-wise weighting внутри второй и последующих supervised
depths. Для токена `t` вес строится из detached cross-entropy предыдущей
supervised точки:

$$
w_{r,t} = \operatorname{clip}\left(
\left(
\frac{\operatorname{stopgrad}(\mathrm{CE}_{r^{-},t})}
{\operatorname{mean}_{j}\operatorname{stopgrad}(\mathrm{CE}_{r^{-},j})}
\right)^{0.5},
0.25,
4.0
\right).
$$

Нормализация выполняется отдельно для каждой последовательности. Ignore-index
токены не участвуют ни в среднем, ни в corrective loss.

$$
L_r^{\mathrm{corr}} =
\frac{\sum_t w_{r,t}\,\mathrm{CE}_{r,t}}
{\sum_t w_{r,t}},
\qquad
L_r = 0.5L_r^{\mathrm{uniform}} + 0.5L_r^{\mathrm{corr}}.
$$

Первая supervised depth использует обычный uniform loss. `stopgrad` не позволяет
модели искусственно повышать предыдущий loss ради изменения будущих весов.
Clipping и 50% uniform component защищают objective от единичных выбросов.

## Контроль и критерий успеха

Контроль повторно не обучается: используется многосидовый
`random_r8_24_intermediate` Experiment 005. Основное парное сравнение на seed 42:

| Вариант | Token weighting | Новый train run |
|---|---|---:|
| Experiment 005 control | uniform | нет |
| `hard_token_g05` | previous-loss, `γ=0.5` | да |

Основные критерии:

1. `PPL@24 ≤ PPL@16` и `PPL@32 ≤ PPL@24`;
2. PPL в диапазоне 8–16 не хуже seed-42 control;
3. лучший depth сдвигается правее 11 или поздний regret заметно уменьшается;
4. обучение завершается без NaN/divergence в том же token budget.

Evaluation выполняется на каждой целой глубине 4–32 по 100 одинаковых
validation batches. JSON содержит loss, perplexity и hidden-state diagnostics.

## Результаты

Запуск завершил все 305 optimizer steps и обработал 9,994,240 train-токенов
без NaN или divergence. Лучшим оказался последний checkpoint, step 304, со
средним validation loss 7.04983 на глубинах 8/12/16/20/24. Dense evaluation
использует 100 одинаковых validation batches на каждой из 29 глубин.

| Loops | Uniform control PPL | Hard-token PPL | Изменение |
|---:|---:|---:|---:|
| 4 | 1209.48 | **1185.18** | **−2.01%** |
| 8 | **1109.66** | 1131.28 | +1.95% |
| 10 | **1107.32** | 1129.88 | +2.04% |
| 16 | **1111.38** | 1136.76 | +2.28% |
| 24 | **1122.60** | 1148.91 | +2.34% |
| 32 | **1136.57** | 1161.52 | +2.19% |

| Метрика | Uniform control | Hard-token |
|---|---:|---:|
| Best depth | 10 | 10 |
| Best PPL | **1107.32** | 1129.88 |
| PPL@32 regret | **2.64%** | 2.80% |
| Adjacent PPL increases, R=4…32 | 22 | 22 |

Hard-token weighting улучшил только экстраполяционный early exit `R=4`, который
лежит ниже минимальной train depth. На каждой глубине `R=5…32` контроль лучше;
в основном train/eval диапазоне `R=8…24` hard-token objective проигрывает на
1.95–2.35%. Оптимум не сдвинулся вправо, а поздний regret немного вырос.

Следовательно, гипотеза в проверенной форме **не подтверждена**. Высокая
cross-entropy предыдущего состояния не является достаточным признаком того,
что токен можно исправить дополнительными recurrent steps. Она смешивает как
минимум три случая: действительно исправимые токены, неоднозначный контекст и
редкие/шумные targets. Повышенный градиент последних двух групп уводит обучение
от обычного LM objective, не создавая положительного marginal gain поздних
loops.

Улучшение при `R=4` согласуется с более широким, но слабым эффектом
регуляризации представлений до train horizon. Поскольку это единственная из 29
глубин с улучшением и она не является рабочей train depth, мы не считаем её
подтверждением основной гипотезы.

Сравнение проведено на одном парном seed 42. Этого достаточно для отклонения
данной конфигурации как кандидата на лучший метод, но недостаточно для общего
утверждения, что любое difficulty-aware weighting обязательно вредно.

## Артефакты

- `results/hard_token_g05_eval.json` — исходный dense evaluation и hidden-state
  diagnostics;
- `results/summary.csv` — обе кривые по 29 глубинам;
- `results/metrics.csv` — компактные итоговые показатели;
- `results/training_metrics.jsonl` — полный train/validation log;
- `results/run_config.json` и `results/training_summary.json` — фактическая
  конфигурация и выбранный checkpoint.

## Логирование и Colab

Runner поддерживает `--in-process`, поэтому training, validation и evaluation
работают внутри активного Jupyter kernel. `tqdm` показывает ETA, loss, LR,
gradient norm, tokens/s, sampled depth `Rμ`, supervised heads и режим `tok-w`.

В `metrics.jsonl` дополнительно записываются:

- смешанные component losses;
- uniform component losses;
- corrective component losses;
- средние hard-token weights для каждой supervised depth;
- `gamma` и доля uniform loss.

Resume выполняется из `outputs/experiments/006/hard_token_g05/last_state.pt`.
Повторный запуск notebook безопасно продолжает незавершённое обучение.

## Запуск

Из корня репозитория:

```bash
python experiments/006_hard_token_correction/run.py \
  --variant hard_token_g05 \
  --resume \
  --in-process
```

После завершения:

```bash
python experiments/006_hard_token_correction/collect_results.py
```

Будет создан `experiment_006_results.zip` без весов checkpoint.

## Позиционирование и следующий вывод

Механизм не является learned halting, monotonic penalty или depth
conditioning. Он не меняет inference graph и использует target-aware веса
только во время обучения. Формулировка результата должна оставаться аккуратной:
это предлагаемое применение detached previous-depth token errors к
intermediate supervision, а не доказанное утверждение об абсолютном отсутствии
сходных идей во всей литературе.

Отрицательный результат также уточняет следующий архитектурный эксперимент:
поздние loops следует делать функционально отличающимися от ранних, а не только
перераспределять тот же LM loss между токенами. Поэтому hard-token weighting не
переносится в следующий эксперимент и не смешивается с его control.
