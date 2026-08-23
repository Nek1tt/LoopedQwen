#!/usr/bin/env python3
"""Проверка целостности и ключевых чисел опубликованных результатов."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"Пустая таблица: {relative_path}")
    return rows


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Ожидалось {expected}, получено {actual}")


def row_for(rows: list[dict[str, str]], **keys: str | int) -> dict[str, str]:
    for row in rows:
        if all(row[name] == str(value) for name, value in keys.items()):
            return row
    raise AssertionError(f"Строка не найдена: {keys}")


def check_json_files() -> int:
    count = 0
    for path in sorted((ROOT / "experiments").glob("**/results/**/*.json")):
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
        count += 1
    if count < 20:
        raise AssertionError(f"Неожиданно мало JSON-файлов: {count}")
    return count


def check_early_experiments() -> None:
    expected_rows = {
        "experiments/001_skip_connections/results/summary.csv": 18,
        "experiments/002_loop_input_dropout/results/summary.csv": 18,
        "experiments/003_loop_state_noise/results/summary.csv": 27,
        "experiments/004_normalized_loop_updates/results/summary.csv": 27,
        "experiments/005_anytime_depth/results/summary.csv": 27,
        "experiments/006_hard_token_correction/results/summary.csv": 58,
    }
    for path, expected in expected_rows.items():
        rows = read_csv(path)
        if len(rows) != expected:
            raise AssertionError(f"{path}: ожидалось {expected} строк, получено {len(rows)}")

    exp1 = read_csv("experiments/001_skip_connections/results/summary.csv")
    close(float(row_for(exp1, variant="baseline_r8", eval_loops=8)["perplexity"]), 1125.2115240689943)

    exp4 = read_csv("experiments/004_normalized_loop_updates/results/summary.csv")
    close(float(row_for(exp4, variant="projected_fixed_a025_r16", eval_loops=16)["perplexity"]), 1431.8119217686634)
    close(float(row_for(exp4, variant="projected_learned_a025_r16", eval_loops=32)["perplexity"]), 1964.7065154836986)

    exp6 = read_csv("experiments/006_hard_token_correction/results/summary.csv")
    close(float(row_for(exp6, variant="uniform_intermediate_control", eval_loops=32)["perplexity"]), 1136.571291063948)
    close(float(row_for(exp6, variant="hard_token_g05", eval_loops=32)["perplexity"]), 1161.5165931475205)


def check_final_experiment() -> None:
    aggregate = read_csv(
        "experiments/007_alternating_operators/results/multiseed/summary_aggregate.csv"
    )
    if len(aggregate) != 58:
        raise AssertionError(f"Сводная кривая 007 содержит {len(aggregate)} строк вместо 58")
    for variant in ("fixed_am_control", "fixed_ma"):
        depths = {int(row["eval_loops"]) for row in aggregate if row["variant"] == variant}
        if depths != set(range(4, 33)):
            raise AssertionError(f"Неполный набор глубин для {variant}")

    close(
        float(row_for(aggregate, variant="fixed_am_control", eval_loops=16)["mean_perplexity"]),
        1112.719135010392,
    )
    close(
        float(row_for(aggregate, variant="fixed_ma", eval_loops=16)["mean_perplexity"]),
        947.0742136229703,
    )
    close(
        float(row_for(aggregate, variant="fixed_ma", eval_loops=32)["mean_perplexity"]),
        980.6495590364789,
    )

    paired = read_csv(
        "experiments/007_alternating_operators/results/multiseed/paired_comparison.csv"
    )
    if len(paired) != 87:
        raise AssertionError(f"Парное сравнение содержит {len(paired)} строк вместо 87")
    if any(float(row["relative_delta_percent"]) >= 0 for row in paired):
        raise AssertionError("MLP → Attention выиграл не во всех парных точках")
    for seed in (42, 43, 44):
        depths = {int(row["eval_loops"]) for row in paired if row["seed"] == str(seed)}
        if depths != set(range(4, 33)):
            raise AssertionError(f"Неполное парное сравнение для зерна {seed}")

    by_seed = read_csv(
        "experiments/007_alternating_operators/results/multiseed/summary_by_seed.csv"
    )
    seed43 = [row for row in by_seed if row["variant"] == "fixed_ma" and row["seed"] == "43"]
    best = min(seed43, key=lambda row: float(row["perplexity"]))
    if int(best["eval_loops"]) != 11:
        raise AssertionError("Лучший результат зерна 43 должен находиться на глубине 11")
    close(float(best["perplexity"]), 905.8229211869554)


def main() -> None:
    json_count = check_json_files()
    check_early_experiments()
    check_final_experiment()
    print(
        "Проверка пройдена: 7 экспериментов, "
        f"{json_count} JSON-файлов, полные плотные кривые и 87/87 парных побед."
    )


if __name__ == "__main__":
    main()
