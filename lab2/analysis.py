"""ЛР2. Аналіз порога рішення зі збережених прогнозів (пункти 2-4 завдання).

Запуск (без навчання та без CSV):
    uv run python lab2_analysis.py --saved results --out analysis

Порядок виконання зафіксований у коді:
    1) validation: криві, ROC-AUC, повна таблиця кандидатів, поріг t*;
    2) перевірка пошуку на наборах A/B/C (скрипт зупиняється, якщо хоч одна не пройдена);
    3) лише після цього - одноразова оцінка test із зафіксованим t*.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions, write_json

COST_FN = 10  # ціна пропущеного шахрайства
COST_FP = 1  # ціна хибної тривоги
COLS = ["threshold", "TP", "FP", "FN", "TN", "C"]


# --------------------------------------------------------------------------- #
# Базові величини: матриця плутанини та вартість за правилом s >= t
# --------------------------------------------------------------------------- #
def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """TP, FP, FN, TN, C для правила «шахрайство, якщо s >= t» (t може бути +inf)."""
    y = np.asarray(y)
    pred = np.asarray(s) >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": COST_FN * fn + COST_FP * fp}


def precision_recall(c: dict[str, int]) -> tuple[float | None, float | None]:
    """None, якщо знаменник нульовий (метрика не визначена)."""
    p = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] > 0 else None
    r = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] > 0 else None
    return p, r


# --------------------------------------------------------------------------- #
# Швидкий пошук: сортування + кумулятивні суми, O(n log n)
# --------------------------------------------------------------------------- #
def candidates_fast(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Таблиця всіх кандидатів: унікальні оцінки та +inf. Без циклу по порогах."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    order = np.argsort(-s, kind="stable")  # за спаданням оцінки
    s_sorted, y_sorted = s[order], y[order]
    cum_tp = np.cumsum(y_sorted)  # скільки позитивів серед перших k+1
    cum_fp = np.cumsum(1 - y_sorted)

    # Рівні оцінки - одна група: беремо лише останній індекс кожної групи,
    # тоді в позитивний прогноз потрапляють усі з s >= t одночасно.
    is_last = np.ones(n, dtype=bool)
    is_last[:-1] = s_sorted[:-1] != s_sorted[1:]
    thr = s_sorted[is_last]
    tp = cum_tp[is_last]
    fp = cum_fp[is_last]

    thr = np.concatenate(([np.inf], thr))  # +inf: «усі негативні»
    tp = np.concatenate(([0], tp))
    fp = np.concatenate(([0], fp))
    fn = n_pos - tp
    tn = n_neg - fp
    cost = COST_FN * fn + COST_FP * fp
    return pd.DataFrame(
        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost}, columns=COLS
    )


def pick_threshold(table: pd.DataFrame) -> pd.Series:
    """Мінімум C; за рівної вартості - найбільший поріг."""
    best = table[table["C"] == table["C"].min()]
    return best.loc[best["threshold"].idxmax()]


# --------------------------------------------------------------------------- #
# Незалежний повільний перебір (лише для перевірки на малих наборах)
# --------------------------------------------------------------------------- #
def brute_force(y: list[int], s: list[float]) -> tuple[pd.DataFrame, float]:
    """Для кожного кандидата застосовуємо правило s_i >= t напряму."""
    cands = sorted({float(v) for v in s}) + [math.inf]
    rows = []
    for t in cands:
        pred = [1 if si >= t else 0 for si in s]
        tp = sum(1 for p, yi in zip(pred, y) if p == 1 and yi == 1)
        fp = sum(1 for p, yi in zip(pred, y) if p == 1 and yi == 0)
        fn = sum(1 for p, yi in zip(pred, y) if p == 0 and yi == 1)
        tn = sum(1 for p, yi in zip(pred, y) if p == 0 and yi == 0)
        rows.append((t, tp, fp, fn, tn, 10 * fn + fp))
    table = pd.DataFrame(rows, columns=COLS)
    best_c = min(r[5] for r in rows)
    best_t = max(r[0] for r in rows if r[5] == best_c)
    return table, best_t


CHECK_SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}
# Очікування, пораховані вручну (незалежно від обох реалізацій): (t*, C(t*))
EXPECTED = {"A": (0.4, 2), "B": (math.inf, 0), "C": (0.8, 10)}


def run_checks(out: Path) -> dict[str, dict[str, object]]:
    """Порівнює швидкий алгоритм із прямим перебором; кидає AssertionError при розбіжності."""
    report: dict[str, dict[str, object]] = {}
    for name, (s, y) in CHECK_SETS.items():
        fast = candidates_fast(np.array(y), np.array(s)).sort_values("threshold").reset_index(drop=True)
        slow, t_slow = brute_force(y, s)
        slow = slow.sort_values("threshold").reset_index(drop=True)
        same_table = fast.equals(slow) or (
            np.array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy())
            and (fast[COLS[1:]].to_numpy() == slow[COLS[1:]].to_numpy()).all()
        )
        t_fast = float(pick_threshold(candidates_fast(np.array(y), np.array(s)))["threshold"])
        exp_t, exp_c = EXPECTED[name]
        c_star = int(slow.loc[slow["threshold"] == t_slow, "C"].iloc[0])
        ok = same_table and t_fast == t_slow == exp_t and c_star == exp_c
        fast.to_csv(out / f"check_{name}_fast.csv", index=False)
        slow.to_csv(out / f"check_{name}_bruteforce.csv", index=False)
        report[name] = {
            "tables_identical": bool(same_table),
            "t_star_fast": t_fast if math.isfinite(t_fast) else "inf",
            "t_star_bruteforce": t_slow if math.isfinite(t_slow) else "inf",
            "C_star": c_star,
            "passed": bool(ok),
        }
        print(f"  набір {name}: таблиці збігаються={same_table}, t*(fast)={t_fast}, "
              f"t*(brute)={t_slow}, C*={c_star} -> {'OK' if ok else 'ПОМИЛКА'}")
        assert ok, f"Перевірка набору {name} не пройдена - тест не оцінюється."
    return report


# --------------------------------------------------------------------------- #
# Графіки
# --------------------------------------------------------------------------- #
def plot_curves(y: np.ndarray, s: np.ndarray, marks: dict[str, float], out: Path) -> dict[str, float]:
    auc = float(roc_auc_score(y, s))
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    styles = {"0.5": ("tab:orange", "s"), "t*": ("tab:red", "o")}

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(rec, prec, color="tab:blue", lw=1.8, label="PR-крива (validation)")
    ax.axhline(y.mean(), color="gray", ls="--", lw=1, label=f"частка позитивів = {y.mean():.4f}")
    for label, t in marks.items():
        p, r = precision_recall(confusion(y, s, t))
        if p is not None and r is not None:
            color, m = styles[label]
            ax.scatter([r], [p], color=color, marker=m, s=60, zorder=5,
                       label=f"t = {label}: P={p:.3f}, R={r:.3f}")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)",
           title="Крива точність-повнота, validation", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="tab:blue", lw=1.8, label=f"ROC-крива, AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="випадкове вгадування")
    for label, t in marks.items():
        c = confusion(y, s, t)
        color, m = styles[label]
        ax.scatter([c["FP"] / (c["FP"] + c["TN"])], [c["TP"] / (c["TP"] + c["FN"])],
                   color=color, marker=m, s=60, zorder=5, label=f"t = {label}")
    ax.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR = recall",
           title="ROC-крива, validation", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png")
    plt.close(fig)

    ap = float(-np.sum(np.diff(rec) * prec[:-1]))  # середня точність (лише для звіту)
    return {"roc_auc": auc, "average_precision": ap}


def plot_cost(table: pd.DataFrame, t_star: float, out: Path) -> None:
    """Допоміжний графік C(t): видно, де лежить мінімум і плато рівної вартості."""
    d = table[np.isfinite(table["threshold"])].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.step(d["threshold"], d["C"], where="post", color="tab:blue", lw=1.5)
    ax.axvline(t_star, color="tab:red", ls="--", label=f"t* = {t_star:.6f}")
    ax.axvline(0.5, color="tab:orange", ls=":", label="t = 0.5")
    ax.set_yscale("log")
    ax.set(xlabel="Поріг t", ylabel="C(t) = 10·FN + FP (лог. шкала)",
           title="Вартість помилок залежно від порога, validation")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cost_vs_threshold_validation.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Головна процедура
# --------------------------------------------------------------------------- #
def fmt(x: float | None, nd: int = 4) -> str:
    return "не визначено" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", type=Path, required=True, help="Тека з predictions.npz")
    ap.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    data = load_predictions(args.saved)
    yv, sv = data["validation_y"], data["validation_scores"].astype(np.float64)
    yt, st = data["test_y"], data["test_scores"].astype(np.float64)
    ids_t, amount_t = data["test_row_ids"], data["test_amount"]

    # ---- 2. Validation ---------------------------------------------------- #
    print("== Validation: таблиця кандидатів ==")
    table = candidates_fast(yv, sv)
    # самоперевірка: кожен рядок таблиці збігається з прямим підрахунком
    for row in table.sample(n=min(200, len(table)), random_state=0).itertuples():
        d = confusion(yv, sv, row.threshold)
        assert (d["TP"], d["FP"], d["FN"], d["TN"], d["C"]) == (row.TP, row.FP, row.FN, row.TN, row.C)
    table.sort_values("threshold", ascending=False).to_csv(args.out / "validation_candidates.csv", index=False)
    best = pick_threshold(table)
    t_star = float(best["threshold"])

    rules = {
        "Стандартний поріг": 0.5,
        "Мінімум вартості": t_star,
        "Усі негативні": math.inf,
    }
    summary_rows = []
    for name, t in rules.items():
        c = confusion(yv, sv, t)
        summary_rows.append({"rule": name, "threshold": t, **c})
    summary_val = pd.DataFrame(summary_rows)
    summary_val.to_csv(args.out / "validation_summary.csv", index=False)
    print(summary_val.to_string(index=False))
    print(f"t* = {t_star!r}, кандидатів: {len(table)}, "
          f"рівних мінімуму: {int((table['C'] == table['C'].min()).sum())}")

    marks = {"0.5": 0.5, "t*": t_star}
    metrics = plot_curves(yv, sv, marks, args.out)
    plot_cost(table, t_star, args.out)
    print(f"ROC-AUC (validation) = {metrics['roc_auc']:.6f}, AP = {metrics['average_precision']:.6f}")

    # ---- Фіксація порога ---------------------------------------------------- #
    write_json(args.out / "threshold.json", {
        "t_star": t_star, "validation_C": int(best["C"]),
        "rule": "predict fraud if score >= t_star; min C=10*FN+FP; ties -> largest threshold",
    })

    # ---- 3. Перевірка пошуку ---------------------------------------------- #
    print("== Перевірка на наборах A/B/C ==")
    checks = run_checks(args.out)

    # ---- 4. Test (один раз, поріг уже зафіксований) ------------------------ #
    print("== Test ==")
    test_rows = []
    for name, t in rules.items():
        t_use = t_star if name == "Мінімум вартості" else t
        c = confusion(yt, st, t_use)
        p, r = precision_recall(c)
        test_rows.append({"rule": name, "threshold": t_use, **c, "precision": p, "recall": r})
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(args.out / "test_summary.csv", index=False)
    print(test_df.to_string(index=False))

    pred = st >= t_star
    order = np.argsort(ids_t, kind="stable")  # за номером рядка CSV
    fp_idx = [i for i in order if pred[i] and yt[i] == 0]
    fn_idx = [i for i in order if (not pred[i]) and yt[i] == 1]
    errs = []
    for kind, idx in (("FP", fp_idx[:5]), ("FN", fn_idx[:5])):
        for i in idx:
            errs.append({"csv_row": int(ids_t[i]), "type": kind, "label": int(yt[i]),
                         "score": float(st[i]), "amount": float(amount_t[i]),
                         "score_minus_t_star": float(st[i] - t_star)})
    err_df = pd.DataFrame(errs)
    err_df.to_csv(args.out / "test_error_examples.csv", index=False)
    print(err_df.to_string(index=False))

    # Агрегована статистика за ВСІМА помилками test (лише описова)
    def stats(idx: list[int]) -> dict[str, float | int | None]:
        if not idx:
            return {"n": 0}
        a, d = amount_t[idx], st[idx] - t_star
        return {"n": len(idx), "amount_median": float(np.median(a)), "amount_mean": float(a.mean()),
                "amount_min": float(a.min()), "amount_max": float(a.max()),
                "score_dist_median": float(np.median(d)), "score_dist_min": float(d.min()),
                "score_dist_max": float(d.max())}

    all_fp = [i for i in range(len(yt)) if pred[i] and yt[i] == 0]
    all_fn = [i for i in range(len(yt)) if (not pred[i]) and yt[i] == 1]
    err_stats = {"FP_all": stats(all_fp), "FN_all": stats(all_fn),
                 "Amount_median_all_test": float(np.median(amount_t))}
    print(err_stats)

    write_json(args.out / "summary.json", {
        "t_star": t_star, "validation": summary_val.assign(
            threshold=summary_val["threshold"].map(lambda v: v if math.isfinite(v) else "inf")
        ).to_dict("records"),
        "metrics_validation": metrics, "checks": checks,
        "test": test_df.assign(threshold=test_df["threshold"].map(lambda v: v if math.isfinite(v) else "inf"))
        .astype(object).where(test_df.notna(), None).to_dict("records"),
        "error_stats": err_stats,
    })
    print(f"Результати збережено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
