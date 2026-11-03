import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

COLS = ["threshold", "TP", "FP", "FN", "TN", "C"]
CHECKS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}


def confusion(y, s, t):
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn, 10 * fn + fp


def search(y, s):
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last = np.append(ss[:-1] != ss[1:], True)
    tp = np.concatenate(([0], np.cumsum(ys)[last]))
    fp = np.concatenate(([0], np.cumsum(1 - ys)[last]))
    fn, tn = y.sum() - tp, (1 - y).sum() - fp
    return pd.DataFrame(
        {"threshold": np.concatenate(([np.inf], ss[last])), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp}
    )


def best(table):
    return table.loc[table["C"] == table["C"].min(), "threshold"].max()


def brute(y, s):
    rows = []
    for t in sorted(set(s)) + [math.inf]:
        pred = [v >= t for v in s]
        tp = sum(p and l == 1 for p, l in zip(pred, y))
        fp = sum(p and l == 0 for p, l in zip(pred, y))
        fn = sum((not p) and l == 1 for p, l in zip(pred, y))
        tn = sum((not p) and l == 0 for p, l in zip(pred, y))
        rows.append((t, tp, fp, fn, tn, 10 * fn + fp))
    c_min = min(r[5] for r in rows)
    return pd.DataFrame(rows, columns=COLS), max(r[0] for r in rows if r[5] == c_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.saved / "predictions.npz")

    print("Поділ")
    rows = [(k, len(d[f"{k}_y"]), int(d[f"{k}_y"].sum()), d[f"{k}_y"].mean()) for k in ("train", "validation", "test")]
    print(pd.DataFrame(rows, columns=["split", "n", "fraud", "share"]).to_string(index=False))

    print("\nValidation")
    yv, sv = d["validation_y"], d["validation_scores"].astype(np.float64)
    table = search(yv, sv)
    table.to_csv(args.out / "validation_candidates.csv", index=False)
    t_star = float(best(table))
    rows = [
        (name, t, *confusion(yv, sv, t))
        for name, t in (("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", math.inf))
    ]
    print(pd.DataFrame(rows, columns=["rule"] + COLS).to_string(index=False))
    print(f"t* = {t_star!r}; мінімумів C: {int((table['C'] == table['C'].min()).sum())}")

    precision, recall, _ = precision_recall_curve(yv, sv)
    fpr, tpr, _ = roc_curve(yv, sv)
    auc = roc_auc_score(yv, sv)
    print(f"ROC-AUC = {auc:.6f}")
    for name, x, y_, xl, yl, title in (
        ("pr_curve.png", recall, precision, "Повнота (recall)", "Точність (precision)", "PR-крива, validation"),
        ("roc_curve.png", fpr, tpr, "FPR", "TPR (recall)", f"ROC-крива, validation (AUC = {auc:.4f})"),
    ):
        plt.figure(figsize=(6, 5))
        plt.plot(x, y_)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.savefig(args.out / name, dpi=150)
        plt.close()

    print("\nПеревірка пошуку порога")
    for name, (s, y) in CHECKS.items():
        fast = search(np.array(y), np.array(s))
        slow, t_slow = brute(y, s)
        same = np.array_equal(fast.sort_values("threshold").to_numpy(float), slow.sort_values("threshold").to_numpy(float))
        print(f"\nНабір {name} (прямий перебір):")
        print(slow.sort_values("threshold", ascending=False).to_string(index=False))
        print(f"таблиці збігаються: {same}; t*(швидкий) = {best(fast)}; t*(перебір) = {t_slow}")
        assert same and best(fast) == t_slow, f"Набір {name}: перевірка не пройдена"

    print("\nTest (t* зафіксовано)")
    yt, st, ids, amount = d["test_y"], d["test_scores"].astype(np.float64), d["test_row_ids"], d["test_amount"]
    rows = []
    for name, t in (("Стандартний поріг", 0.5), ("Мінімум вартості", t_star)):
        tp, fp, fn, tn, c = confusion(yt, st, t)
        rows.append((name, t, tp, fp, fn, tn, c, tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None))
    print(pd.DataFrame(rows, columns=["rule"] + COLS + ["precision", "recall"]).to_string(index=False))

    pred = st >= t_star
    errors = [(int(ids[i]), "FP" if pred[i] else "FN", int(yt[i]), st[i], amount[i]) for i in np.argsort(ids) if pred[i] != yt[i]]
    shown = [e for e in errors if e[1] == "FP"][:5] + [e for e in errors if e[1] == "FN"][:5]
    print("\nПерші помилки за номером рядка CSV:")
    print(pd.DataFrame(shown, columns=["row", "type", "label", "score", "Amount"]).to_string(index=False))


if __name__ == "__main__":
    main()