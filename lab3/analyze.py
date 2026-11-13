from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix
import common as C

RESULTS = C.HERE / "results"
RUNS_DIR = C.HERE / "runs"
SPLIT_CSV = C.HERE / "split.csv"
COLORS = {"scratch": "tab:gray", "frozen": "tab:blue", "unfreeze": "tab:red"}
STYLES = {1: "-", 2: "--", 3: ":"}


def load_logs():
    logs = {}
    for regime, seed in C.RUNS:
        p = RUNS_DIR / f"{regime}_seed{seed}" / "run.json"
        if not p.exists():
            raise SystemExit(f"Немає журналу {p}")
        logs[(regime, seed)] = json.loads(p.read_text(encoding="utf-8"))
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    args = ap.parse_args()
    root = args.data_root
    RESULTS.mkdir(exist_ok=True)
    md = ["Результати аналізу\n"]
    cur_hash = C.sha256_file(SPLIT_CSV)
    logs = load_logs()
    hashes = {l["split_sha256"] for l in logs.values()}
    if len(hashes) != 1 or hashes != {cur_hash}:
        raise SystemExit("ПОМИЛКА: SHA-256 split.csv у журналах не однаковий або не збігається з поточним split.csv. "
                         "Запуски на різних поділах порівнювати не можна - перенавчіть усі сім.")
    rows = C.read_split(SPLIT_CSV)
    md.append(f"SHA-256 split.csv однаковий у всіх 7 журналах і збігається з поточним: `{cur_hash}`\n")
    md.append("Сім запусків\n")
    md.append("| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с | Пристрій | Коміт (зміни?) |\n|---|---|---|---|---|---|---|---|")
    final = {}
    for i, ((regime, seed), l) in enumerate(logs.items(), 1):
        e8 = l["epochs"][-1]
        final[(regime, seed)] = e8["val_macro_f1"]
        g = l["git"]
        commit = f"{(g['commit'] or 'н/д')[:8]} ({'так' if g['uncommitted_changes'] else 'ні'})"
        md.append(f"| {i} | {regime} | {seed} | {e8['train_loss']:.4f} | {e8['val_macro_f1']:.4f} | "
                  f"{l['train_time_s']:.1f} | {l['device']} | {commit} |")
    commits = {l["git"]["commit"] for l in logs.values()}
    if len(commits) > 1:
        md.append("\nУвага: запуски зроблено на різних комітах - поясніть у README.")
    md.append("\nmacro-F1 validation за епохами:\n")
    md.append("| Запуск | " + " | ".join(str(e) for e in range(1, C.EPOCHS + 1)) + " |\n|---|" + "---|" * C.EPOCHS)
    for (regime, seed), l in logs.items():
        md.append(f"| {regime} s{seed} | " + " | ".join(f"{e['val_macro_f1']:.3f}" for e in l["epochs"]) + " |")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (regime, seed), l in logs.items():
        ep = [e["epoch"] for e in l["epochs"]]
        kw = dict(color=COLORS[regime], ls=STYLES[seed], marker="o", ms=3, label=f"{regime} s{seed}")
        axes[0].plot(ep, [e["train_loss"] for e in l["epochs"]], **kw)
        axes[1].plot(ep, [e["val_macro_f1"] for e in l["epochs"]], **kw)
    axes[0].set(title="Втрата на train", xlabel="епоха", ylabel="втрата")
    axes[1].set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "curves.png", dpi=130)
    plt.close(fig)

    def vals(r):
        return [final[(r, s)] for s in (1, 2, 3)]
    
    Ff, Fu = float(np.mean(vals("frozen"))), float(np.mean(vals("unfreeze")))
    Rf, Ru = max(vals("frozen")) - min(vals("frozen")), max(vals("unfreeze")) - min(vals("unfreeze"))
    R = max(Rf, Ru)
    Fs = final[("scratch", 1)]
    d_pre, d_unf = Ff - Fs, Fu - Ff
    md.append("\nРозклад ефектів\n")
    md.append("| Величина | Значення |\n|---|---|")
    for name, v in [("F_scratch,1", Fs), ("F̄_frozen", Ff), ("F̄_unfreeze", Fu), ("R_frozen", Rf),
                    ("R_unfreeze", Ru), ("R = max(R_frozen, R_unfreeze)", R),
                    ("Δ_pre = F̄_frozen − F_scratch,1", d_pre), ("Δ_unf = F̄_unfreeze − F̄_frozen", d_unf),
                    ("Δ_pre + Δ_unf = F̄_unfreeze − F_scratch,1", d_pre + d_unf)]:
        md.append(f"| {name} | {v:.4f} |")
    md.append(f"\n- Δ_pre: {C.interpret(d_pre, R)}\n- Δ_unf: {C.interpret(d_unf, R)}")
    md.append(f"\nЧас навчання (с, цикл навчання): " + ", ".join(
        f"{r} s{s}: {logs[(r, s)]['train_time_s']:.1f}" for r, s in C.RUNS))
    regime, seed = C.select_run(final)
    ck = torch.load(C.HERE / "lab3_model.pt", map_location="cpu", weights_only=True)
    if (ck["regime"], ck["seed"]) != (regime, seed) or ck["img_size"] != C.IMG_SIZE or list(ck["classes"]) != C.CLASSES:
        raise SystemExit(f"lab3_model.pt ({ck['regime']}, зерно {ck['seed']}) не відповідає правилу вибору "
                         f"за журналами ({regime}, зерно {seed}).")
    y_val = C.labels_of(rows, "validation")
    for key, l in logs.items():  # цілісність: F1 зі збережених логітів = F1 у журналі
        lg = np.load(RUNS_DIR / f"{key[0]}_seed{key[1]}" / "val_logits.npy")
        assert lg.shape == (len(y_val), 6)
        if abs(C.macro_f1(y_val, lg) - l["final_val_macro_f1"]) > 1e-9:
            raise SystemExit(f"val_logits.npy не збігається з журналом для {key}")
    md.append(f"\nВибір конфігурації (лише за validation)\n\nВибрано:{regime}, зерно {seed}, "
              f"macro-F1 validation = {final[(regime, seed)]:.4f}. lab3_model.pt відповідає правилу вибору ✓. "
              f"Режим із найбільшим середнім: {regime}.")
    if abs(d_unf) <= R:
        md.append("\nУвага: |Δ_unf| ≤ R - вибір між frozen і unfreeze визначив шум.")

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"])
    model.to("cpu")
    loader = C.make_eval_loader(rows, root, "test")
    logits = C.predict_logits(model, loader, torch.device("cpu"))
    np.save(RESULTS / "test_logits.npy", logits)
    test_rows = C.part_rows(rows, "test")
    y = C.labels_of(rows, "test")
    pred = logits.argmax(1)
    f1_test = C.macro_f1(y, logits)
    acc = float((pred == y).mean())
    n = len(y)
    se = float(np.sqrt(acc * (1 - acc) / n))
    f1_val = final[(regime, seed)]
    cm = confusion_matrix(y, pred, labels=list(range(6)))

    md.append("\nТестова оцінка (CPU, один прохід)\n")
    md.append(f"- macro-F1 test = **{f1_test:.4f}**; accuracy test = {acc:.4f} (n = {n})")
    md.append(f"- macro-F1 validation вибраного запуску = {f1_val:.4f}; різниця (validation − test) = {f1_val - f1_test:+.4f}")
    md.append(f"- SE частки правильних відповідей √(p(1−p)/n) = {se:.4f}; різниця в одиницях SE = {(f1_val - f1_test) / se:+.2f}\n")
    md.append("Матриця плутанини (рядки - справжній клас, стовпці - передбачений):\n")
    md.append("| справжній \\ передбачений | " + " | ".join(C.CLASSES) + " |\n|---|" + "---|" * 6)
    for i, c in enumerate(C.CLASSES):
        md.append(f"| {c} | " + " | ".join(str(v) for v in cm[i]) + " |")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="передбачений клас", ylabel="справжній клас",
           title=f"Матриця плутанини, test (macro-F1 = {f1_test:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    probs = torch.softmax(torch.from_numpy(logits).double(), dim=1).numpy()
    by_pair: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        if pred[i] != y[i]:
            by_pair.setdefault((int(y[i]), int(pred[i])), []).append(i)
    for k in by_pair:
        by_pair[k].sort(key=lambda i: test_rows[i]["file"])  # рядкове сортування за іменем
    order = sorted(by_pair.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    chosen = [(idxs[0], pair) for pair, idxs in order[:5]]
    if len(chosen) < 5:
        for pair, idxs in order:
            if len(chosen) >= 5:
                break
            if len(idxs) >= 2:
                chosen.append((idxs[1], pair))
    md.append("\nП'ять характерних помилок\n")
    md.append("| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test | Категорія |\n|---|---|---|---|---|---|")
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.3 * len(chosen), 3.8), squeeze=False)
    for ax, (i, (t, p)) in zip(axes[0], chosen):
        f = test_rows[i]["file"]
        md.append(f"| {f} | {C.CLASSES[t]} | {C.CLASSES[p]} | {probs[i, p]:.3f} | {len(by_pair[(t, p)])} | *(допишіть)* |")
        with Image.open(C.image_path(root, test_rows[i])) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{f}\n{C.CLASSES[t]} → {C.CLASSES[p]}\np = {probs[i, p]:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "five_errors.png", dpi=130)
    plt.close(fig)

    text = "\n".join(md)
    (RESULTS / "analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
