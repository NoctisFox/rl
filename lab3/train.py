from __future__ import annotations
import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import common as C

RESULTS = C.HERE / "results"
RUNS_DIR = C.HERE / "runs"
SPLIT_CSV = C.HERE / "split.csv"

BN_PREDICTION = (
    "Прогноз (записано до запуску): у заморожених модулях (conv1, layer1-layer4) параметри НЕ зміняться "
    "(requires_grad=False, оптимізатор їх не бачить), а ковзні статистики BN (running_mean, running_var, "
    "num_batches_tracked) ЗМІНЯТЬСЯ в усіх BN-шарах, бо model.train() вмикає оновлення статистик. "
    "fc змінюється як завжди. З кривої втрати на train це було б непомітно: втрата однаково спадає, "
    "бо градієнти йдуть лише в fc, а зсув статистик малий і не дає видимого стрибка."
)


def check_split(rows, md):
    ok = True
    table = C.split_table(rows)
    md.append("Контроль поділу (кількість / сума номерів у назвах файлів)\n")
    md.append("| Клас | train | validation | test |\n|---|---|---|---|")
    totals = {p: [0, 0] for p in C.PARTS}
    for lab in C.CLASSES:
        cells = []
        for p in C.PARTS:
            got, exp = table[(lab, p)], C.EXPECTED_SPLIT[lab][p]
            good = got == exp
            ok &= good
            totals[p][0] += got[0]
            totals[p][1] += got[1]
            cells.append(f"{got[0]} / {got[1]:,} {'✓' if good else '✗ (очікувано %d / %s)' % (exp[0], f'{exp[1]:,}')}".replace(",", " "))
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md.append("| Разом | " + " | ".join(f"{totals[p][0]} / {totals[p][1]:,}".replace(",", " ") for p in C.PARTS) + " |")
    sets = {p: {r["file"] for r in C.part_rows(rows, p)} for p in C.PARTS}
    disjoint = (not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"])
                and not (sets["validation"] & sets["test"]) and sum(map(len, sets.values())) == len(rows))
    ok &= disjoint
    md.append(f"\nУсі 18 клітинок збігаються: **{'так' if ok and disjoint else 'ні'}**. "
              f"Файлів у двох частинах одночасно немає: **{'так' if disjoint else 'ні'}**.\n")
    return ok


def check_batch_figure(rows, root, md):
    torch.manual_seed(1)
    loader = C.make_train_loader(rows, root)
    x, y = next(iter(loader))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
    for ax, img, lab in zip(axes.ravel(), x[:16], y[:16]):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(C.CLASSES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета (після аугментацій)")
    fig.tight_layout()
    fig.savefig(RESULTS / "batch_after_transforms.png", dpi=120)
    plt.close(fig)
    md.append("Пакет після перетворень\n\nРисунок: `results/batch_after_transforms.png`.\n")


def check_initial_loss(rows, root, device, md):
    md.append("Початкова втрата\n")
    md.append(f"Рівномірний прогноз: ln 6 = {math.log(6):.4f}\n")
    md.append("| Режим | Втрата L0 | L0 - ln 6 |\n|---|---|---|")
    crit = nn.CrossEntropyLoss()
    for regime in ["scratch", "frozen"]:
        model, _ = C.build_model(regime, 1, device)
        loader = C.make_train_loader(rows, root)
        C.set_train_mode(model, regime)
        x, y = next(iter(loader))
        with torch.no_grad():
            loss = crit(model(x.to(device)), y.to(device)).item()
        md.append(f"| {regime} | {loss:.4f} | {loss - math.log(6):+.4f} |")


def check_param_counts(device, md):
    ok = True
    md.append("Параметри та групи оптимізатора\n")
    md.append("| Режим | Усього | requires_grad=True | Очікувано | Групи (параметрів, lr) | requires_grad=False в оптимізаторі |\n|---|---|---|---|---|---|")
    for regime in C.REGIMES:
        model, opt = C.build_model(regime, 1, device)
        total = C.count_params(model.parameters())
        trainable = C.count_params(p for p in model.parameters() if p.requires_grad)
        groups = [(g["name"], C.count_params(g["params"]), g["lr"]) for g in opt.param_groups]
        frozen_in_opt = sum(1 for g in opt.param_groups for p in g["params"] if not p.requires_grad)
        good = (total == C.TOTAL_PARAMS and trainable == C.EXPECTED_TRAINABLE[regime]
                and sum(g[1] for g in groups) == trainable and frozen_in_opt == 0)
        ok &= good
        gtxt = "; ".join(f"{n}: {c:,}, lr={lr:g}".replace(",", " ") for n, c, lr in groups)
        md.append(f"| {regime} | {total:,} | {trainable:,} | {C.EXPECTED_TRAINABLE[regime]:,} | {gtxt} | "
                  f"{'жодного ✓' if frozen_in_opt == 0 else frozen_in_opt} |".replace(",", " "))
    md.append("")
    return ok


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_experiment(regime, rows, root, device, use_set_train_mode, mode_label):
    model, opt = C.build_model(regime, 1, device)
    loader = C.make_train_loader(rows, root)
    before = _snapshot(model)
    if use_set_train_mode:
        C.set_train_mode(model, regime)
    else:
        model.train()
    crit = nn.CrossEntropyLoss()
    for x, y in islice(loader, 5):
        opt.zero_grad()
        crit(model(x.to(device)), y.to(device)).backward()
        opt.step()
    after = _snapshot(model)

    table, all_ok = [], True
    for name in C.TOP_MODULES:
        keys = [k for k in before if k.startswith(name + ".")]
        stat_keys = [k for k in keys if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        par_keys = [k for k in keys if k not in stat_keys]
        p_ch = any(not torch.equal(before[k], after[k]) for k in par_keys)
        s_ch = any(not torch.equal(before[k], after[k]) for k in stat_keys)
        has_bn = bool(stat_keys)
        if use_set_train_mode:
            if name in C.EVAL_MODULES[regime]:
                exp_p, exp_s = False, False
            else:
                exp_p, exp_s = True, has_bn
        else:  # прогноз для досліду з model.train() у frozen
            exp_p, exp_s = name == "fc", has_bn
        good = p_ch == exp_p and (s_ch == exp_s if has_bn else True)
        all_ok &= good
        table.append((name, p_ch, s_ch if has_bn else None, good))
    k = "layer1.0.bn1."
    d_mean = (after[k + "running_mean"] - before[k + "running_mean"]).abs().max().item()
    d_var = (after[k + "running_var"] - before[k + "running_var"]).abs().max().item()
    return table, all_ok, d_mean, d_var


def _fmt_table(regime, table, label):
    yn = lambda v: "немає BN" if v is None else ("так" if v else "ні")
    lines = [f"**{label}**\n", "| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |\n|---|---|---|---|---|"]
    for name, p, s, good in table:
        lines.append(f"| {regime} | {name} | {'так' if p else 'ні'} | {yn(s)} | {'так' if good else 'ні'} |")
    return lines + [""]


def check_modes(rows, root, device, md):
    ok = check_param_counts(device, md)
    md.append("Що змінюється під час навчання (5 кроків, set_train_mode)\n")
    for regime in ["frozen", "unfreeze"]:
        table, good, _, _ = update_experiment(regime, rows, root, device, True, "")
        ok &= good
        md += _fmt_table(regime, table, f"Режим {regime}")
    md.append("Дослід із навмисною помилкою (frozen, model.train() замість set_train_mode)\n")
    md.append(BN_PREDICTION + "\n")
    print(BN_PREDICTION)
    table, good, d_mean, d_var = update_experiment("frozen", rows, root, device, False, "")
    ok &= good
    md += _fmt_table("frozen + model.train()", table, "Результат")
    _, _, c_mean, c_var = update_experiment("frozen", rows, root, device, True, "")
    md.append(f"Максимальна абсолютна зміна у `layer1.0.bn1` після 5 кроків: "
              f"running_mean = {d_mean:.3e}, running_var = {d_var:.3e} "
              f"(із set_train_mode: {c_mean:.1e} і {c_var:.1e}).\n")
    md.append("Збіг із прогнозом: **%s**\n"
              % ("так" if good else "ні"))
    return ok


def train_run(regime, seed, rows, root, device, val_loader, y_val, split_hash):
    run_dir = RUNS_DIR / f"{regime}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model, opt = C.build_model(regime, seed, device)
    loader = C.make_train_loader(rows, root)
    crit = nn.CrossEntropyLoss()
    n_train = len(loader.dataset)
    epochs, train_time, t_all = [], 0.0, time.perf_counter()
    logits = None
    for epoch in range(1, C.EPOCHS + 1):
        C.set_train_mode(model, regime)
        t0, total = time.perf_counter(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_time += time.perf_counter() - t0
        logits = C.predict_logits(model, val_loader, device)
        f1 = C.macro_f1(y_val, logits)
        epochs.append({"epoch": epoch, "train_loss": total / n_train, "val_macro_f1": f1})
        print(f"[{regime} seed{seed}] epoch {epoch}: loss={total / n_train:.4f} val_macro_f1={f1:.4f}", flush=True)
    total_time = time.perf_counter() - t_all

    np.save(run_dir / "val_logits.npy", logits)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, run_dir / "model.pt")
    log = {
        "regime": regime, "seed": seed,
        "config": {
            "split": {"procedure": "default_rng(0) по класах для train (200/клас); новий default_rng(0) для seg_test, cut=n//2",
                      "train": n_train, "validation": len(y_val)},
            "transforms": {
                "train": f"RandomResizedCrop({C.IMG_SIZE}, scale=(0.7, 1.0)) -> RandomHorizontalFlip() -> ToTensor() -> Normalize",
                "eval": f"Resize(({C.IMG_SIZE}, {C.IMG_SIZE})) -> ToTensor() -> Normalize",
                "mean": C.MEAN, "std": C.STD, "image_open": 'Pillow .convert("RGB")'},
            "model": {"arch": "resnet18", "weights": None if regime == "scratch" else "IMAGENET1K_V1",
                      "fc": "Linear(512, 6), стандартна ініціалізація",
                      "trainable_modules": C.TRAINABLE_MODULES[regime] or "all",
                      "eval_modules_during_training": C.EVAL_MODULES[regime],
                      "trainable_params": C.count_params(p for p in model.parameters() if p.requires_grad),
                      "total_params": C.count_params(model.parameters()),
                      "bn": {"momentum": 0.1, "eps": 1e-5, "note": "за замовчуванням"},
                      "dtype": "float32"},
            "optimizer": {"name": "AdamW", "weight_decay": C.WEIGHT_DECAY,
                          "groups": [{"name": g["name"], "params": C.count_params(g["params"]), "lr": g["lr"]}
                                     for g in opt.param_groups],
                          "defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}},
            "loss": "CrossEntropyLoss() (mean)",
            "loaders": {"train": {"batch_size": C.BATCH_TRAIN, "shuffle": True, "drop_last": False, "num_workers": 0,
                                  "generator": None, "sampler": None},
                        "eval": {"batch_size": C.BATCH_EVAL, "shuffle": False, "num_workers": 0}},
            "budget": {"epochs": C.EPOCHS, "steps_per_epoch": len(loader), "total_steps": C.EPOCHS * len(loader)},
            "not_used": "lr-розклади, обрізання градієнта, AMP, label smoothing, ваги класів, рання зупинка, вибір епохи за validation",
        },
        "git": C.git_info(), "split_sha256": split_hash, "versions": C.versions(),
        "device": C.device_name(device),
        "epochs": epochs, "final_val_macro_f1": epochs[-1]["val_macro_f1"],
        "train_time_s": train_time, "total_time_s": total_time,
        "time_note": "train_time_s - лише цикл навчання (без оцінки на validation); total_time_s - з оцінкою після кожної епохи",
    }
    (run_dir / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="тека з seg_train/ та seg_test/ (розпакований архів)")
    args = ap.parse_args()
    root = args.data_root
    RESULTS.mkdir(exist_ok=True)
    device = C.get_device()
    print("Пристрій:", C.device_name(device))

    rows = C.build_split(root)
    C.write_split(rows, SPLIT_CSV)
    split_hash = C.sha256_file(SPLIT_CSV)
    rows = C.read_split(SPLIT_CSV)  # далі всі дані - лише за split.csv
    print("split.csv sha256:", split_hash)

    md = ["Результати перевірок пунктів 1-2\n", f"SHA-256 split.csv: `{split_hash}`\n"]
    ok = check_split(rows, md)
    check_batch_figure(rows, root, md)
    check_initial_loss(rows, root, device, md)
    ok &= check_modes(rows, root, device, md)
    (RESULTS / "checks.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if not ok:
        raise SystemExit("Перевірки не збіглися з очікуваним (див. results/checks.md). "
                         "Виправте реалізацію до основного експерименту.")

    val_loader = C.make_eval_loader(rows, root, "validation", cache=True)
    y_val = C.labels_of(rows, "validation")
    final = {}
    for regime, seed in C.RUNS:
        log = train_run(regime, seed, rows, root, device, val_loader, y_val, split_hash)
        final[(regime, seed)] = log["final_val_macro_f1"]

    regime, seed = C.select_run(final)
    state = torch.load(RUNS_DIR / f"{regime}_seed{seed}" / "model.pt", map_location="cpu")
    torch.save({"state_dict": state, "regime": regime, "seed": seed,
                "classes": C.CLASSES, "img_size": C.IMG_SIZE}, C.HERE / "lab3_model.pt")
    print(f"Вибрано: {regime}, зерно {seed} (F1 validation = {final[(regime, seed)]:.4f}) -> lab3_model.pt")


if __name__ == "__main__":
    main()
