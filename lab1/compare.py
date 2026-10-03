import numpy as np
from forward_backward import loss_fn
from with_torch import torch_loss_and_grads


TOL_TORCH = 1e-12
TOL_NUM = 1e-7
EPS = 1e-6

NUM_TARGETS = [
    ("W1", (0, 0)),
    ("b1", (0,)),
    ("W2", (0, 0)),
    ("b2", (0,)),
]


def compare_with_torch(params, X, y, loss_np, grads_np):
    loss_t, grads_t = torch_loss_and_grads(params, X, y)

    print("Звірка з PyTorch")
    print(f"  Втрата NumPy   = {loss_np!r}")
    print(f"  Втрата PyTorch = {loss_t!r}")

    all_ok = True
    d = abs(loss_np - loss_t)
    ok = bool(np.isfinite(loss_np) and np.isfinite(loss_t) and d <= TOL_TORCH)
    all_ok &= ok
    print(f"  Втрата: макс. абс. різниця = {d:.3e}, перевірку пройдено: {'так' if ok else 'ні'}")

    for name in ("W1", "b1", "W2", "b2"):
        a, b = grads_np[name], grads_t[name]
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        d = float(np.abs(a - b).max())
        ok = finite and d <= TOL_TORCH
        all_ok &= ok
        print(f"  Градієнт {name}: макс. абс. різниця = {d:.3e}, " f"перевірку пройдено: {'так' if ok else 'ні'}")
    return all_ok


def numerical_check(params, X, y, grads_np):
    print("Чисельна перевірка градієнтів ")
    all_ok = True
    for name, idx in NUM_TARGETS:
        p = params[name]
        orig = p[idx]

        p[idx] = orig + EPS
        L_plus = loss_fn(params, X, y)
        p[idx] = orig - EPS
        L_minus = loss_fn(params, X, y)
        p[idx] = orig
        g_num = (L_plus - L_minus) / (2 * EPS)

        g_man = grads_np[name][idx]
        d = abs(g_num - g_man)
        ok = bool(np.isfinite(g_num) and np.isfinite(g_man) and d <= TOL_NUM)
        all_ok &= ok
        label = f"{name}[{', '.join(map(str, idx))}]"
        print(f"  {label}: backward() = {g_man:.12e}, чисельна = {g_num:.12e}, "
              f"|різниця| = {d:.3e}, перевірку пройдено: {'так' if ok else 'ні'}")
    return all_ok
