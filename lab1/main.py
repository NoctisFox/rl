import argparse
import numpy as np
from compare import compare_with_torch, numerical_check
from data import load_data
from forward_backward import backward, forward, init_params, cross_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", action="store_true", help="прибрати ділення на N у градієнті за логітами")
    args = ap.parse_args()

    X, y, X_test, y_test = load_data()

    params = init_params()
    logits, cache = forward(params, X)
    loss = float(cross_entropy(logits, y))
    grads = backward(params, cache, y, buggy=args.bug)

    print()
    print("РЕЖИМ: " + ("з навмисною помилкою" if args.bug else "правильна реалізація"))
    print()
    print(f"Найбільший за модулем елемент кожного градієнта:")
    for k, v in grads.items():
        print(f"  max|d{k}| = {np.abs(v).max():.12e}")
    print()

    ok1 = compare_with_torch(params, X, y, loss, grads)
    print()
    ok2 = numerical_check(params, X, y, grads)
    print()
    print("Звірка з PyTorch:", "ПРОЙДЕНО" if ok1 else "НЕ ПРОЙДЕНО")
    print("Чисельна перевірка:", "ПРОЙДЕНО" if ok2 else "НЕ ПРОЙДЕНО")
    if args.bug:
        print("Дослід: помилку " + ("виявлено." if not (ok1 and ok2) else "НЕ виявлено."))


if __name__ == "__main__":
    main()
