import numpy as np


def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return {
        "W1": W1,
        "b1": np.zeros(8),
        "W2": W2,
        "b2": np.zeros(3),
    }


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(logits, y):
    logp = log_softmax(logits)
    return -logp[np.arange(len(y)), y].mean()


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]   # (N, 8)
    A1 = np.maximum(Z1, 0.0)               # (N, 8)
    Z2 = A1 @ params["W2"] + params["b2"]  # (N, 3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return Z2, cache


def loss_fn(params, X, y):
    logits, _ = forward(params, X)
    return cross_entropy(logits, y)


def backward(params, cache, y, buggy=False):
    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    N = X.shape[0]

    P = np.exp(log_softmax(Z2))
    dZ2 = P.copy()
    dZ2[np.arange(N), y] -= 1.0
    if not buggy:
        dZ2 /= N

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
