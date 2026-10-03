import numpy as np
import torch


def torch_loss_and_grads(params, X, y):
    torch.set_default_dtype(torch.float64)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        model[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        model[2].bias.copy_(torch.from_numpy(params["b2"].copy()))

    Xt = torch.from_numpy(X.copy())
    yt = torch.from_numpy(y.copy())
    loss = torch.nn.functional.cross_entropy(model(Xt), yt)  # mean за замовчуванням
    loss.backward()

    grads = {
        "W1": model[0].weight.grad.numpy().T.copy(),
        "b1": model[0].bias.grad.numpy().copy(),
        "W2": model[2].weight.grad.numpy().T.copy(),
        "b2": model[2].bias.grad.numpy().copy(),
    }
    return loss.item(), grads
