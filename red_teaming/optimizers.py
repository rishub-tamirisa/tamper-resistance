import torch
import schedulefree


def get_sgd_with_momentum(
    model, learning_rate, momentum: float = 0.9, warmup_steps: int = None
):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def get_sgd_with_nesterov_momentum(
    model, learning_rate, momentum: float = 0.9, warmup_steps: int = None
):
    return torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True
    )


def get_adam(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_adamW(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def get_adagrad(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    return torch.optim.Adagrad(model.parameters(), lr=learning_rate)


def get_adadelta(
    model, learning_rate, momentum: float = None, warmup_steps: int = None
):
    return torch.optim.Adadelta(model.parameters(), lr=learning_rate)


def get_adamW_schedule_free(
    model, learning_rate, momentum: float = None, warmup_steps: int = 100
):
    return schedulefree.AdamWScheduleFree(
        model.parameters(), lr=learning_rate, warmup_steps=warmup_steps
    )
