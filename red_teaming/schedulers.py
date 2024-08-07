from torch.optim.lr_scheduler import LambdaLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim import Optimizer
import torch

def get_warmup_with_annealing_scheduler(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    num_devices_mult: int = 4,
    eta_min: float = 5e-6,
):
    """
    Creates a learning rate scheduler with linear warmup followed by cosine annealing.

    For more details on SGDR, see: https://timm.fast.ai/SGDR

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        max_steps: Maximum number of training steps.
        num_devices_mult: Multiplier for max_steps to account for multiple devices (default: 4).
        eta_min: Minimum learning rate (default: 5e-6).

    Returns:
        torch.optim.lr_scheduler.SequentialLR: A sequential scheduler combining warmup and annealing.
    """
    if warmup_steps == 0:
        raise ValueError("warmup_steps must be greater than 0")
    
    # Linear warmup scheduler
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    # Cosine annealing scheduler
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_devices_mult * max_steps, eta_min=eta_min
    )
    # Combine both schedulers
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )

def get_exponential_warmup_scheduler(
    optimizer,
    warmup_steps,
    initial_lr,
    final_lr,
):
    """
    Creates a learning rate scheduler with exponential warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        initial_lr: Initial learning rate.
        final_lr: Final learning rate after warmup.

    Returns:
        LambdaLR: A scheduler with exponential warmup.
    """
    warmup_lr_lambda = (
        lambda step: initial_lr * (final_lr / initial_lr) ** (step / warmup_steps)
        if step < warmup_steps
        else 1.0
    )
    lr_lambda = lambda step: warmup_lr_lambda(step)
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def get_linear_warmup_scheduler(
    optimizer,
    warmup_steps,
    initial_lr,
    final_lr,
):
    """
    Creates a learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        initial_lr: Initial learning rate.
        final_lr: Final learning rate after warmup.

    Returns:
        LambdaLR: A scheduler with linear warmup.
    """
    warmup_lr_lamda = (
        lambda step: initial_lr + (final_lr - initial_lr) * step / warmup_steps
        if step < warmup_steps
        else 1.0
    )
    lr_lambda = lambda step: warmup_lr_lamda(step)
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def get_no_scheduler(optimizer):
    """
    Creates a constant learning rate scheduler (no scheduling).

    Args:
        optimizer: The optimizer to schedule.

    Returns:
        LambdaLR: A scheduler with constant learning rate.
    """
    lr_lambda = lambda step: 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def get_sgdr_scheduler(
    optimizer: Optimizer,
    t_initial: int = 100,
    lr_min: float = 5e-6,
    cycle_mul: float = 1,
    cycle_decay: float = 1,
    cycle_limit: int = 1000,
    warmup_t=30,
    warmup_lr_init=0,
    warmup_prefix=True,
    t_in_epochs=True,
    noise_range_t=None,
    noise_pct=0.67,
    noise_std=1.0,
    noise_seed=42,
    k_decay=1.0,
    initialize=True,
):
    """
    Creates a Stochastic Gradient Descent with Warm Restarts (SGDR) scheduler.

    Args:
        optimizer (Optimizer): The optimizer to schedule.
        t_initial (int): Initial number of epochs in the first cycle.
        lr_min (float): Minimum learning rate.
        cycle_mul (float): Cycle length multiplier.
        cycle_decay (float): Cycle amplitude decay factor.
        cycle_limit (int): Maximum number of cycles.
        warmup_t (int): Number of warmup epochs.
        warmup_lr_init (float): Initial learning rate for warmup.
        warmup_prefix (bool): If True, warmup is done before cycling starts.
        t_in_epochs (bool): If True, t is treated as epochs, otherwise as steps.
        noise_range_t (tuple): Range of noise to add to the learning rate.
        noise_pct (float): Percentage of cycle amplitude to use as noise.
        noise_std (float): Standard deviation of noise.
        noise_seed (int): Seed for noise generation.
        k_decay (float): Learning rate decay factor.
        initialize (bool): If True, initialize the learning rates.

    Returns:
        CosineLRScheduler: An SGDR scheduler.
    """
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=t_initial,
        lr_min=lr_min,
        warmup_t=warmup_t,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=warmup_prefix,
        t_in_epochs=t_in_epochs,
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
        noise_range_t=noise_range_t,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
        k_decay=k_decay,
        initialize=initialize,
    )
    return scheduler