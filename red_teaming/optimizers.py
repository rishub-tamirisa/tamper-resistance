import torch
import schedulefree

def get_sgd_with_momentum(
    model, learning_rate, momentum: float = 0.9, warmup_steps: int = None
):
    """
    Returns an SGD optimizer with momentum.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: The momentum factor (default: 0.9).
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.SGD: SGD optimizer with momentum.
    """
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def get_sgd_with_nesterov_momentum(
    model, learning_rate, momentum: float = 0.9, warmup_steps: int = None
):
    """
    Returns an SGD optimizer with Nesterov momentum.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: The momentum factor (default: 0.9).
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.SGD: SGD optimizer with Nesterov momentum.
    """
    return torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True
    )

def get_adam(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    """
    Returns an Adam optimizer.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: Not used in Adam, included for consistency.
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.Adam: Adam optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_adamW(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    """
    Returns an AdamW optimizer.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: Not used in AdamW, included for consistency.
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.AdamW: AdamW optimizer.
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_adagrad(model, learning_rate, momentum: float = None, warmup_steps: int = None):
    """
    Returns an Adagrad optimizer.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: Not used in Adagrad, included for consistency.
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.Adagrad: Adagrad optimizer.
    """
    return torch.optim.Adagrad(model.parameters(), lr=learning_rate)

def get_adadelta(
    model, learning_rate, momentum: float = None, warmup_steps: int = None
):
    """
    Returns an Adadelta optimizer.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: Not used in Adadelta, included for consistency.
        warmup_steps: Not used in this function, included for consistency.
    
    Returns:
        torch.optim.Adadelta: Adadelta optimizer.
    """
    return torch.optim.Adadelta(model.parameters(), lr=learning_rate)

def get_adamW_schedule_free(
    model, learning_rate, momentum: float = None, warmup_steps: int = 100
):
    """
    Returns an AdamW optimizer with schedule-free learning rate adjustment.
    
    Args:
        model: The model whose parameters will be optimized.
        learning_rate: The learning rate for the optimizer.
        momentum: Not used in AdamW, included for consistency.
        warmup_steps: The number of warmup steps for the learning rate schedule (default: 100).
    
    Returns:
        schedulefree.AdamWScheduleFree: AdamW optimizer with schedule-free learning rate adjustment.
    """
    return schedulefree.AdamWScheduleFree(
        model.parameters(), lr=learning_rate, warmup_steps=warmup_steps
    )