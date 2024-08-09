
import torch
from accelerate.optimizer import AcceleratedOptimizer, move_to_device

def _optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def accelerated_optimizer_to_cpu(optimizer: AcceleratedOptimizer):
    assert isinstance(
        optimizer, AcceleratedOptimizer
    ), "optimizer must be an AcceleratedOptimizer"
    _optimizer_to(optimizer, "cpu")

def accelerated_optimizer_to_gpu(optimizer: AcceleratedOptimizer):
    assert isinstance(
        optimizer, AcceleratedOptimizer
    ), "optimizer must be an AcceleratedOptimizer"
    device = optimizer.accelerator_state.device
    state_dict = optimizer.state_dict()
    state_dict = move_to_device(state_dict, device)
    optimizer.load_state_dict(state_dict)