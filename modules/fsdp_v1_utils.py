from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


# TODO: refactor for FSDP2
def fsdp_v1_model_params(model: FSDP):
    """
    Get all model parameters via FSDP handles
    """
    sharded_params = set()
    nonsharded_params = set()  # `NO
    for _, handle in enumerate(model._all_handles):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        target_set.add(handle.flat_param)
        yield handle.flat_param
    for _, param in model.named_parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            yield param


class FSDPModelStorage:
    """
    Storage for sharded model parameters and gradients for accumulation during TAR
    """

    def __init__(self):
        self.storage_dict = {
            "params": {},
            "grads": {},
        }

    def clear_params(self):
        self.storage_dict["params"].clear()

    def clear_grads(self):
        self.storage_dict["grads"].clear()

    def collect_param_or_grad(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        to_cpu: bool = False,
        mode: str = "grads",
        scale: float = 1.0,
    ):
        """
        Collect parameters or gradients from the FSDP model and store them.

        Args:
            model (FSDP): The FSDP model to collect from.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            to_cpu (bool): Whether to move the collected data to CPU.
            mode (str): Either "params" or "grads" to collect parameters or gradients.
            scale (float): Scaling factor for gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            if mode == "params":
                self.storage_dict["params"][i] = param.detach().clone()
                if to_cpu:
                    self.storage_dict["params"][i] = self.storage_dict["params"][
                        i
                    ].cpu()
            if param.grad is not None:
                if mode == "grads":
                    if i not in self.storage_dict["grads"]:
                        self.storage_dict["grads"][i] = (
                            param.grad.detach().clone() * scale
                        )
                    else:
                        self.storage_dict["grads"][i] = self.storage_dict["grads"][
                            i
                        ].to(param.grad.device)
                        self.storage_dict["grads"][i] += (
                            param.grad.detach().clone() * scale
                        )
                    if to_cpu:
                        self.storage_dict["grads"][i] = self.storage_dict["grads"][
                            i
                        ].cpu()

    def add_from_storage_to_model(
        self,
        model: FSDP = None,
        accelerator: Accelerator = None,
        skip_check: bool = False,
        mode: str = "grads",
    ):
        """
        Add parameters or gradients from storage to the FSDP model.

        Args:
            model (FSDP): The FSDP model to add to.
            accelerator (Accelerator): The Accelerator object (unused in this function).
            skip_check (bool): Whether to skip the assertion check for gradient existence.
            mode (str): Either "params" or "grads" to add parameters or gradients.
        """
        for i, param in enumerate(fsdp_v1_model_params(model)):
            if mode == "params":
                param.data.copy_(self.storage_dict["params"][i].to(param.device))
            # assert either both storage and handle have grads or neither do
            if not skip_check:
                assert (i in self.storage_dict["grads"]) == (param.grad is not None)
            if i in self.storage_dict["grads"] and param.grad is not None:
                if mode == "grads":
                    param.grad += self.storage_dict["grads"][i].to(param.device)
