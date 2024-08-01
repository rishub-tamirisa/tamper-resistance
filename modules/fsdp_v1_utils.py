from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def fsdp_model_params(model: FSDP):
    sharded_params = set()
    nonsharded_params = set()  # `NO
    for i, handle in enumerate(model._all_handles):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        target_set.add(handle.flat_param)
        yield handle.flat_param
    for name, param in model.named_parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            yield param


class FSDPModelStorage:
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
        for i, param in enumerate(fsdp_model_params(model)):
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
        for i, param in enumerate(fsdp_model_params(model)):
            if mode == "params":
                param.data.copy_(self.storage_dict["params"][i].to(param.device))
            # assert either both storage and handle have grads or neither do
            if not skip_check:
                assert (i in self.storage_dict["grads"]) == (param.grad is not None)
            if i in self.storage_dict["grads"] and param.grad is not None:
                if mode == "grads":
                    param.grad += self.storage_dict["grads"][i].to(param.device)
                    # param.requires_grad_(True)
