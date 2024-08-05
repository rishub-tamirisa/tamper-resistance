import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate import utils as accelerate_utils
from accelerate.optimizer import AcceleratedOptimizer, move_to_device
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def return_coin_flip_batch_selection(prob_of_selecting_retain_batch: float = 0.5, **kwargs):
    return torch.rand(1) < prob_of_selecting_retain_batch

def return_step_based_batch_selection(current_step: int, max_steps: int, prop_steps_for_batch_selection: float = 0.60, **kwargs):
    bool_result = current_step < int(max_steps * np.abs(prop_steps_for_batch_selection))
    return bool_result if prop_steps_for_batch_selection > 0 else not bool_result

def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in inputs.items() if k != "concept_label" and k != "labels"}

def get_distributed_random_number(accelerator: Accelerator):
    random_number = torch.rand(1).to(accelerator.device)
    dist.broadcast(random_number, src=0)
    accelerator.wait_for_everyone()
    return random_number.item()

def delete_optimizer(optim):
    # go through all states and delete the param groups
    for state in optim.state.values():
        state.clear()
    optim.param_groups = []
    del optim


def get_next_batch(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def next_n_batches(iterator, dataloader, n):
    batches = []
    for _ in range(n):
        batch, iterator = get_next_batch(iterator, dataloader)
        batches.append(batch)
    return batches, iterator


def distributed_sample_task(adversaries):
    # generate shared random number across all GPUs via broadcasting:
    # e.g., {task1: 0.33, task2: 0.66, task3: 0.01} etc
    task_probs = {
        adv.split(":")[0]: float(adv.split(":")[1]) for adv in adversaries.split(",")
    }
    task_type = random.choices(
        list(task_probs.keys()), weights=list(task_probs.values()), k=1
    )[0]
    dist.barrier()
    task_type = accelerate_utils.broadcast_object_list([task_type], 0)[0]
    return task_type


def distributed_sample_adversary_lr(adversary_lr_samples, accelerator):
    dist.barrier()
    rand_num = get_distributed_random_number(accelerator)
    adversary_lr = adversary_lr_samples[
        math.floor(rand_num * len(adversary_lr_samples))
    ]
    return adversary_lr


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or (
                        "decoder" in k
                    ):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch
