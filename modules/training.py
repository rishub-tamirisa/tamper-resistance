import random
from typing import List, Union

import accelerate
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer, move_to_device
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.fsdp_v1_utils import FSDPModelStorage
from modules.objectives import (
    log_p_loss,
    dpo_loss_obj,
    obj_model_mse_representations,
    obj_standard_max_next_token,
    random_vector_cosine_obj,
    obj_mismatch_next_token,
    obj_max_entropy_next_token,
    max_entropy_loss,
    log_1_minus_p_loss,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from modules.utils import (
    delete_optimizer,
    distributed_sample_adversary_lr,
    distributed_sample_task,
    get_next_batch,
    next_n_batches,
    _filter_inputs,
    return_coin_flip_batch_selection,
    return_step_based_batch_selection,
)


#######################################################################
# BASELINES
#######################################################################


def random_mapping_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 1000,
    **kwargs,
):
    """
    Performs a training loop using random vectors for tamper-resistant model training.

    Args:
        model (Union[AutoModelForCausalLM, FSDP]): The model to be trained.
        dataloaders (dict[str, torch.utils.data.DataLoader]): Dictionary containing dataloaders for 'retain' and 'forget_train' datasets.
        optimizer (AcceleratedOptimizer): The optimizer for model parameter updates.
        accelerator (Accelerator): The Accelerator object for distributed training.
        num_epochs (int, optional): Number of training epochs. Defaults to 1.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients. Defaults to 8.
        max_steps (int, optional): Maximum number of training steps per epoch. Defaults to 1000.
        **kwargs: Additional keyword arguments.

    Returns:
        Union[AutoModelForCausalLM, FSDP]: The trained model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_iterator, forget_dataloader = (
        iter(dataloaders["forget_train"]),
        dataloaders["forget_train"],
    )
    stream_hash_table = torch.randn(
        model.config.vocab_size, model.config.hidden_size, requires_grad=False
    ).to(accelerator.device)
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(
                colour="blue",
                desc=f"Main Training Epoch: {epoch+1}",
                total=max_steps,
                dynamic_ncols=True,
            )
        for i in range(max_steps):
            total_lm_loss = 0
            total_cos_loss = 0
            for _ in range(gradient_accumulation_steps):
                retain_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )
                forget_batch, forget_iterator = get_next_batch(
                    forget_iterator, forget_dataloader
                )
                lm_loss, cos_loss = random_vector_cosine_obj(
                    model=model,
                    x_r=retain_batch,
                    x_f=forget_batch,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    stream_hash_table=stream_hash_table,
                    compute_lm_loss=True,
                )
                total_lm_loss += lm_loss
                total_cos_loss += cos_loss
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix({"lm_loss": total_lm_loss, "cos_loss": total_cos_loss})
                wandb.log({"lm_loss": total_lm_loss, "cos_loss": total_cos_loss})
    return model


def min_posterior_training_loop(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int = -1,
    **kwargs,
):
    """
    Performs minimum posterior training loop for a given model.

    This function trains the model to minimize the posterior probability on forget data
    while maximizing it on retain data.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing dataloaders for 'retain' and 'forget_train' datasets.
        optimizer (AcceleratedOptimizer): The optimizer used for training.
        accelerator (Accelerator): The Accelerator object for distributed training.
        num_epochs (int): The number of epochs to train for.
        gradient_accumulation_steps (int): The number of steps to accumulate gradients before performing a backward/update pass.
        max_steps (int, optional): The maximum number of steps to train for. If -1, train for the entire dataset. Defaults to -1.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The trained model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_iterator, forget_dataloader = (
        iter(dataloaders["forget_train"]),
        dataloaders["forget_train"],
    )
    total_length = max_steps
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
        for _ in range(total_length):
            total_loss = 0
            for _ in range(gradient_accumulation_steps):
                retain_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )

                batch_squeezed = {
                    key: value.squeeze()
                    for key, value in retain_batch.items()
                    if key in ["input_ids", "labels", "attention_mask"]
                }
                outputs = model(
                    **_filter_inputs(batch_squeezed), output_hidden_states=False
                )
                retain_loss = (
                    log_p_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                accelerator.backward(retain_loss)

                forget_batch, forget_iterator = get_next_batch(
                    forget_iterator, forget_dataloader
                )
                forget_loss = (
                    log_1_minus_p_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                accelerator.backward(forget_loss)
                total_loss += retain_loss.item() + forget_loss.item()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix({"loss": total_loss})
                wandb.log({"loss": total_loss})
    return model


def max_entropy_training_loop(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int = -1,
    **kwargs,
):
    """
    Performs a training loop using a max entropy loss on forget-set data.

    This function trains the model on retain data using log probability loss and on forget data
    using max entropy loss. It supports distributed training and gradient accumulation.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloaders (dict[str, torch.utils.data.DataLoader]): Dictionary containing dataloaders
            for 'retain' and 'forget_train' datasets.
        optimizer (AcceleratedOptimizer): The optimizer for model parameter updates.
        accelerator (Accelerator): The Accelerator object for distributed training.
        num_epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        max_steps (int, optional): Maximum number of training steps per epoch. Defaults to -1.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The trained model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_iterator, forget_dataloader = (
        iter(dataloaders["forget_train"]),
        dataloaders["forget_train"],
    )
    total_length = max_steps
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
        for _ in range(total_length):
            total_loss = 0
            for _ in range(gradient_accumulation_steps):
                retain_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )
                batch_squeezed = {
                    key: value.squeeze()
                    for key, value in retain_batch.items()
                    if key in ["input_ids", "labels", "attention_mask"]
                }
                outputs = model(
                    **_filter_inputs(batch_squeezed), output_hidden_states=False
                )
                retain_loss = (
                    log_p_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                accelerator.backward(retain_loss)

                forget_batch, forget_iterator = get_next_batch(
                    forget_iterator, forget_dataloader
                )
                batch_squeezed = {
                    key: value.squeeze()
                    for key, value in retain_batch.items()
                    if key in ["input_ids", "labels", "attention_mask"]
                }
                outputs = model(
                    **_filter_inputs(batch_squeezed), output_hidden_states=False
                )
                forget_loss = (
                    max_entropy_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                accelerator.backward(forget_loss)
                total_loss += retain_loss.item() + forget_loss.item()

            optimizer.step()
            model.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix({"loss": total_loss})
                wandb.log({"loss": total_loss})
    return model


def llmu_training_loop(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int = -1,
    **kwargs,
):
    """
    Performs a training loop from the LLMU (Large Language Model Unlearning) paper (https://arxiv.org/pdf/2310.10683).

    This function implements a training loop with three types of losses:
    1. Retain loss: For retaining existing knowledge
    2. Forget loss: For forgetting specific information
    3. Mismatch loss: For ensuring mismatched predictions on forgotten data

    We replace the KL-divergence loss w/base model from the original paper with a cross-entropy loss on the retain set.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloaders (dict[str, torch.utils.data.DataLoader]): Dictionary containing dataloaders
            for 'retain' and 'forget_train' datasets.
        optimizer (AcceleratedOptimizer): The optimizer for model parameter updates.
        accelerator (Accelerator): The Accelerator object for distributed training.
        num_epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        max_steps (int, optional): Maximum number of training steps per epoch. Defaults to -1 (no limit).
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The trained model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_iterator, forget_dataloader = (
        iter(dataloaders["forget_train"]),
        dataloaders["forget_train"],
    )
    total_length = max_steps
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
        for _ in range(total_length):
            total_loss = 0
            for _ in range(gradient_accumulation_steps):
                retain_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )
                retain_loss = obj_standard_max_next_token(model, retain_batch)
                retain_loss = retain_loss / gradient_accumulation_steps
                accelerator.backward(retain_loss)

                forget_batch, forget_iterator = get_next_batch(
                    forget_iterator, forget_dataloader
                )
                forget_loss = obj_standard_max_next_token(model, forget_batch) * -1
                forget_loss = forget_loss / gradient_accumulation_steps

                accelerator.backward(forget_loss)

                mismatch_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )
                mismatch_loss = obj_mismatch_next_token(
                    model, forget_batch, mismatch_batch
                )
                mismatch_loss = mismatch_loss / gradient_accumulation_steps
                accelerator.backward(mismatch_loss)

                total_loss += (
                    retain_loss.item() + forget_loss.item() + mismatch_loss.item()
                )

            optimizer.step()
            model.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix({"loss": total_loss})
                wandb.log({"loss": total_loss})
    return model


#######################################################################
# RED TEAMING
#######################################################################


def single_dataloader_accel_finetune_loop(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    retain_dataloader: torch.utils.data.DataLoader,
    forget_train_dataloader: torch.utils.data.DataLoader,
    forget_test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: Accelerator,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int = -1,
    **kwargs,
):
    """
    Performs a single dataloader accelerated finetuning loop.

    This function finetunes the model using either the retain or forget dataset,
    based on the specified finetuning_data_type. It supports distributed training
    and gradient accumulation.

    Args:
        model (torch.nn.Module): The model to be finetuned.
        tokenizer (AutoTokenizer): The tokenizer for processing input data.
        retain_dataloader (torch.utils.data.DataLoader): DataLoader for retain dataset.
        forget_train_dataloader (torch.utils.data.DataLoader): DataLoader for forget training dataset.
        forget_test_dataloader (torch.utils.data.DataLoader): DataLoader for forget testing dataset.
        optimizer (torch.optim.Optimizer): The optimizer for model parameter updates.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        accelerator (Accelerator): The Accelerator object for distributed training.
        num_epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        max_steps (int, optional): Maximum number of training steps per epoch. Defaults to -1.
        **kwargs: Additional keyword arguments, including finetuning_data_type and scheduler_type.

    Returns:
        torch.nn.Module: The finetuned model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)

    total_length = max_steps

    if kwargs["finetuning_data_type"] == "retain":
        with_grad_iterator = iter(retain_dataloader)
        with_no_grad_iterator = iter(forget_train_dataloader)
        with_grad_dataloader = retain_dataloader
        with_no_grad_dataloader = forget_train_dataloader
        wandb_with_grad_label = "finetuning_retain_loss"
        wandb_with_no_grad_label = "finetuning_training_loss"
    elif kwargs["finetuning_data_type"] == "forget":
        with_grad_iterator = iter(forget_train_dataloader)
        with_no_grad_iterator = iter(retain_dataloader)
        with_grad_dataloader = forget_train_dataloader
        with_no_grad_dataloader = retain_dataloader
        wandb_with_grad_label = "finetuning_training_loss"
        wandb_with_no_grad_label = "finetuning_retain_loss"
    else:
        raise ValueError("Invalid finetune type")

    for epoch in range(num_epochs):
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for i in range(max_steps):
            with_grad_loss = 0
            with_no_grad_loss = 0
            for _ in range(gradient_accumulation_steps):
                accelerator.wait_for_everyone()
                batch, with_grad_iterator = get_next_batch(
                    with_grad_iterator, with_grad_dataloader
                )
                batch_squeezed = {
                    key: value.squeeze()
                    for key, value in batch.items()
                    if key in ["input_ids", "labels", "attention_mask"]
                }
                outputs = model(
                    **_filter_inputs(batch_squeezed), output_hidden_states=False
                )
                loss = (
                    log_p_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                with_grad_loss += loss.item()
                accelerator.backward(loss)
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        wandb_with_grad_label: with_grad_loss,
                        wandb_with_no_grad_label: with_no_grad_loss,
                    }
                )
            optimizer.step()
            optimizer.zero_grad()
            if kwargs["scheduler_type"] == "sgdr":
                scheduler.step(i)
            else:
                scheduler.step()
            model.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix({"loss": with_grad_loss})
    return model


def double_dataloader_accel_finetune_loop(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    retain_dataloader: torch.utils.data.DataLoader,
    forget_train_dataloader: torch.utils.data.DataLoader,
    forget_test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: Accelerator,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: int = -1,
    **kwargs,
):
    """
    Performs accelerated finetuning using two dataloaders for retain and forget datasets.

    Args:
        model (torch.nn.Module): The model to be finetuned.
        tokenizer (AutoTokenizer): The tokenizer for processing input text.
        retain_dataloader (torch.utils.data.DataLoader): DataLoader for the retain dataset.
        forget_train_dataloader (torch.utils.data.DataLoader): DataLoader for the forget training dataset.
        forget_test_dataloader (torch.utils.data.DataLoader): DataLoader for the forget testing dataset.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        accelerator (Accelerator): Accelerator for distributed training.
        num_epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before performing a backward/update pass.
        max_steps (int, optional): Maximum number of training steps. Defaults to -1 (no limit).
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The finetuned model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)

    total_length = max_steps

    if kwargs["finetuning_data_type"] == "retain":
        with_grad_iterator = iter(retain_dataloader)
        with_no_grad_iterator = iter(forget_train_dataloader)
        with_grad_dataloader = retain_dataloader
        with_no_grad_dataloader = forget_train_dataloader
        wandb_with_grad_label = "finetuning_retain_loss"
        wandb_with_no_grad_label = "finetuning_training_loss"
    elif kwargs["finetuning_data_type"] == "forget":
        with_grad_iterator = iter(forget_train_dataloader)
        with_no_grad_iterator = iter(retain_dataloader)
        with_grad_dataloader = forget_train_dataloader
        with_no_grad_dataloader = retain_dataloader
        wandb_with_grad_label = "finetuning_training_loss"
        wandb_with_no_grad_label = "finetuning_retain_loss"
    else:
        raise ValueError("Invalid finetune type")

    for epoch in range(num_epochs):
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for i in range(max_steps):
            finetuning_loss = 0
            for _ in range(gradient_accumulation_steps):
                accelerator.wait_for_everyone()
                if kwargs["batch_selection_method"](
                    current_step=i,
                    max_steps=max_steps,
                    prop_steps_for_batch_selection=kwargs[
                        "prop_steps_for_batch_selection"
                    ],
                ):
                    batch, with_grad_iterator = get_next_batch(
                        with_grad_iterator, with_grad_dataloader
                    )
                else:
                    batch, with_no_grad_iterator = get_next_batch(
                        with_no_grad_iterator, with_no_grad_dataloader
                    )

                accelerator.wait_for_everyone()

                batch_squeezed = {
                    key: value.squeeze()
                    for key, value in batch.items()
                    if key in ["input_ids", "labels", "attention_mask"]
                }
                outputs = model(
                    **_filter_inputs(batch_squeezed), output_hidden_states=False
                )
                loss = (
                    log_p_loss(
                        outputs.logits, batch_squeezed.get("labels"), model.vocab_size
                    )
                    / gradient_accumulation_steps
                )
                finetuning_loss += loss.item()
                accelerator.backward(loss)
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                wandb.log({"finetuning_training_loss": finetuning_loss})

            optimizer.step()
            optimizer.zero_grad()
            if kwargs["scheduler_type"] == "sgdr":
                scheduler.step(i)
            else:
                scheduler.step()
            model.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix({"loss": finetuning_loss})

    return model


#######################################################################
# TAMPER RESISTANCE
#######################################################################


def adversary_next_token_obj_step(
    model: torch.nn.Module,
    adversary_batches: list,
    accelerator: Accelerator,
    sub_pbar: tqdm,
    gradient_accumulation_steps: int,
) -> float:
    """
    Perform an adversarial next token objective step.

    This function computes the loss for the adversarial next token objective,
    accumulates gradients, and updates the progress bar and logging.

    Args:
        model (torch.nn.Module): The model being trained.
        adversary_batches (list): A list of batches for adversarial training.
        accelerator (Accelerator): The Hugging Face Accelerator object.
        sub_pbar (tqdm): A progress bar for displaying step progress.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.

    Returns:
        float: The total loss accumulated over the gradient accumulation steps.
    """
    total_loss = 0
    for i in range(gradient_accumulation_steps):
        loss = obj_standard_max_next_token(model, adversary_batches[i])
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        total_loss += loss.item()
    if accelerator.is_main_process:
        sub_pbar.update(1)
        sub_pbar.set_postfix({"inner loss": total_loss})
        wandb.log({"inner_next_token_loss": total_loss})
    return total_loss


def tamper_resistance_obj(
    batches: list,
    model: torch.nn.Module,
    gradient_accumulation_steps: int,
    accelerator: Accelerator,
    scale: float = 0.5,
    tamper_resistance_loss_type: str = "max_entropy",
):
    """
    Compute the tamper resistance objective loss.

    This function calculates the loss for the tamper resistance objective,
    which can be either max entropy or DPO (Direct Preference Optimization) based. It also includes a diagnostic loss for evaluating the next-token loss on the tamper-resistance held-out batch x_tr.

    Args:
        batches (list): List of input batches for max entropy calculation.
        dpo_pref_batches (list): List of preference batches for DPO calculation.
        model (torch.nn.Module): The model being trained.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        accelerator (Accelerator): The Hugging Face Accelerator object.
        scale (float, optional): Scaling factor for the loss. Defaults to 0.5.
        tamper_resistance_loss_type (str, optional): Type of tamper resistance loss to use.
            Can be "max_entropy" or "dpo". Defaults to "max_entropy".

    Returns:
        float: The total accumulated loss.
    """
    total_loss = 0
    total_diagnostic_loss = 0
    for i in range(gradient_accumulation_steps):
        diagnostic_loss = None
        loss = None
        if tamper_resistance_loss_type == "max_entropy":
            diagnostic_name = "next_token"
            loss, diagnostic_loss = obj_max_entropy_next_token(model, batches[i])
            loss = loss * scale
            diagnostic_loss = diagnostic_loss / gradient_accumulation_steps
        elif tamper_resistance_loss_type == "dpo":
            diagnostic_name = "reward_accs"
            loss, reward_accs = dpo_loss_obj(
                policy_model=model,
                batch=batches[i],
                accelerator=accelerator,
                scale=scale,
                gradient_accumulation_steps=gradient_accumulation_steps,
                backprop=True,
            )
            total_loss += loss
            total_diagnostic_loss += reward_accs
            continue  # backprop done in function

        loss = loss / (gradient_accumulation_steps)
        accelerator.backward(loss)
        total_loss += loss.item()
        total_diagnostic_loss += diagnostic_loss
    if accelerator.is_main_process:
        wandb.log(
            {
                f"tr_{tamper_resistance_loss_type}_loss": total_loss / scale,
                f"tr_{diagnostic_name}_loss": total_diagnostic_loss,
            }
        )
    return total_loss


def inner_loop_step(
    model: Union[AutoModelForCausalLM, FSDP],
    adversary_batches: List[torch.Tensor],
    meta_forget_batches: List[torch.Tensor],
    inner_optimizer: AcceleratedOptimizer,
    inner_scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 8,
    meta_grad_scale: float = 0.25,
    model_storage: FSDPModelStorage = None,
    sub_pbar: tqdm = None,
    tamper_resistance_loss_type: str = "max_entropy",
    compute_tamper_resistance_grad: bool = True,
) -> float:
    """
    Perform a single inner loop step in the tamper-resistant training process.

    This function executes an adversarial step, updates the model, and optionally computes
    the tamper resistance gradient.

    Args:
        model (Union[AutoModelForCausalLM, FSDP]): The model being trained.
        adversary_batches (List[torch.Tensor]): Batches of adversarial data.
        meta_forget_batches (List[torch.Tensor]): Batches of meta-forget data.
        dpo_pref_batches (List[torch.Tensor]): Batches of DPO preference data.
        inner_optimizer (AcceleratedOptimizer): The optimizer for the inner loop.
        inner_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        accelerator (Accelerator): The Hugging Face Accelerator object.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients. Defaults to 8.
        meta_grad_scale (float, optional): Scaling factor for meta gradients. Defaults to 0.25.
        model_storage (FSDPModelStorage, optional): Storage for FSDP model parameters and gradients.
        sub_pbar (tqdm, optional): Progress bar for sub-steps.
        tamper_resistance_loss_type (str, optional): Type of tamper resistance loss. Defaults to "ascent_grad".
        compute_tamper_resistance_grad (bool, optional): Whether to compute tamper resistance gradient. Defaults to True.

    Returns:
        float: The computed tamper resistance loss.
    """

    # Adversary next-token objective step
    adversary_next_token_obj_step(
        model=model,
        adversary_batches=adversary_batches,
        accelerator=accelerator,
        sub_pbar=sub_pbar,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    inner_optimizer.step()
    if inner_scheduler:
        inner_scheduler.step()
    model.zero_grad(set_to_none=True)

    # Compute tamper-resistance loss
    tamper_resistance_loss = 0
    if compute_tamper_resistance_grad:
        tamper_resistance_loss = tamper_resistance_obj(
            batches=meta_forget_batches,
            model=model,
            gradient_accumulation_steps=gradient_accumulation_steps,
            accelerator=accelerator,
            scale=meta_grad_scale,
            tamper_resistance_loss_type=tamper_resistance_loss_type,
        )

        # Accumulate sharded TR loss grads in FSDPModelStorage data structure
        # This is done so everything is computed in place
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=False,
            mode="grads",
        )
        # Clear grads from model to be ready for next adversary step
        model.zero_grad(set_to_none=False)
    return tamper_resistance_loss


def schedule(i: int = None, K: int = None, schedule_lambda: float = 0.5):
    """
    Calculate a schedule value based on the current step and total steps.

    This function computes an exponential schedule value used for weighting
    or scaling purposes during TAR.

    Args:
        i (int): The current step or iteration number.
        K (int): The total number of steps or iterations.
        schedule_lambda (float, optional): A scaling factor for the exponent. Defaults to 0.5.

    Returns:
        float: The computed schedule value as a Python float.
    """
    return torch.exp(schedule_lambda * (torch.tensor(i) - (K - 1))).item()


def _sample_switching_point(
    switching_point_coeffs: str, tar_inner_loop_steps: int
) -> int:
    coeffs = {
        k: float(v)
        for k, v in [item.split(":") for item in switching_point_coeffs.split(",")]
    }
    M = int(
        torch.distributions.Beta(coeffs["alpha"], coeffs["beta"]).sample()
        * tar_inner_loop_steps
    )
    M = accelerate.utils.broadcast_object_list([M], 0)[0]
    return M


def tar_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 2,
    max_steps: int = 1000,
    tar_inner_loop_steps: int = 4,
    tar_tamper_resistance_loss_lower_bound: float = -11.76,
    tar_tamper_resistance_grad_scale: float = 4.0,
    tar_tamper_resistance_loss_type: str = "max_entropy",
    schedule_lambda: float = 0.5,
    inner_optimizer_warmup_steps: float = 20,
    unbounded: bool = False,
    use_weighting_schedule: bool = False,
    adversary_dist_types: str = "next_token:0.65,benign_next_token:0.35",
    adversary_lr_schedulers: str = "constant:1.0,linear_warmup:0.25",
    tar_num_tasks_sampled: int = 3,
    adversary_lr_samples: str = "2e-5,4e-5,1e-4",
    tar_inner_loop_subsample: int = 1,
    tar_retain_scale: float = 1.0,
    retain_representations: bool = False,
    switching_point_coeffs: str = "alpha:6.0,beta:3.0",
    retain_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    **kwargs,
):
    """
    Perform the Tamper-Attack Resistance (TAR) training loop.

    This function implements the main training loop for the TAR method, which aims to make
    language models resistant to tampering attacks while preserving their capabilities.

    Args:
        model (Union[AutoModelForCausalLM, FSDP]): The model to be trained.
        dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary of data loaders for different tasks.
        optimizer (AcceleratedOptimizer): The optimizer for the outer loop.
        accelerator (Accelerator): The Hugging Face Accelerator object for distributed training.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients. Defaults to 2.
        max_steps (int, optional): Maximum number of training steps. Defaults to 1000.
        tar_inner_loop_steps (int, optional): Number of steps in the inner loop. Defaults to 4.
        tar_tamper_resistance_loss_lower_bound (float, optional): Lower bound for the tamper resistance loss. Defaults to -15.0.
        tar_tamper_resistance_grad_scale (float, optional): Scaling factor for meta-gradients. Defaults to 0.25.
        tar_tamper_resistance_loss_type (str, optional): Type of tamper resistance loss. Defaults to "max_entropy".
        schedule_lambda (float, optional): Lambda parameter for the schedule function. Defaults to 0.5.
        inner_optimizer_warmup_steps (float, optional): Warmup steps for the inner optimizer. Defaults to 20.
        unbounded (bool, optional): Whether to use unbounded tamper resistance loss. Defaults to False.
        use_weighting_schedule (bool, optional): Whether to use a weighting schedule. Defaults to False.
        adversary_dist_types (str, optional): Distribution types for adversary sampling. Defaults to "next_token:0.65,benign_next_token:0.35".
        adversary_lr_schedulers (str, optional): Learning rate schedulers for adversary. Defaults to "constant:1.0,linear_warmup:0.25".
        tar_num_tasks_sampled (int, optional): Number of tasks to sample in each outer loop. Defaults to 3.
        adversary_lr_samples (str, optional): Learning rate samples for adversary. Defaults to "2e-5,4e-5,1e-4".
        tar_inner_loop_subsample (int, optional): Subsampling rate for inner loop. Defaults to 1.
        tar_random_subsample (bool, optional): Whether to use random subsampling. Defaults to False.
        tar_retain_scale (float, optional): Scaling factor for retain loss. Defaults to 1.0.
        retain_representations (bool, optional): Whether to retain representations. Defaults to False.
        switching_point_coeffs (str, optional): Coefficients for switching point in beta distribution. Defaults to "alpha:6.0,beta:3.0".
        retain_model_name (str, optional): Name of the model to retain representations. Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
        **kwargs: Additional keyword arguments.

    Returns:
        Union[AutoModelForCausalLM, FSDP]: The trained model.
    """
    model.config.use_cache = False
    model.train()

    # Retain and heldout forget dataloaders use `retain` and `meta` keys from dataloader funcs
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_val_iterator, forget_task_val_dataloader = (
        iter(dataloaders["meta"]),
        dataloaders["meta"],
    )

    # Adversary dataloaders use the remaining keys
    adversary_dataloaders = {
        key: {"iter": iter(value), "dataloader": value}
        for key, value in dataloaders.items()
        if key not in ["retain", "meta"]
    }

    if accelerator.is_main_process:
        pbar = tqdm(
            colour="blue",
            desc=f"Outer Training Loop",
            total=max_steps,
            dynamic_ncols=True,
        )

    model_storage = FSDPModelStorage()
    adversary_lr_samples = [float(lr) for lr in adversary_lr_samples.split(",")]

    # Setup model for representation-engineering retain loss
    retain_model = None
    if retain_representations:
        retain_model = AutoModelForCausalLM.from_pretrained(retain_model_name)
        retain_model.config.use_cache = False
        retain_model = accelerator.prepare_model(retain_model)
        retain_model.eval()

    # FSDP requires initial forward/backward for sharded params to be accessible in `FlatParamHandle`
    # Necessary for storing sharded params in FSDPModelStorage before the first TR step
    accelerator.backward(
        obj_standard_max_next_token(model, next(retain_iterator), accelerator)
    )
    model.zero_grad(set_to_none=False)

    tamper_resistance_loss = 0
    for _ in range(max_steps):
        tamper_resistance_loss = 0
        # Save params for tamper-resistance-optimizer step
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="params",
        )

        optimizer.load_state_dict(move_to_device(optimizer.state_dict(), "cpu"))
        for _ in range(tar_num_tasks_sampled):
            sub_pbar = None
            adversary_type = distributed_sample_task(adversary_dist_types)
            if accelerator.is_main_process:
                sub_pbar = tqdm(
                    colour="blue",
                    desc=f"Inner Training Loop ({adversary_type})",
                    total=tar_inner_loop_steps,
                    dynamic_ncols=True,
                )
            adversary_lr_scheduler = distributed_sample_task(adversary_lr_schedulers)

            # Sample heldout tamper-resistance batches (x_tr in Algorithm 1)
            meta_forget_batches, forget_val_iterator = next_n_batches(
                forget_val_iterator,
                forget_task_val_dataloader,
                gradient_accumulation_steps,
            )

            # Sample adversary learning rate
            adversary_lr = distributed_sample_adversary_lr(
                adversary_lr_samples, accelerator
            )

            # Setup adversary optimizer
            inner_optimizer = torch.optim.AdamW(model.parameters(), lr=adversary_lr)
            inner_optimizer = accelerator.prepare_optimizer(inner_optimizer)
            inner_scheduler = None

            # Setup adversary learning rate scheduler
            if adversary_lr_scheduler == "linear_warmup":
                inner_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    inner_optimizer,
                    lambda step: tar_inner_loop_steps / inner_optimizer_warmup_steps,
                )
                inner_scheduler = accelerator.prepare(inner_scheduler)

            # Sample switching point from beta distribution
            M = None
            if adversary_type == "retain_forget_switch":
                M = _sample_switching_point(
                    switching_point_coeffs, tar_inner_loop_steps
                )
                accelerator.print(f"Switching at step: {M}")

            for inner_step in range(tar_inner_loop_steps):
                # Sample adversary batches
                _adversary_type = adversary_type
                if adversary_type == "retain_forget_switch":
                    if inner_step < M:
                        _adversary_type = "adv_retain"
                    else:
                        _adversary_type = "forget_train"
                (
                    adversary_batches,
                    adversary_dataloaders[_adversary_type]["iter"],
                ) = next_n_batches(
                    adversary_dataloaders[_adversary_type]["iter"],
                    adversary_dataloaders[_adversary_type]["dataloader"],
                    gradient_accumulation_steps,
                )

                # Per-step tamper-resistance loss weighting schedule
                scheduled_weighting = (
                    schedule(inner_step, tar_inner_loop_steps, schedule_lambda)
                    if use_weighting_schedule
                    else 1 / tar_inner_loop_steps
                )

                # Whether to compute TR grad for current step (sub-sampling trick from appendix)
                compute_tamper_resistance_grad = (
                    inner_step + 1
                ) % tar_inner_loop_subsample == 0

                # Compute adversary step and tamper-resistance loss
                tamper_resistance_loss += inner_loop_step(
                    model=model,
                    adversary_batches=adversary_batches,
                    meta_forget_batches=meta_forget_batches,
                    inner_optimizer=inner_optimizer,
                    inner_scheduler=inner_scheduler,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    meta_grad_scale=tar_tamper_resistance_grad_scale
                    * scheduled_weighting
                    / tar_num_tasks_sampled,
                    model_storage=model_storage,
                    sub_pbar=sub_pbar,
                    tamper_resistance_loss_type=tar_tamper_resistance_loss_type,
                    compute_tamper_resistance_grad=compute_tamper_resistance_grad,
                )

            inner_optimizer.load_state_dict(
                move_to_device(inner_optimizer.state_dict(), "cpu")
            )
            delete_optimizer(inner_optimizer)

            model_storage.add_from_storage_to_model(
                model=model,
                accelerator=accelerator,
                skip_check=True,
                mode="params",
            )
        optimizer.load_state_dict(
            move_to_device(optimizer.state_dict(), optimizer.accelerator_state.device)
        )

        # Representation-engineering retain loss
        outer_retain_batches, retain_iterator = next_n_batches(
            retain_iterator, retain_dataloader, gradient_accumulation_steps
        )
        total_retain_loss = 0
        for i in range(gradient_accumulation_steps):
            if retain_representations:
                retain_loss = obj_model_mse_representations(
                    model, outer_retain_batches[i], retain_model
                )
            else:
                retain_loss = obj_standard_max_next_token(
                    model, outer_retain_batches[i]
                )
            retain_loss = retain_loss / gradient_accumulation_steps * tar_retain_scale
            accelerator.backward(retain_loss)
            total_retain_loss += retain_loss.item()

        # Add tamper-resistance gradients to model
        if (
            tamper_resistance_loss >= tar_tamper_resistance_loss_lower_bound
            or unbounded
        ):
            model_storage.add_from_storage_to_model(
                model=model,
                accelerator=accelerator,
                mode="grads",
            )

        # Clear from storage to reduce peak memory usage
        model_storage.clear_grads()
        model_storage.clear_params()

        # Tamper-resistance meta-optimizer step
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if accelerator.is_main_process:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "retain loss / tamper_resistance_loss": f"{total_retain_loss} / {tamper_resistance_loss}"
                }
            )
            wandb.log(
                {
                    "retain_loss": total_retain_loss,
                    "tamper_resistance_loss": tamper_resistance_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    return model
