import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from tqdm import tqdm
from modules.objectives import (
    critic_obj,
    distribution_matching_obj,
)
from probes import LlamaForCausalLMWithWassersteinCriticProbes
from modules.utils import (
    get_next_batch,
)
from extra.utils import (
    accelerated_optimizer_to_cpu,
    accelerated_optimizer_to_gpu,
)
from typing import Iterator
import wandb


def probe_critic_training_loop(
    model: LlamaForCausalLMWithWassersteinCriticProbes,
    retain_iterator: Iterator,
    retain_dataloader: torch.utils.data.DataLoader,
    forget_iterator: Iterator,
    forget_dataloader: torch.utils.data.DataLoader,
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    gradient_accumulation_steps: int,
    epochs: int = 1,
    probe_steps: int = 5,
):
    """
    Train the Lipschitz-constrained critic functions used for Wasserstein GAN based on MLP probes.

    This function trains the critic to produce a scalar score difference between two batches of sampled activations
    from the retain and forget datasets. The critic is implemented as MLP probes attached to the model's layers.

    Args:
        model (LlamaForCausalLMWithWassersteinCriticProbes): The model with attached MLP probes.
        retain_iterator (Iterator): Iterator for the retain dataset.
        retain_dataloader (torch.utils.data.DataLoader): DataLoader for the retain dataset.
        forget_iterator (Iterator): Iterator for the forget dataset.
        forget_dataloader (torch.utils.data.DataLoader): DataLoader for the forget dataset.
        optimizer (AcceleratedOptimizer): Optimizer for the probe parameters.
        accelerator (Accelerator): Accelerator for distributed training.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        probe_steps (int, optional): Number of training steps per epoch. Defaults to 5.

    Returns:
        None: The function updates the model's probe parameters in-place.
    """
    model.set_probe_requires_grad(True)
    model.set_non_probe_requires_grad(False)
    for epoch in range(epochs):
        if accelerator.is_main_process:
            probe_pbar = tqdm(
                colour="blue",
                desc=f"probe training, epoch: {epoch+1}",
                total=probe_steps,
                dynamic_ncols=True,
            )
        for _ in range(probe_steps):
            total_batch_loss = 0
            for _ in range(gradient_accumulation_steps):
                retain_batch, retain_iterator = get_next_batch(
                    retain_iterator, retain_dataloader
                )
                forget_batch, forget_iterator = get_next_batch(
                    forget_iterator, forget_dataloader
                )
                batch_loss = critic_obj(
                    model=model,
                    x_r=retain_batch,
                    x_f=forget_batch,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                total_batch_loss += batch_loss
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                probe_pbar.update(1)
                probe_pbar.set_postfix({"probe loss": total_batch_loss})
                wandb.log({"probe_loss": total_batch_loss})
        accelerator.wait_for_everyone()
    model.set_probe_requires_grad(False)
    model.set_non_probe_requires_grad(True)


def residual_stream_gan_training_loop(
    model: LlamaForCausalLMWithWassersteinCriticProbes,
    retain_dataloader: torch.utils.data.DataLoader,
    retain_subset_dataloader: torch.utils.data.DataLoader,
    forget_dataloader: torch.utils.data.DataLoader,
    optimizer: AcceleratedOptimizer,
    probe_optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 1000,
    probe_learn_freq: int = 1,
    probe_steps: int = 5,
    probe_headstart: int = 0,
    **kwargs,
):
    """
    Performs a GAN-style training loop on the residual stream of a language model.

    This function trains the model to push activations from the forget set to resemble
    the distribution of a subset of the retain set (the "match" distribution), while
    regularizing via a standard cross-entropy loss on the full retain set. It alternates
    between training the critic (probe) and the main model.

    Args:
        model (LlamaForCausalLMWithWassersteinCriticProbes): The model to be trained.
        retain_dataloader (torch.utils.data.DataLoader): DataLoader for the full retain set.
        retain_subset_dataloader (torch.utils.data.DataLoader): DataLoader for the subset of retain data to match.
        forget_dataloader (torch.utils.data.DataLoader): DataLoader for the forget set.
        optimizer (AcceleratedOptimizer): Optimizer for the main model parameters.
        probe_optimizer (AcceleratedOptimizer): Optimizer for the probe (critic) parameters.
        accelerator (Accelerator): Accelerator for distributed training.
        num_epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        max_steps (int): Maximum number of training steps per epoch.
        probe_learn_freq (int): Frequency of probe training iterations.
        probe_steps (int): Number of steps for each probe training iteration.
        probe_headstart (int): Number of initial steps to train only the probe.
        **kwargs: Additional keyword arguments.

    Returns:
        LlamaForCausalLMWithWassersteinCriticProbes: The trained model.
    """
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)
    retain_iterator = iter(retain_dataloader)
    retain_subset_iterator = iter(retain_subset_dataloader)
    forget_iterator = iter(forget_dataloader)
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(
                colour="blue",
                desc=f"Main Training Epoch: {epoch+1}",
                total=max_steps,
                dynamic_ncols=True,
            )
        for i in range(max_steps):
            if i % probe_learn_freq == 0:
                accelerated_optimizer_to_cpu(optimizer)
                accelerated_optimizer_to_gpu(probe_optimizer)
                probe_critic_training_loop(
                    model=model,
                    retain_iterator=retain_subset_iterator,
                    retain_dataloader=retain_subset_dataloader,
                    forget_iterator=forget_iterator,
                    forget_dataloader=forget_dataloader,
                    optimizer=probe_optimizer,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    epochs=1,
                    probe_steps=probe_steps,
                )
                accelerated_optimizer_to_cpu(probe_optimizer)
                accelerated_optimizer_to_gpu(optimizer)

            # NOTE: do not do gradient synchronization here! (i.e. with accelerator.accumulate)
            total_batch_loss = 0
            if i >= probe_headstart:
                for _ in range(gradient_accumulation_steps):
                    retain_batch, retain_iterator = get_next_batch(
                        retain_iterator, retain_dataloader
                    )
                    retain_subset_batch, retain_subset_iterator = get_next_batch(
                        retain_subset_iterator, retain_subset_dataloader
                    )
                    forget_batch, forget_iterator = get_next_batch(
                        forget_iterator, forget_dataloader
                    )

                    batch_loss = distribution_matching_obj(
                        model=model,
                        x_r=retain_batch,
                        x_r_match=retain_subset_batch,
                        x_f=forget_batch,
                        accelerator=accelerator,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )
                    total_batch_loss += batch_loss

                optimizer.step()
                model.zero_grad(set_to_none=True)
                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix({"loss": total_batch_loss})
                    wandb.log({"main_loss": total_batch_loss})
    return model