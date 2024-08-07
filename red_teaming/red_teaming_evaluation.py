import os
import torch
import argparse
import random
import numpy as np
from ..configs.config import SAVE_MODELS_DIR
import wandb

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from ..modules.dataloaders import (
    get_red_team_tar_bio_dataloaders,
    get_red_team_tar_cyber_dataloaders,
)
from ..modules.training import (
    single_dataloader_accel_finetune_loop,
    double_dataloader_accel_finetune_loop,
)
from schedulers import (
    get_exponential_warmup_scheduler,
    get_sgdr_scheduler,
    get_no_scheduler,
    get_linear_warmup_scheduler,
    get_warmup_with_annealing_scheduler,
)
from optimizers import (
    get_sgd_with_momentum,
    get_sgd_with_nesterov_momentum,
    get_adam,
    get_adamW,
    get_adagrad,
    get_adadelta,
    get_adamW_schedule_free,
)
import mmlu_eval.eval as eval
from ..modules.utils import return_step_based_batch_selection

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from peft import LoraConfig, TaskType, get_peft_model

from torch import distributed as dist
from ..modules.utils import fix_seed

# Disable Weights & Biases logging if needed
os.environ["WANDB_DISABLED"] = "false"

# Define allowed modules for FSDP wrapping
ALLOWED_MODULES = [
    LlamaDecoderLayer,
]

# Function to determine if a module should be wrapped
def lambda_fn(module: torch.nn.Module):
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False

# Create auto wrap policy for FSDP
auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

# Configure FSDP plugin
FSDP_PLUGIN = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy,
)

def sft_red_teaming_evaluation(
    model_name: str,
    model_type: str,
    output_dir: str,
    loop_type=single_dataloader_accel_finetune_loop,
    dataloader_type=get_red_team_tar_bio_dataloaders,
    finetuning_data_type="forget",
    optimizer_type=get_adamW,
    args=None,
):
    """
    Main function for SFT (Supervised Fine-Tuning) Red Teaming Evaluation.

    Args:
        model_name (str): Name of the model to be fine-tuned.
        model_type (str): Type of the model (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
        output_dir (str): Directory to save the fine-tuned model.
        loop_type (function): Training loop function to use.
        dataloader_type (function): Function to get dataloaders.
        finetuning_data_type (str): Type of fine-tuning data ("forget" or "retain").
        optimizer_type (function): Function to get the optimizer.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None. Saves the fine-tuned model to the specified output directory.
    """
    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

    # Initialize Weights & Biases logging for the main process
    if accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="relearning_evaluation",
            config=args,
            name="_".join(output_dir.split("/")),
            mode="online",
        )

    accelerator.print("Starting relearning evaluation on model: ", model_name)

    gradient_accumulation_steps = args.gradient_accumulation_steps

    # Load model and tokenizer
    if model_name == "random_llama":
        config = LlamaConfig()
        model = LlamaForCausalLM(config)
        model_type = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure PEFT (Parameter Efficient Fine-Tuning) if enabled
    if args.peft:
        accelerator.print("Parameter Efficient Fine-Tuning (PEFT) enabled")
        lora_config = LoraConfig(
            r=16,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare dataloaders
    with accelerator.main_process_first():
        if (
            dataloader_type == get_red_team_tar_bio_dataloaders
            or dataloader_type == get_red_team_tar_cyber_dataloaders
        ):
            all_dataloaders = dataloader_type(
                tokenizer=tokenizer, accelerator=accelerator, args=args
            )
        else:
            retain, forget_train, forget_test = dataloader_type(
                tokenizer=tokenizer, accelerator=accelerator, args=args
            )

    if (
        dataloader_type == get_red_team_tar_bio_dataloaders
        or dataloader_type == get_red_team_tar_cyber_dataloaders
    ):
        forget_train = all_dataloaders[TRAINING_CONFIG[args.training_strategy]["multi_dist_key_name"]]
        dataloaders = [
            all_dataloaders["pile-retain"],
            forget_train,
            all_dataloaders["meta"],
        ]
    else:
        dataloaders = [retain, forget_train, forget_test]

    # Prepare model, optimizer, and scheduler
    accelerator.free_memory()
    model = accelerator.prepare_model(model)
    accelerator.print(f"Model prepared.")
    accelerator.print(f"Output dir: {output_dir}")

    num_epochs = args.num_epochs
    warmup_steps = args.num_warmup_steps

    optimizer = optimizer_type(model, args.learning_rate, warmup_steps=warmup_steps)

    # Select scheduler based on args
    if args.scheduler_type == "sgdr":
        scheduler = get_sgdr_scheduler(optimizer)
    elif args.scheduler_type == "warmup_with_annealing":
        scheduler = get_warmup_with_annealing_scheduler(
            optimizer=optimizer, warmup_steps=warmup_steps, max_steps=args.max_steps
        )
    elif args.scheduler_type == "linear":
        scheduler = get_linear_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            initial_lr=0,
            final_lr=args.learning_rate,
        )
    else:
        scheduler = get_no_scheduler(optimizer)
        accelerator.print("No scheduler provided. Continuing with no scheduling.")

    optimizer, scheduler, *dataloaders = accelerator.prepare(
        optimizer, scheduler, *dataloaders
    )

    accelerator.print(f"Optimizer, Scheduler, and Dataloaders prepared.")

    # Run the training loop
    model = loop_type(
        model,
        tokenizer,
        *dataloaders,
        optimizer,
        scheduler,
        accelerator,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=args.max_steps,
        scheduler_type=args.scheduler_type,
        finetuning_data_type=finetuning_data_type,
        batch_selection_method=args.batch_selection_method,
        prop_steps_for_batch_selection=args.prop_steps_for_batch_selection,
        optimizer_type=args.optimizer_type,
        model_type=args.model_type,
    )

    accelerator.wait_for_everyone()

    # Evaluate the model if specified
    if args.evaluate:
        accelerator.print("Evaluation mode enabled.")
        accelerator.print("Evaluating model.")
        use_eos_token = (
            True
            if args.model_type == "meta-llama/Meta-Llama-3-8B-Instruct"
            or args.model_type == "Qwen/Qwen2-7B-Instruct"
            else False
        )
        user = os.environ.get("USER")
        eval_args = type(
            "Args",
            (object,),
            {
                "batch_size": 2,
                "num_fewshot_examples": 5,
                "max_seq_len": 4096,
                "path_to_data": "mmlu_eval/data",
                "disable_file_writes": True,
                "eos_pad_token": use_eos_token,
                "save_file_dir": args.save_model_name,
            },
        )
        eval.evaluate_model(model, tokenizer, accelerator, eval_args)

    accelerator.print("Evaluation complete.")

    # Save the fine-tuned model
    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    accelerator.print(f"Model saved to {output_dir}.")

# Configuration dictionaries for training strategies and optimizers
TRAINING_CONFIG = {
    # Biosecurity
    "pure_pile_bio_forget": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "forget",
        "multi_dist_key_name": "pile-bio",
    },
    "pure_pile_bio_retain": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "retain",
        "multi_dist_key_name": "pile-bio",
    },
    "pile_bio_retain_followed_by_pile_bio_forget": {
        "loop_type": double_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "retain",
        "multi_dist_key_name": "pile-bio",
    },
    # Cybersecurity
    "cyber_and_pile_forget": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "forget",
        "multi_dist_key_name": "forget_train",
    },
    "cyber_and_pile_retain": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "retain",
        "multi_dist_key_name": "forget_train",
    },
    "cyber_retain_followed_by_forget": {
        "loop_type": double_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "retain",
        "multi_dist_key_name": "forget_train",
    },
}

OPTIMIZER_CONFIG = {
    "sgd_with_momentum": get_sgd_with_momentum,
    "sgd_with_nesterov_momentum": get_sgd_with_nesterov_momentum,
    "adam": get_adam,
    "adamW": get_adamW,
    "adagrad": get_adagrad,
    "adadelta": get_adadelta,
    "adamW_schedule_free": get_adamW_schedule_free,
}

def main():
    """
    Main function to parse command-line arguments and run the SFT red teaming evaluation.
    """
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--save_model_name", "-smn", type=str, default="saved_model")
    parser.add_argument("--scheduler_type", "-st", type=str, default="none")
    parser.add_argument("--num_warmup_steps", "-nws", type=int, default=0)
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", "-gas", type=int, default=2)
    parser.add_argument("--optimizer_type", "-opt", type=str, default="adamW")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", "-ne", type=int, default=1)
    parser.add_argument("--max_steps", "-ms", type=int, default=1000)
    parser.add_argument(
        "--training_strategy", "-ts", type=str, default="pure_pile_bio_forget"
    )

    parser.add_argument(
        "--r->f_batch_selection_method",
        "-bsm",
        type=callable,
        default=return_step_based_batch_selection,
    ) 
    parser.add_argument("--r->f_prop_steps_of_retain", "-psor", type=float, default=0.4)

    parser.add_argument("--peft", "-pft", action="store_true")
    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument(
        "--evaluate_mmlu", "-mmlu", action="store_true"
    )
    parser.add_argument("--seed", "-s", type=int, default=42)