import argparse
import functools
import os
import random
from typing import Callable

import numpy as np
import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from configs.config import SAVE_MODELS_DIR
from modules.dataloaders import (
    get_tar_dpo_dataloaders,
    get_tar_bio_dataloaders,
    get_tar_cyber_dataloaders,
)
from modules.training import random_mapping_training_loop, tar_training_loop
from modules.utils import fix_seed

ALLOWED_MODULES = [
    LlamaDecoderLayer,
]


def lambda_fn(module: torch.nn.Module):
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False


def finetune_no_trainer(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = None,
    model_type: AutoModelForCausalLM = AutoModelForCausalLM,
    loop_type: Callable = tar_training_loop,
    dataloader_type: Callable = get_tar_bio_dataloaders,
    tokenizer: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    args: argparse.Namespace = None,
):
    # Preparing FSDP (will remove for for FSDP2)
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    model = model_type.from_pretrained(model_name)
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

    # Wandb logging
    if accelerator.is_main_process:
        wandb_mode = "online" if args.wandb else "disabled"
        wandb.login()
        wandb.init(
            project=args.wandb_project_name,
            config=args,
            name="_".join(output_dir.split("/")),
            mode=wandb_mode,
        )
    accelerator.print("Beginning Training.")
    accelerator.free_memory()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model before optimizer: https://huggingface.co/blog/pytorch-fsdp
    model = accelerator.prepare_model(model)
    dataloaders = dataloader_type(tokenizer, accelerator, args=args, model=model)

    model.train()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    )
    optimizer = accelerator.prepare(optimizer)
    accelerator.print(f"model, optimizers, dataloaders prepared")
    accelerator.print(f"output_dir: {output_dir}")

    # Calls either the TAR loop or random vectors loop
    model = loop_type(
        model,
        dataloaders,
        optimizer,
        accelerator,
        **vars(args),
    )
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )


# Map the subject to the dataloader
DATALOADER_MAP = {
    "bio": get_tar_bio_dataloaders,
    "cyber": get_tar_cyber_dataloaders,
    "dpo_anthropic": get_tar_dpo_dataloaders,
}

# Map for training loops
TRAINING_CONFIG = {
    "random_mapping_trainer": random_mapping_training_loop,
    "tar_trainer": tar_training_loop,
}

# Map for model types, can add more here
MODEL_MAP = {
    "llama3": LlamaForCausalLM,
}

# Map for tokenizers, can add more here
TOKENIZER_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_model_name", "-od", type=str, default="tar_model")
    parser.add_argument("--trainer_type", "-tt", type=str, default="tar_trainer")
    parser.add_argument("--max_data_size", "-mds", type=int, default=40000)
    parser.add_argument("--concept_data_split", "-cs", type=float, default=0.2)
    parser.add_argument("--lr", "-lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", "-bs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", "-ga", type=int, default=8)
    parser.add_argument("--max_steps", "-ms", type=int, default=750)
    parser.add_argument("--inner_optimizer_warmup_steps", "-iws", type=int, default=20)
    parser.add_argument("--warmup_steps", "-ws", type=int, default=50)
    parser.add_argument("--expname", "-en", type=str, default="latest")
    parser.add_argument(
        "--base_model_name",
        "-bm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--retain_model_name",
        "-rm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--tar_inner_loop_steps", "-is", type=int, default=1)
    parser.add_argument("--tar_num_tasks_sampled", "-mnts", type=int, default=1)
    parser.add_argument("--retain_representations", "-rr", action="store_true")
    parser.add_argument(
        "--tar_tamper_resistance_loss_lower_bound", "-mlb", type=float, default=-11.76
    )
    parser.add_argument("--use_weighting_schedule", "-uws", action="store_true")
    parser.add_argument("--subject", "-st", type=str, default="bio-multi-dists")
    parser.add_argument("--tar_inner_loop_subsample", "-mils", type=int, default=1)
    parser.add_argument("--tar_adversary_batch_size", "-ilbs", type=int, default=1)
    parser.add_argument("--schedule_lambda", "-sl", type=float, default=0.5)
    parser.add_argument(
        "--tar_tamper_resistance_grad_scale", "-mgs", type=float, default=4.0
    )
    parser.add_argument("--tar_retain_scale", "-mrs", type=float, default=1.0)
    parser.add_argument(
        "--tar_tamper_resistance_loss_type", "-mlt", type=str, default="max_entropy"
    )
    parser.add_argument(
        "--adversary_dist_types",
        "-advs",
        type=str,
        default="pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33",
    )
    parser.add_argument(
        "--switching_point_coeffs", "-spc", type=str, default="alpha:6.0,beta:3.0"
    )
    parser.add_argument(
        "--adversary_lr_schedulers", "-alrs", type=str, default="constant:1.0"
    )
    parser.add_argument(
        "--adversary_lr_samples", "-als", type=str, default="2e-6,2e-5,4e-5"
    )
    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument("--unbounded", "-ub", action="store_true")
    parser.add_argument("--retain_same_base", "-rsb", action="store_true")
    parser.add_argument("--base", "-b", type=str, default="llama")
    parser.add_argument(
        "--wandb_project_name", "-wpn", type=str, default="tar_training"
    )
    args = parser.parse_args()
    fix_seed()
    finetune_no_trainer(
        model_name=args.base_model_name,
        output_dir=os.path.join(
            SAVE_MODELS_DIR, f"{args.new_model_name}_{args.expname}"
        ),
        model_type=MODEL_MAP[args.base],
        loop_type=TRAINING_CONFIG[args.trainer_type],
        dataloader_type=DATALOADER_MAP[args.subject],
        tokenizer=TOKENIZER_MAP[args.base],
        args=args,
    )


if __name__ == "__main__":
    main()
