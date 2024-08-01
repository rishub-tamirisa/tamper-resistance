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
    PhiForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from configs.config import SAVE_MODELS_DIR
from modules.dataloaders import (
    get_anthropic_hh_dpo_dataloaders,
    get_bio_multi_dists_dataloaders,
    get_chem_dataloaders,
    get_cyber_dataloaders,
)
from modules.training import random_vectors_training_loop, tar_training_loop

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
    dataset_path: str = None,
    output_dir: str = None,
    model_type: AutoModelForCausalLM = AutoModelForCausalLM,
    loop_type: Callable = tar_training_loop,
    dataloader_type: Callable = get_bio_multi_dists_dataloaders,
    tokenizer: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    args: argparse.Namespace = None,
):
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    model = model_type.from_pretrained(model_name)
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

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
    dataloaders = dataloader_type(
        tokenizer, accelerator, path=dataset_path, args=args, model=model
    )

    model.train()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    )
    optimizer = accelerator.prepare(optimizer)
    accelerator.print(f"model, optimizers, dataloaders prepared")
    accelerator.print(f"output_dir: {output_dir}")
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


def fix_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


DATALOADER_MAP = {
    "bio-multi-dists": {
        "path": "/data/private_models/cais_models/robust_unlearning/data/camel_ai/biology.csv",
        "dataloader_type": get_bio_multi_dists_dataloaders,
    },
    "chemistry-latest": {
        "path": "",
        "dataloader_type": get_chem_dataloaders,
    },
    "cyber-latest": {
        "path": "",
        "dataloader_type": get_cyber_dataloaders,
    },
    "dpo_anthropic": {
        "path": "",
        "dataloader_type": get_anthropic_hh_dpo_dataloaders,
    },
}


TRAINING_CONFIG = {
    "random_vectors_trainer": {
        "loop_type": random_vectors_training_loop,
    },
    "tar_trainer": {
        "loop_type": tar_training_loop,
    },
}


MODEL_MAP = {
    "llama": LlamaForCausalLM,
    "llama3": LlamaForCausalLM,
    "phi": PhiForCausalLM,
    "cohere": AutoModelForCausalLM,
    "phi3-mini": AutoModelForCausalLM,
    "qwen": Qwen2ForCausalLM,
}

TOKENIZER_MAP = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "llama-base": "meta-llama/Llama-2-7b-hf",
    "gated_llama": "meta-llama/Llama-2-7b-chat-hf",
    "llama_with_full_stream": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-base": "meta-llama/Meta-Llama-3-8B",
    "mistral": "HuggingFaceH4/zephyr-7b-beta",
    "cohere": "CohereForAI/c4ai-command-r-plus",
    "phi3-mini": "microsoft/Phi-3-mini-128k-instruct",
    "qwen": "Qwen/Qwen2-7B-Instruct",
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
        "--tar_meta_loss_lower_bound", "-mlb", type=float, default=-15.0
    )
    parser.add_argument("--tar_inner_loop_subsample", "-mils", type=int, default=1)
    parser.add_argument("--tar_inner_loop_batch_size", "-ilbs", type=int, default=1)
    parser.add_argument("--schedule_lambda", "-sl", type=float, default=0.5)
    parser.add_argument("--tar_meta_grad_scale", "-mgs", type=float, default=0.25)
    parser.add_argument("--tar_retain_scale", "-mrs", type=float, default=1.0)
    parser.add_argument("--tar_meta_loss_type", "-mlt", type=str, default="max_entropy")
    parser.add_argument(
        "--adversary_dist_types",
        "-advs",
        type=str,
        default="pile-bio:0.33,camel-bio:0.33,fineweb:0.33",
    )
    parser.add_argument(
        "--switching_point_coeffs", "-spc", type=str, default="alpha:1.0,beta:1.0"
    )
    parser.add_argument(
        "--adversary_lr_schedulers", "-alrs", type=str, default="constant:1.0"
    )
    parser.add_argument(
        "--adversary_lr_samples", "-als", type=str, default="2e-5,4e-5,1e-4"
    )
    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument("--unbounded", "-ub", action="store_true")
    parser.add_argument("--retain_same_base", "-rsb", action="store_true")
    parser.add_argument("--tar_random_subsample", "-mrss", action="store_true")
    parser.add_argument("--base", "-b", type=str, default="llama")
    parser.add_argument(
        "--wandb_project_name", "-wpn", type=str, default="tar_training"
    )
    args = parser.parse_args()
    fix_seed()
    finetune_no_trainer(
        model_name=args.base_model_name,
        dataset_path=DATALOADER_MAP[args.subject]["path"],
        output_dir=os.path.join(
            SAVE_MODELS_DIR, f"{args.new_model_name}_{args.expname}"
        ),
        model_type=MODEL_MAP[args.base],
        loop_type=TRAINING_CONFIG[args.trainer_type]["loop_type"],
        dataloader_type=DATALOADER_MAP[args.subject]["dataloader_type"],
        tokenizer=TOKENIZER_MAP[args.base],
        args=args,
    )


if __name__ == "__main__":
    main()
