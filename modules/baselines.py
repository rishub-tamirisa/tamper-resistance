import os
import torch
import argparse
import random
import numpy as np
from configs.config import SAVE_MODELS_DIR
import datasets

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
import wandb

import schedulefree

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from dataloaders import get_tar_bio_dataloaders, get_tar_cyber_dataloaders
from training import (
    random_mapping_training_loop,
    llmu_training_loop,
    max_entropy_training_loop,
    min_posterior_training_loop,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
)

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer #pip install mistral_common
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
)

import red_teaming.mmlu_eval.eval as eval
from utils import fix_seed

import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

GRAD_ACCUM_STEPS = 1

os.environ["WANDB_DISABLED"] = "false"

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    PhiDecoderLayer,
    MistralDecoderLayer,
    Qwen2DecoderLayer,
    Gemma2DecoderLayer,
]


def lambda_fn(module: torch.nn.Module):
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False


auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

FSDP_PLUGIN = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy,
)

def baseline(
    model_name: str,
    model_type: str,
    output_dir: str,
    loop_type=random_mapping_training_loop,
    dataloader_type=get_tar_bio_dataloaders,
    args=None,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS, fsdp_plugin=FSDP_PLUGIN
    )

    if accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="baselines",
            config=args,
            name="_".join(output_dir.split("/")),
            mode="online",
        )

    gradient_accumulation_steps = GRAD_ACCUM_STEPS

    if model_name == "random_llama":
        config = LlamaConfig()
        model = LlamaForCausalLM(config)
        model_type = "meta-llama/Meta-Llama-3-8B"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name
        )  # , attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

    with accelerator.main_process_first():
        if (
            dataloader_type == get_tar_bio_dataloaders
            or dataloader_type == get_tar_cyber_dataloaders
        ):
            all_dataloaders = dataloader_type(
                tokenizer=tokenizer, accelerator=accelerator, args=args
            )
        else:
            retain, forget_train, forget_test = dataloader_type(
                tokenizer=tokenizer, accelerator=accelerator, args=args
            )

    if (
        dataloader_type == get_tar_bio_dataloaders
        or dataloader_type == get_tar_cyber_dataloaders
    ):
        forget_train = all_dataloaders[
            DATALOADER_MAP[args.dataloader_type]["forget_train_split_key"]
        ]
        dataloaders = [
            all_dataloaders["pile-retain"],
            forget_train,
            all_dataloaders["meta"],
        ]
    else:
        dataloaders = [retain, forget_train, forget_test]

    accelerator.free_memory()
    model = accelerator.prepare_model(model)
    accelerator.print(f"Model prepared.")
    accelerator.print(f"Output dir: {output_dir}")

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.learning_rate, warmup_steps=args.warmup_steps
    )

    num_epochs = args.num_epochs

    optimizer, *dataloaders = accelerator.prepare(optimizer, *dataloaders)

    accelerator.print(f"Optimizer and Dataloaders prepared.")

    model = loop_type(
        model,
        *dataloaders,
        optimizer,
        accelerator,
        num_epochs,
        gradient_accumulation_steps,
        args,
    )

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
                "path_to_data": f"/data/{user}/capabilities-removal/batched_evaluation/data",
                "disable_file_writes": True,
                "eos_pad_token": use_eos_token,
                "save_file_dir": args.save_model_name,
            },
        )
        eval.evaluate_model(model, tokenizer, accelerator, eval_args)
        accelerator.print("Evaluation complete.")

    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.print(f"Model saved to {output_dir}.")


DATALOADER_MAP = {
    "bio": {
        "dataloader_name": get_tar_bio_dataloaders,
        "forget_train_split_key": "bio-combined",
    },
    "cyber": {
        "dataloader_name": get_tar_cyber_dataloaders,
        "forget_train_split_key": "forget_train",
    },
}

BASELINE_MAP = {
    "random_mapping": random_mapping_training_loop,
    "min_posterior": min_posterior_training_loop,
    "max_entropy": max_entropy_training_loop,
    "llmu": llmu_training_loop,
}


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--save_model_name", "-smn", type=str, default="max_ent_test")
    parser.add_argument(
        "--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )

    parser.add_argument("--baseline_type", "-bt", type=str, default="random_mapping")
    parser.add_argument("--dataloader_type", "-dt", type=str, default="bio")

    parser.add_argument("--batch_size", "-bs", type=int, default=8)

    parser.add_argument("--num_epochs", "-ne", type=int, default=1)

    parser.add_argument("--warmup_steps", "-ws", type=int, default=100)
    parser.add_argument("--max_steps", "-ms", type=int, default=1000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)

    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument("--evaluate", "-e", action="store_true")

    args = parser.parse_args()
    fix_seed()
    baseline(
        model_name=args.model_name,
        model_type=args.model_type,
        output_dir=os.path.join(SAVE_MODELS_DIR, args.save_model_name),
        loop_type=BASELINE_MAP[args.baseline_type],
        dataloader_type=DATALOADER_MAP[args.dataloader_type]["dataloader_name"],
        args=args,
    )


if __name__ == "__main__":
    main()
