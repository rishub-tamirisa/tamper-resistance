import os
import torch
import argparse
import random
import numpy as np
from config import SAVE_MODELS_DIR #FIXME: Does this still work?
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
import wandb

import schedulefree

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from dataloader import get_bio_combined_dataloaders, get_chem_combined_dataloaders, get_wmdp_cyber_dataloaders, get_chem_dataloaders, get_cyber_dataloaders #FIXME: Does this still work?
from training import random_vectors_training_loop, fast_single_dataloader_finetune_loop as repair_loop, log_1_minus_p_training_loop, llmu_training_loop, max_entropy_training_loop #FIXME: Does this still work?
from utils import update_optim_lr

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM
#from mistral_common.tokens.tokenizers.mistral import MistralTokenizer #pip install mistral_common
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer, Gemma2ForCausalLM

import mmlu_eval.eval as eval #FIXME: Does this still work?

import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from job_config import get_config

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

def fix_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def random_mapping(model, base_model, r_d, f_d, optimizer, accelerator, num_epochs, gradient_accumulation_steps, args):
    model = random_vectors_training_loop(
        model, 
        base_model,
        r_d,
        f_d, 
        optimizer, 
        accelerator, 
        num_epochs, 
        gradient_accumulation_steps, 
        max_steps=args.max_steps, 
        args=args,
    )
    return model


def min_posterior(model, base_model, r_d, f_d, optimizer, accelerator, num_epochs, gradient_accumulation_steps, args):
    model = log_1_minus_p_training_loop(
        model,
        r_d,
        f_d,
        optimizer,
        accelerator,
        num_epochs,
        gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    return model

def max_entropy(model, base_model, r_d, f_d, optimizer, accelerator, num_epochs, gradient_accumulation_steps, args):
    model = max_entropy_training_loop(
        model,
        r_d,
        f_d,
        optimizer,
        accelerator,
        num_epochs,
        gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    return model

def llmu(model, base_model, r_d, f_d, optimizer, accelerator, num_epochs, gradient_accumulation_steps, args):
    model = llmu_training_loop(
        model,
        r_d,
        f_d,
        optimizer,
        accelerator,
        num_epochs,
        gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    return model

def baseline(
    model_name: str,
    model_type: str,
    output_dir: str,
    loop_type=random_mapping,
    dataloader_type=get_bio_combined_dataloaders,
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
        model = AutoModelForCausalLM.from_pretrained(model_name)#, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

    with accelerator.main_process_first():
        retain, forget_train, forget_test = dataloader_type(tokenizer=tokenizer, accelerator=accelerator, args=args)
    dataloaders = [retain, forget_train]

    accelerator.free_memory()
    model = accelerator.prepare_model(model)
    accelerator.print(f"Model prepared.")
    accelerator.print(f"Output dir: {output_dir}")

    if args.use_rvp_with_mse_retain:
        accelerator.print("Using RVP with MSE Retain!")
        base_model = AutoModelForCausalLM.from_pretrained(model_type)
        base_model = accelerator.prepare_model(base_model)
        base_model.eval()
        accelerator.print("Frozen Base Model prepared.")
    else:
        base_model = None

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.learning_rate, warmup_steps=args.warmup_steps)

    num_epochs = args.num_epochs

    optimizer, *dataloaders = accelerator.prepare(optimizer, *dataloaders)

    accelerator.print(f"Optimizer and Dataloaders prepared.")

    model = loop_type(
        model,
        base_model,
        *dataloaders,
        optimizer,
        accelerator,
        num_epochs,
        gradient_accumulation_steps,
        args,
    )

    if args.evaluate:
        accelerator.print("Evaluation mode enabled. WARNING: Model will not be saved.")
        accelerator.print("Evaluating model.")
        use_eos_token = True if args.model_type == "meta-llama/Meta-Llama-3-8B-Instruct" or args.model_type == "Qwen/Qwen2-7B-Instruct" else False
        user = os.environ.get("USER")
        eval_args = type('Args', (object,), 
        {
            'batch_size': 2, 
            "num_fewshot_examples": 5, 
            'max_seq_len': 4096, 
            'path_to_data': f"/data/{user}/capabilities-removal/batched_evaluation/data",
            'disable_file_writes': True,
            'eos_pad_token': use_eos_token,
            'save_file_dir': args.save_model_name,
        })
        eval.evaluate_model(model, tokenizer, accelerator, eval_args)
        accelerator.print("Evaluation complete.")


    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.print(f"Model saved to {output_dir}.")

TRAINING_CONFIG = {
    "bio_rvp": {
        "loop_type": random_mapping,
        "dataloader_type": get_bio_combined_dataloaders,
    },
    "chem_rvp": {
        "loop_type": random_mapping,
        "dataloader_type": get_chem_dataloaders,
    },
    "cyber_rvp": {
        "loop_type": random_mapping,
        "dataloader_type": get_cyber_dataloaders,
    },

    "bio_min_posterior": {
        "loop_type": min_posterior,
        "dataloader_type": get_bio_combined_dataloaders,
    },
    "chem_min_posterior": {
        "loop_type": min_posterior,
        "dataloader_type": get_chem_dataloaders,
    },
    "cyber_min_posterior": {
        "loop_type": min_posterior,
        "dataloader_type": get_cyber_dataloaders,
    },

    "bio_max_entropy": {
        "loop_type": max_entropy,
        "dataloader_type": get_bio_combined_dataloaders,
    },
    "chem_max_entropy": {
        "loop_type": max_entropy,
        "dataloader_type": get_chem_dataloaders,
    },
    "cyber_max_entropy": {
        "loop_type": max_entropy,
        "dataloader_type": get_cyber_dataloaders,
    },

    "bio_llmu": {
        "loop_type": llmu,
        "dataloader_type": get_bio_combined_dataloaders,
    },
    "chem_llmu": {
        "loop_type": llmu,
        "dataloader_type": get_chem_dataloaders,
    },
    "cyber_llmu": {
        "loop_type": llmu,
        "dataloader_type": get_cyber_dataloaders,
    },
}

def main():
    #"meta-llama/Meta-Llama-3-8B"
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct" #mistralai/Mistral-7B-Instruct-v0.2
    )
    parser.add_argument(
        "--save_model_name", "-smn", type=str, default="max_ent_test"
    )
    parser.add_argument(
        "--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )

    parser.add_argument("--training_config", "-s", type=str, default="bio_min_posterior")

    parser.add_argument("--batch_size", "-bs", type=int, default=8)

    parser.add_argument("--num_epochs", "-ne", type=int, default=1)
    parser.add_argument("--num_impair_steps", "-nis", type=int, default=400)
    parser.add_argument("--num_repair_steps", "-nrs", type=int, default=600)
    parser.add_argument("--warmup_steps", "-ws", type=int, default=100)
    parser.add_argument("--max_steps", "-ms", type=int, default=1000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5) #5e-6 for good result
    parser.add_argument("--repair_lr", "-rlr", type=float, default=2e-6)

    parser.add_argument("--use_rvp_with_mse_retain", "-rvp_mse", action="store_true")


    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument("--evaluate", "-e", action="store_true")
    parser.add_argument("--job_config_string", "-jci", type=str, default="test_unlearning_random_mapping")

    args = parser.parse_args()
    fix_seed()
    is_slurm = os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    if is_slurm:
        print("Running on SLURM ARRAY")
        set_args = get_config(args.job_config_string)
        for key, value in set_args.items():
            setattr(args, key, value)
    baseline(
        model_name=args.model_name,
        model_type=args.model_type,
        output_dir=os.path.join(SAVE_MODELS_DIR, args.save_model_name),
        loop_type=TRAINING_CONFIG[args.training_config]["loop_type"],
        dataloader_type=TRAINING_CONFIG[args.training_config]["dataloader_type"],
        args=args,
    )

if __name__ == "__main__":
    main()

#  accelerate launch --config_file $FxACCEL_CONFIG relearning_evaluation.py --model_name /scratch/bcar/models/mlac_trainer_llama2v2 --model_type meta-llama/Llama-2-7b-chat-hf --batch_size 4 --subject cell_mol_bio --training_strategy retain_strategy
