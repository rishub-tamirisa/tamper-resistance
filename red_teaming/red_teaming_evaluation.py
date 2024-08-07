import os
import torch
import argparse
import random
import numpy as np
from ..configs.config import SAVE_MODELS_DIR  # FIXME: Ensure import path is correct
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
)  # FIXME: Ensure import path is correct
from ..modules.training import (
    single_dataloader_accel_finetune_loop,
    double_dataloader_accel_finetune_loop,
)  # FIXME: Ensure import path is correct
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

os.environ["WANDB_DISABLED"] = "false"

ALLOWED_MODULES = [
    LlamaDecoderLayer,
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


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

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

    if model_name == "random_llama":
        config = LlamaConfig()
        model = LlamaForCausalLM(config)
        model_type = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

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
        forget_train = all_dataloaders[MULTI_DIST_MAP[args.training_strategy]]
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

    num_epochs = args.num_epochs
    warmup_steps = args.num_warmup_steps

    optimizer = optimizer_type(model, args.learning_rate, warmup_steps=warmup_steps)

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


MULTI_DIST_MAP = {
    "pure_pile_bio_forget": "pile-bio",
    "pure_pile_bio_retain": "pile-bio",
    "pile_bio_retain_followed_by_pile_bio_forget": "pile-bio",
    "cyber_and_pile_forget": "forget_train",
    "cyber_and_pile_retain": "forget_train",
    "cyber_retain_followed_by_forget": "forget_train",
}

TRAINING_CONFIG = {
    # Biosecurity
    "pure_pile_bio_forget": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "forget",
    },
    "pure_pile_bio_retain": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "retain",
    },
    "pile_bio_retain_followed_by_pile_bio_forget": {
        "loop_type": double_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_bio_dataloaders,
        "finetuning_data_type": "retain",
    },
    # NOTE: The Biosecurity OOD-Forget Dataset can be requested at https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform
    # NOTE: The Chemical Security dataset is private.
    # Cybersecurity
    "cyber_and_pile_forget": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "forget",
    },
    "cyber_and_pile_retain": {
        "loop_type": single_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "retain",
    },
    "cyber_retain_followed_by_forget": {
        "loop_type": double_dataloader_accel_finetune_loop,
        "dataloader_type": get_red_team_tar_cyber_dataloaders,
        "finetuning_data_type": "retain",
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
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--save_model_name", "-smn", type=str, default="saved_model")
    parser.add_argument(
        "--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
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
    )  # NOTE: For double_dataloader_accel_finetune_loop
    parser.add_argument("--r->f_prop_steps_of_retain", "-psor", type=float, default=0.4)

    parser.add_argument("--peft", "-pft", action="store_true")
    parser.add_argument("--wandb", "-wb", action="store_true")
    parser.add_argument(
        "--evaluate_mmlu", "-mmlu", action="store_true"
    )  # Evaluates the model when the relearning evaluation concludes and does not save the model.
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    fix_seed(args.seed)

    sft_red_teaming_evaluation(
        model_name=args.model_name,
        model_type=args.model_type,
        output_dir=os.path.join(SAVE_MODELS_DIR, args.save_model_name),
        loop_type=TRAINING_CONFIG[args.training_strategy]["loop_type"],
        dataloader_type=TRAINING_CONFIG[args.training_strategy]["dataloader_type"],
        finetuning_data_type=TRAINING_CONFIG[args.training_strategy][
            "finetuning_data_type"
        ],
        optimizer_type=OPTIMIZER_CONFIG[args.optimizer_type],
        args=args,
    )


if __name__ == "__main__":
    main()
