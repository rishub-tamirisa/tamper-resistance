import torch
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import custom modules
from .categories import subjects
from .eval_dataloader import get_eval_dataloaders
from .add_headers_to_csv import check_dir_has_headers, add_header_to_csv
from .question_hash_table import QuestionHashTable, update_serialized_table

# Import Accelerate and model-specific modules
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
)
import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    all_gather_into_tensor,
)
from ...modules.utils import fix_seed

# Define allowed modules for FSDP wrapping
ALLOWED_MODULES = [
    LlamaDecoderLayer,
    PhiDecoderLayer,
    MistralDecoderLayer,
]

# Define lambda function for FSDP auto wrap policy
def lambda_fn(module: torch.nn.Module):
    return any(isinstance(module, allowed_module) for allowed_module in ALLOWED_MODULES)

auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

# Configure FSDP plugin
FSDP_PLUGIN = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy,
)

@torch.no_grad()
def evaluate_subject(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    **kwargs,
):
    """
    Evaluate the model on a specific subject.
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer for the model
        accelerator: Accelerator for distributed training
        **kwargs: Additional arguments
    
    Returns:
        acc: Accuracy for the subject
        final_table: Table with evaluation results
    """
    model.config.use_cache = False
    model.eval()
    serialized_table = []
    highest_batch_idx = 0

    for idx, batch in enumerate(eval_dataloader):
        # Process batch and get model predictions
        batch_squeezed = {key: value.squeeze(1) for key, value in batch.items()}
        labels = batch_squeezed["labels"][:, -1]
        labels = [tokenizer.decode(label.item()) for label in labels]

        model_outputs = model(
            input_ids=batch_squeezed["input_ids"],
            attention_mask=batch_squeezed["attention_mask"],
        )
        logits = model_outputs.logits[:, -1]

        # Calculate probabilities for each answer option
        A_logits, B_logits, C_logits, D_logits = [logits[:, tokenizer(option).input_ids[-1]] for option in "ABCD"]
        probs = torch.nn.functional.softmax(
            torch.stack([A_logits, B_logits, C_logits, D_logits], dim=1), dim=1
        )
        preds = torch.argmax(probs, dim=1)
        preds = [["A", "B", "C", "D"][i.item()] for i in preds]

        truthy_array = [pred == label for pred, label in zip(preds, labels)]

        # Update serialized table with batch results
        serialized_table = update_serialized_table(
            accelerator,
            serialized_table,
            batch_squeezed["input_ids"],
            probs,
            labels,
            preds,
            truthy_array,
        )
        highest_batch_idx = idx

    # Gather results from all processes
    serialized_table_tensor = torch.cat(serialized_table, dim=0)
    gathered_table = torch.zeros(
        (
            get_world_size(),
            kwargs["batch_size"] * (highest_batch_idx + 1),
            kwargs["max_seq_len"] + 7,  # 7 is the size of other metadata
        ),
        dtype=torch.float32,
    ).to(accelerator.device)
    _temp = all_gather_into_tensor(gathered_table, serialized_table_tensor)
    gathered_table = gathered_table.view(
        gathered_table.shape[0] * gathered_table.shape[1], gathered_table.shape[2]
    )

    # Compute accuracy using QuestionHashTable
    question_hash_table = QuestionHashTable()
    for tensor in gathered_table:
        question_hash_table.add_entry(tensor)

    acc = question_hash_table.compute_accuracy()
    final_table = question_hash_table.get_serialized_table()

    return acc, final_table


PAD_TOKEN = ""

def evaluate_model(model, tokenizer, accelerator, args):
    """
    Evaluate a language model on the MMLU benchmark.

    This function sets up the model, tokenizer, and accelerator if not provided,
    prepares the datasets, and evaluates the model on each subject in the MMLU benchmark. This is a distributed implementation inspured heavily by https://github.com/ollmer/mmlu/blob/master/evaluate_hf.py

    Args:
        model (AutoModelForCausalLM, optional): The pre-trained language model. If None, it will be loaded.
        tokenizer (AutoTokenizer, optional): The tokenizer for the model. If None, it will be loaded.
        accelerator (Accelerator, optional): The Accelerator object for distributed training. If None, it will be created.
        args (argparse.Namespace): Command-line arguments containing model and evaluation settings.

    The function performs the following steps:
    1. Load and prepare the model, tokenizer, and accelerator if not provided.
    2. Configure the tokenizer's padding settings.
    3. Prepare the evaluation datasets.
    4. Evaluate the model on each subject in the MMLU benchmark.
    5. Write evaluation results to files if enabled.

    Global variables:
        PAD_TOKEN (str): The padding token used by the tokenizer.

    Note:
        - The function uses global variables and imported functions from other modules.
        - It assumes the existence of subjects, evaluation dataloaders, and file writing functions.
    """

    global PAD_TOKEN

    # Load and prepare model, tokenizer, and accelerator if not provided
    if model is None and tokenizer is None and accelerator is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        accelerator = Accelerator(fsdp_plugin=FSDP_PLUGIN)

        accelerator.free_memory()
        model = accelerator.prepare_model(model)
        accelerator.print(args.model_name + " prepared.")

    # Configure tokenizer padding settings
    tokenizer.padding_side = "left"  # NOTE: EXTREMELY CRUCIAL
    if args.eos_pad_token:
        tokenizer.pad_token = tokenizer.eos_token  # NOTE: Use for meta-llama-3-8b
        PAD_TOKEN = tokenizer.eos_token
    else:
        tokenizer.pad_token = "[PAD]"  # NOTE: Use for meta-llama-2-7b-chat-hf
        PAD_TOKEN = "[PAD]"

    # Prepare evaluation datasets
    user = os.environ.get("USER")
    data_dir = f"/data/{user}/capabilities-removal/batched_evaluation/data"
    if not check_dir_has_headers(data_dir):
        add_header_to_csv(data_dir)
    accelerator.print(f"Datasets prepared.")

    subject_accs = {}  # Store the accuracy for each subject

    # Evaluate model on each subject
    for subject in subjects:
        with accelerator.main_process_first():
            dataloader = get_eval_dataloaders(
                tokenizer, args.path_to_data, subject, args
            )
        dataloader = accelerator.prepare(dataloader)

        acc, subject_table = evaluate_subject(
            model,
            dataloader,
            tokenizer,
            accelerator,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
        accelerator.print("Average accuracy {:.3f} - {}".format(acc, subject))
        subject_accs[subject] = acc

        # Write results to files if enabled
        if accelerator.is_main_process:
            if not args.disable_file_writes:
                write_subject_table_to_file(subject_table, subject, tokenizer, args)
            write_accs_to_file(subject_accs, args)
        accelerator.wait_for_everyone()


def write_subject_table_to_file(subject_table, subject, tokenizer, args):
    """
    Write evaluation results for a specific subject to a CSV file.

    Args:
        subject_table (list): List of tensors containing evaluation results.
        subject (str): The subject being evaluated.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
        args (argparse.Namespace): Command-line arguments.
    """
    # Create directory for saving results
    os.makedirs(f"mmlu_{args.save_file_dir}", exist_ok=True)
    file_name = f"mmlu_{args.save_file_dir}/{subject}_metadata.csv"

    # Write header to the CSV file
    with open(file_name, "w") as f:
        f.write("prompt, probs, label, prediction, correctness\n")

    for tensor in subject_table:
        # Extract input_ids and probabilities from the tensor
        input_ids = tensor[: args.max_seq_len].tolist()
        LEN_OF_PROBS_DISTRIBUTION_IN_TENSOR = 4
        probs = tensor[
            args.max_seq_len : args.max_seq_len + LEN_OF_PROBS_DISTRIBUTION_IN_TENSOR
        ].tolist()

        # Extract label, prediction, and correctness
        label = ["A", "B", "C", "D"][int(tensor[-3].item())]
        pred = ["A", "B", "C", "D"][int(tensor[-2].item())]
        correctness = tensor[-1].item() == 1

        # Convert input_ids to integers
        input_ids = [int(element) for element in input_ids]

        # Determine the pad token ID to ignore
        if PAD_TOKEN == "[PAD]":
            pad_token_id_to_ignore = 0
        else:
            pad_token_id_to_ignore = tokenizer.encode(tokenizer.pad_token)[1]  # NOTE: This will mark eos token (token_id 128009) for filtering.

        # Remove padding from input_ids
        unpadded_input_ids = list(
            filter(lambda x: x != pad_token_id_to_ignore, input_ids)
        )

        # Decode the prompt and format probabilities
        prompt = tokenizer.decode(unpadded_input_ids)
        probs_string = "[" + ", ".join(map(str, probs)) + "]"

        # Append results to the CSV file
        with open(file_name, "a") as f:
            f.write(
                f'"{prompt}", "{probs_string}", "{label}", "{pred}", "{correctness}"\n'
            )

def write_accs_to_file(subject_accs, args):
    """
    Write accuracies for all subjects to a CSV file.

    Args:
        subject_accs (dict): Dictionary containing accuracies for each subject.
        args (argparse.Namespace): Command-line arguments.
    """
    # Create directory for saving results
    os.makedirs(f"mmlu_{args.save_file_dir}", exist_ok=True)
    file_name = f"mmlu_{args.save_file_dir}/mmlu_accs.csv"

    # Write header to the CSV file
    with open(file_name, "w") as f:
        f.write("subject, accuracy\n")

    # Write accuracies for each subject
    for subject, acc in subject_accs.items():
        with open(file_name, "a") as f:
            f.write(f"{subject}, {acc}\n")

def main():
    """
    Main function to set up and run the MMLU evaluation.
    """
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Set up argument parser
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER")

    # Add command-line arguments
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--num_fewshot_examples", "-nfe", type=int, default=5)
    parser.add_argument("--max_seq_len", "-msl", type=int, default=4096)
    parser.add_argument(
        "--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--eos_pad_token", "-eopt", action="store_true"
    )  # NOTE: SET TRUE FOR LLAMA3
    parser.add_argument(
        "--path_to_data",
        "-ptd",
        type=str,
        default=".../tamper-resistance/red_teaming/mmlu_eval/data",  # NOTE: You need to setup this data directory as described in the README!
    )
    parser.add_argument("--save_file_dir", "-sfn", type=str, default="interactive_test")
    parser.add_argument("--disable_file_writes", "-dfw", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)

    # Parse arguments
    args = parser.parse_args()

    # Set random seed for reproducibility
    fix_seed(args.seed)

    # Run evaluation
    evaluate_model(model=None, tokenizer=None, accelerator=None, args=args)

if __name__ == "__main__":
    main()
