import torch

import os
import random

import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .categories import subjects
from .eval_dataloader import get_eval_dataloaders
from .add_headers_to_csv import check_dir_has_headers, add_header_to_csv
from .question_hash_table import QuestionHashTable, update_serialized_table

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM
import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed import get_world_size, get_rank, all_gather, all_gather_into_tensor

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    PhiDecoderLayer,
    MistralDecoderLayer,
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

@torch.no_grad()
def evaluate_subject(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    **kwargs,
):
    model.config.use_cache = False
    model.eval()
    serialized_table = []
    highest_batch_idx = 0

    for idx, batch in enumerate(eval_dataloader):
        batch_squeezed = {key: value.squeeze(1) for key, value in batch.items()}
        
        labels = batch_squeezed["labels"][:, -1]
        labels = [tokenizer.decode(label.item()) for label in labels]

        model_outputs = model(input_ids=batch_squeezed["input_ids"], attention_mask=batch_squeezed["attention_mask"])
        logits = model_outputs.logits[:, -1]

        A_logits = logits[:, tokenizer("A").input_ids[-1]]
        B_logits = logits[:, tokenizer("B").input_ids[-1]]
        C_logits = logits[:, tokenizer("C").input_ids[-1]]
        D_logits = logits[:, tokenizer("D").input_ids[-1]]

        probs = torch.nn.functional.softmax(torch.stack([A_logits, B_logits, C_logits, D_logits], dim=1), dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = [["A", "B", "C", "D"][i.item()] for i in preds]

        truthy_array = [pred == label for pred, label in zip(preds, labels)]

        serialized_table = update_serialized_table(accelerator, serialized_table, batch_squeezed["input_ids"], probs, labels, preds, truthy_array)
        highest_batch_idx = idx

    serialized_table_tensor = torch.cat(serialized_table, dim=0)

    SIZE_OF_PROBS_DISTRIBUTION_IN_TENSOR = 4
    SIZE_OF_LABELS_IN_TENSOR = 1
    SIZE_OF_PREDS_IN_TENSOR = 1
    SIZE_OF_TRUTHY_IN_TENSOR = 1
    SIZE_OF_OTHER_METADATA = SIZE_OF_PROBS_DISTRIBUTION_IN_TENSOR + SIZE_OF_LABELS_IN_TENSOR + SIZE_OF_PREDS_IN_TENSOR + SIZE_OF_TRUTHY_IN_TENSOR

    gathered_table = torch.zeros((get_world_size(), kwargs["batch_size"] * (highest_batch_idx + 1), kwargs["max_seq_len"] + SIZE_OF_OTHER_METADATA), dtype=torch.float32).to(accelerator.device)
    _temp = all_gather_into_tensor(gathered_table, serialized_table_tensor)
    gathered_table = gathered_table.view(gathered_table.shape[0] * gathered_table.shape[1], gathered_table.shape[2])

    question_hash_table = QuestionHashTable()
    for tensor in gathered_table:
        question_hash_table.add_entry(tensor)

    acc = question_hash_table.compute_accuracy()
    final_table = question_hash_table.get_serialized_table()

    return acc, final_table

PAD_TOKEN  = ''

def evaluate_model(model, tokenizer, accelerator, args):
    if model is None and tokenizer is None and accelerator is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        accelerator = Accelerator(fsdp_plugin=FSDP_PLUGIN)

        accelerator.free_memory()
        model = accelerator.prepare_model(model)
        accelerator.print(args.model_name + " prepared.")

    tokenizer.padding_side = "left" # NOTE: EXTREMELY CRUCIAL
    if args.eos_pad_token:
        tokenizer.pad_token = tokenizer.eos_token # NOTE: Use for meta-llama-3-8b
        PAD_TOKEN = tokenizer.eos_token
    else:
        tokenizer.pad_token = "[PAD]" # NOTE: Use for meta-llama-2-7b-chat-hf
        PAD_TOKEN = "[PAD]"

    user = os.environ.get("USER")
    data_dir = f"/data/{user}/capabilities-removal/batched_evaluation/data"
    if not check_dir_has_headers(data_dir):
        add_header_to_csv(data_dir)
    accelerator.print(f"Datasets prepared.")

    subject_accs = {} # Store the accuracy for each subject

    for subject in subjects:
        with accelerator.main_process_first():
            dataloader = get_eval_dataloaders(tokenizer, args.path_to_data, subject, args)
        dataloader = accelerator.prepare(dataloader)

        acc, subject_table = evaluate_subject(model, dataloader, tokenizer, accelerator, batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        accelerator.print("Average accuracy {:.3f} - {}".format(acc, subject))
        subject_accs[subject] = acc
        if accelerator.is_main_process:
            if not args.disable_file_writes:
                write_subject_table_to_file(subject_table, subject, tokenizer, args)
            write_accs_to_file(subject_accs, args)
        accelerator.wait_for_everyone()

def write_subject_table_to_file(subject_table, subject, tokenizer, args):
    os.makedirs(f"mmlu_{args.save_file_dir}", exist_ok=True)
    file_name = f"mmlu_{args.save_file_dir}/{subject}_metadata.csv"

    with open(file_name, "w") as f:
        f.write("prompt, probs, label, prediction, correctness\n")
    
    for tensor in subject_table:
        input_ids = tensor[:args.max_seq_len].tolist()
        LEN_OF_PROBS_DISTRIBUTION_IN_TENSOR = 4
        probs = tensor[args.max_seq_len:args.max_seq_len + LEN_OF_PROBS_DISTRIBUTION_IN_TENSOR].tolist()
        label = ["A", "B", "C", "D"][int(tensor[-3].item())]
        pred = ["A", "B", "C", "D"][int(tensor[-2].item())]
        correctness = tensor[-1].item() == 1

        input_ids = [int(element) for element in input_ids]
        
        if(PAD_TOKEN == "[PAD]"):
            pad_token_id_to_ignore = 0
        else:
            pad_token_id_to_ignore = tokenizer.encode(tokenizer.pad_token)[1] #NOTE: Will mark token_id 128009 for filtering.
        unpadded_input_ids = list(filter(lambda x: x != pad_token_id_to_ignore, input_ids))

        prompt = tokenizer.decode(unpadded_input_ids)
        probs_string = "[" + ', '.join(map(str, probs)) + "]"

        with open(file_name, "a") as f:
            f.write(f"\"{prompt}\", \"{probs_string}\", \"{label}\", \"{pred}\", \"{correctness}\"\n")

def write_accs_to_file(subject_accs, args):
    os.makedirs(f"mmlu_{args.save_file_dir}", exist_ok=True)
    file_name = f"mmlu_{args.save_file_dir}/mmlu_accs.csv"

    with open(file_name, "w") as f:
        f.write("subject, accuracy\n")
    
    for subject, acc in subject_accs.items():
        with open(file_name, "a") as f:
            f.write(f"{subject}, {acc}\n")

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER")
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--num_fewshot_examples", "-nfe", type=int, default=5)
    parser.add_argument("--max_seq_len", "-msl", type=int, default=4096)
    parser.add_argument("--model_name", "-mn", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") #meta-llama/Llama-2-7b-chat-hf
    parser.add_argument("--model_type", "-mt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--eos_pad_token", "-eopt", action="store_true") # NOTE: SET TRUE FOR LLAMA3
    parser.add_argument("--path_to_data", "-ptd", type=str, default=f"/data/{user}/capabilities-removal/batched_evaluation/data")
    parser.add_argument("--save_file_dir", "-sfn", type=str, default="interactive_test")
    parser.add_argument("--disable_file_writes", "-dfw", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()
    fix_seed(args.seed)
    evaluate_model(model=None, tokenizer=None, accelerator=None, args=args)
    

if __name__ == "__main__":
    main()

