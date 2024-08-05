from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import torch
import os
import pandas as pd

def load_csv_dataset(path: str = None):
    ds = load_dataset("csv", data_files=path)
    return ds

def get_eval_datasets(
    tokenizer,
    path: str = None,
    subject: str = None,
    args=None,
):
    def tokenize(sample, include_answer=True):
        choices = ["A", "B", "C", "D"]

        def format_subject(subject):
            l = subject.split("_")
            s = ""
            for entry in l:
                s += " " + entry
            return s

        def format_fewshot_example(df, idx, include_answer=True):
            prompt = df.iloc[idx, 0]
            k = df.shape[1] - 2
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer: "
            if include_answer:
                prompt += "{}\n\n".format(df.iloc[idx, k + 1])
            return prompt

        def format_question(
            question,
            choice1,
            choice2,
            choice3,
            choice4,
            answer,
            include_answer=include_answer,
        ):
            prompt = question
            for i, choice in enumerate(choices):
                prompt += "\n{}. {}".format(choice, locals()[f"choice{i+1}"])
            prompt += "\nAnswer:"
            if include_answer:
                prompt += "{}".format(answer)
            return prompt

        def gen_prompt(train_df, subject, k=-1):
            prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
                format_subject(subject)
            )
            if k == -1:
                k = train_df.shape[0]
            for i in range(k):
                prompt += format_fewshot_example(train_df, i)
            return prompt

        k = args.num_fewshot_examples

        dev_directory = os.path.join(path, "dev")

        dev_df = pd.read_csv(os.path.join(dev_directory, subject + "_dev.csv"), header=0)[:k]

        prompt_end = format_question(
            sample["question"],
            sample["choice1"],
            sample["choice2"],
            sample["choice3"],
            sample["choice4"],
            sample["answer"],
            include_answer=include_answer,
        )
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        result = tokenizer.__call__(
            prompt,
            max_length=args.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        while result["input_ids"].shape[-1] > args.max_seq_len:
            if k == 0:
                break
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            result = tokenizer.__call__(
                prompt,
                max_length=args.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_label = tokenizer.__call__(
            sample["answer"], padding="longest", return_tensors="pt"
        )

        # We want to supply the correct answers during training, but not during validation. That's why include_answers is an argument to this function.
        if include_answer == True:
            result["labels"] = result["input_ids"].clone()
        else:
            result["labels"] = tokenized_label["input_ids"]
        return result
    
    test_directory = os.path.join(path, "test")
    eval_dataset = load_csv_dataset(os.path.join(test_directory, subject + "_test.csv"))

    tokenize_without_answer = lambda sample: tokenize(sample, include_answer=False)

    eval_dataset["train"] = eval_dataset["train"].map(tokenize_without_answer)

    columns_to_remove = [
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "answer",
    ]

    eval_dataset["train"] = eval_dataset["train"].remove_columns(columns_to_remove)

    tokenized_eval = eval_dataset["train"]

    tokenized_dataset = DatasetDict(
        {"train": tokenized_eval}
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, return_tensors="pt"
    )

    return tokenized_dataset, data_collator


def get_eval_dataloaders(tokenizer, path, subject, args):
    tokenized_dataset, data_collator = get_eval_datasets(
        tokenizer, path=path, subject=subject, args=args
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset.get("train"),
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
    )

    return eval_dataloader