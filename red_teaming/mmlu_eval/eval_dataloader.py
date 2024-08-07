# Import necessary libraries
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import torch
import os
import pandas as pd


def load_csv_dataset(path: str = None):
    """Load a CSV dataset from the given path."""
    ds = load_dataset("csv", data_files=path)
    return ds


def get_eval_datasets(
    tokenizer,
    path: str = None,
    subject: str = None,
    args=None,
):
    """
    To improve our evaluation speed, we adapt the data processing steps in https://github.com/ollmer/mmlu/blob/master/evaluate_hf.py converting each subject level CSV file into a tokenized dataset. This function tokenizes the evaluation dataset for a given subject.
    
    Args:
    tokenizer (AutoTokenizer): The tokenizer to use.
    path (str): The path to the directory containing the original evaluation CSV files. Details on how to set these directories up can be found in the README.
    subject (str): The subject to process.
    args: Contains the allowed maximum sequence length, number of few-shot examples, and batch size.

    Returns:
    DatasetDict: The subject level tokenized evaluation dataset.
    DataCollatorForTokenClassification: The data collator.
    """

    def tokenize(sample, include_answer=True):
        """
        Tokenizes a question with prepended few-shot prompts from the evaluation dataset.

        Args:
        sample (dict): The sample to tokenize.
        include_answer (bool): Whether to include the answer in the tokenization.
        
        Returns:
        dict: The formatted, tokenized result.
        """
        choices = ["A", "B", "C", "D"]

        def format_subject(subject):
            """Format the subject by joining words with spaces."""
            l = subject.split("_")
            s = ""
            for entry in l:
                s += " " + entry
            return s

        def format_fewshot_example(df, idx, include_answer=True):
            """Format a single few-shot example from the dataframe."""
            prompt = df.iloc[idx, 0]
            k = df.shape[1] - 2
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer: "
            if include_answer:
                prompt += "{}\n\n".format(df.iloc[idx, k + 1])
            return prompt

        def format_question(question, choice1, choice2, choice3, choice4, answer, include_answer=include_answer):
            """Format the main question with its choices and answer."""
            prompt = question
            for i, choice in enumerate(choices):
                prompt += "\n{}. {}".format(choice, locals()[f"choice{i+1}"])
            prompt += "\nAnswer:"
            if include_answer:
                prompt += "{}".format(answer)
            return prompt

        def gen_prompt(train_df, subject, k=-1):
            """Generate the full prompt with few-shot examples."""
            prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
                format_subject(subject)
            )
            if k == -1:
                k = train_df.shape[0]
            for i in range(k):
                prompt += format_fewshot_example(train_df, i)
            return prompt

        # Set the number of few-shot examples
        k = args.num_fewshot_examples

        # Load the original dataset containing the few-shot examples
        dev_directory = os.path.join(path, "dev")
        dev_df = pd.read_csv(
            os.path.join(dev_directory, subject + "_dev.csv"), header=0
        )[:k]

        # Format the current question
        prompt_end = format_question(
            sample["question"],
            sample["choice1"],
            sample["choice2"],
            sample["choice3"],
            sample["choice4"],
            sample["answer"],
            include_answer=include_answer,
        )

        # Generate the full prompt with few-shot examples
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # Tokenize the full prompt
        result = tokenizer.__call__(
            prompt,
            max_length=args.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Reduce the number of few-shot examples if the sequence is too long
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

        # Tokenize the answer separately
        tokenized_label = tokenizer.__call__(
            sample["answer"], padding="longest", return_tensors="pt"
        )

        # Set up labels based on whether to include answers
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

    tokenized_dataset = DatasetDict({"train": tokenized_eval})

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")

    return tokenized_dataset, data_collator


def get_eval_dataloaders(tokenizer, path, subject, args):
    """
    Create a DataLoader out of the corresponding subject level dataset.

    Args:
    tokenizer: The tokenizer to use for processing text.
    path (str): Path to the dataset directory.
    subject (str): The subject of the dataset.
    args: Additional arguments, including batch size.

    Returns:
    torch.utils.data.DataLoader: A DataLoader for the subject level evaluation.
    """
    
    # Get the tokenized dataset and data collator
    tokenized_dataset, data_collator = get_eval_datasets(
        tokenizer, path=path, subject=subject, args=args
    )
    
    # Create a DataLoader for the evaluation dataset
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset.get("train"),  # Use the 'train' split of the dataset
        batch_size=args.batch_size,      # Set batch size from args
        collate_fn=data_collator,        # Use the data collator for batching
        shuffle=False,                   # Don't shuffle for evaluation
        drop_last=False,                 # Keep all samples, even if last batch is smaller
    )
    
    return eval_dataloader
