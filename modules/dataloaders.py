import re

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from objectives import DPOLoss
from transformers import DataCollatorForLanguageModeling
from utils import DPODataCollatorWithPadding


def load_pile_bio_retain_forget_data():
    full_dataset = load_dataset("lapisrocks/biology-pile-labeled", token=True)["train"]
    forget_data = full_dataset.filter(lambda x: x["label"] == True)
    retain_data = full_dataset.filter(lambda x: x["label"] == False)

    return retain_data, forget_data


def _preprocess(dataset, tokenize):
    raw_dataset = dataset.remove_columns(["meta", "reasoning"])
    raw_dataset = raw_dataset.rename_column("txt_chunk", "text")
    raw_dataset = raw_dataset.rename_column("label", "concept_label")
    tokenized_dataset = raw_dataset.map(tokenize).remove_columns(["text"])
    return tokenized_dataset


def get_pile_bio_retain_forget_heldout_datasets(
    tokenizer,
    cutoff_len: int = 256,
    refusal: bool = False,
    accelerator=None,
):
    def tokenize(sample, cutoff_len=cutoff_len):
        prompt = sample["text"]

        if refusal:
            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": REFUSAL},
            ]
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
            )

        result = tokenizer.__call__(
            prompt.strip(tokenizer.eos_token).strip(tokenizer.bos_token),
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            add_special_tokens=False,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    retain_dataset, forget_dataset = load_pile_bio_retain_forget_data()
    tokenized_retain_dataset = _preprocess(retain_dataset, tokenize)
    tokenized_forget_dataset = _preprocess(forget_dataset, tokenize)
    split = tokenized_forget_dataset.train_test_split(test_size=0.20, seed=42)
    tokenized_forget_heldout_dataset = split["test"]
    tokenized_forget_dataset = split["train"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return (
        tokenized_retain_dataset,
        tokenized_forget_dataset,
        tokenized_forget_heldout_dataset,
        data_collator,
    )


def get_bio_pilecamel_forget_with_heldout_dataloaders(
    tokenizer, accelerator, path, args
):
    (
        tokenized_retain_dataset,
        tokenized_forget_dataset,
        tokenized_forget_heldout_dataset,
        data_collator,
    ) = get_pile_bio_retain_forget_heldout_datasets(
        tokenizer, cutoff_len=256, refusal=False, accelerator=accelerator
    )
    (
        tokenized_camel_forget_dataset,
        tokenized_camel_forget_heldout_dataset,
        _,
    ) = get_camel_ai_datasets(tokenizer, path, args, accelerator)
    magpie_train, _ = get_magpie_datasets(tokenizer, path, args, cutoff_len=256)

    pure_retain_dataloader = torch.utils.data.DataLoader(
        tokenized_retain_dataset.remove_columns(["concept_label"]),
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    mixed_magpie_retain_dataloader = torch.utils.data.DataLoader(
        concatenate_datasets(
            [
                tokenized_retain_dataset.remove_columns(["concept_label"]),
                magpie_train.select(range(len(tokenized_retain_dataset) // 2)),
            ]
        ),  # 2/3 raw text 1/3 instruction split between retain and magpie
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    pile_forget_dataloader = torch.utils.data.DataLoader(
        tokenized_forget_dataset,
        batch_size=args.mlac_inner_loop_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    camel_forget_dataloader = torch.utils.data.DataLoader(
        tokenized_camel_forget_dataset,
        batch_size=args.mlac_inner_loop_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    bio_combined_dataloader = torch.utils.data.DataLoader(
        concatenate_datasets(
            [tokenized_forget_dataset, tokenized_camel_forget_dataset]
        ),
        batch_size=args.mlac_inner_loop_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    retain_bio_combined_dataloader = torch.utils.data.DataLoader(
        concatenate_datasets(
            [
                tokenized_retain_dataset,
                tokenized_forget_dataset,
                tokenized_camel_forget_dataset,
            ]
        ),
        batch_size=args.mlac_inner_loop_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    heldout_dataset = concatenate_datasets(
        [tokenized_forget_heldout_dataset, tokenized_camel_forget_heldout_dataset]
    )

    heldout_dataloader = torch.utils.data.DataLoader(
        heldout_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    if accelerator is not None:
        # retain_dataloader = accelerator.prepare(retain_dataloader)
        pure_retain_dataloader = accelerator.prepare(pure_retain_dataloader)
        mixed_magpie_retain_dataloader = accelerator.prepare(
            mixed_magpie_retain_dataloader
        )
        pile_forget_dataloader = accelerator.prepare(pile_forget_dataloader)
        camel_forget_dataloader = accelerator.prepare(camel_forget_dataloader)
        heldout_dataloader = accelerator.prepare(heldout_dataloader)

    return (
        pure_retain_dataloader,
        mixed_magpie_retain_dataloader,
        pile_forget_dataloader,
        camel_forget_dataloader,
        bio_combined_dataloader,
        heldout_dataloader,
        retain_bio_combined_dataloader,
    )


######## DATALOADERS


def load_csv_dataset(path: str = None):
    ds = load_dataset("csv", data_files=path)
    return ds


def get_camel_ai_datasets(
    tokenizer,
    path: str = "/data/private_models/cais_models/robust_unlearning/data/camel_ai/biology.csv",
    args=None,
    accelerator=None,
):
    # combine use pilebio retain, combine pilebio forget with camel ai forget
    def camel_tokenize(sample, cutoff_len=256):
        prompt = sample["camel_ai_message_2_text_chunk"]
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            add_special_tokens=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    camel_dataset = load_csv_dataset(path)
    camel_dataset = camel_dataset["train"].add_column(
        "concept_label", [True] * len(camel_dataset["train"])
    )
    tokenized_camel_dataset = camel_dataset.map(camel_tokenize).remove_columns(
        ["camel_ai_message_2_text_chunk"]
    )
    split = tokenized_camel_dataset.train_test_split(test_size=0.20, seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return split["train"], split["test"], data_collator


def get_fineweb_dataloaders(tokenizer, accelerator, path, args):
    fw = load_dataset("lapisrocks/fineweb-300k")["train"]

    def tokenize(sample):
        result = tokenizer.__call__(
            sample["text"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["input_ids"] = result["input_ids"].squeeze()
        result["labels"] = result["input_ids"].clone()
        result["attention_mask"] = result["attention_mask"].squeeze()
        return result

    # remove columns that aren't text 'text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'
    columns_to_remove = [
        "id",
        "dump",
        "url",
        "date",
        "file_path",
        "language",
        "language_score",
        "token_count",
    ]
    fw = fw.remove_columns(columns_to_remove)
    tokenized_fw = fw.map(tokenize).remove_columns(["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    fw_dataloader = torch.utils.data.DataLoader(
        tokenized_fw,
        batch_size=args.mlac_inner_loop_batch_size,
        collate_fn=data_collator,
    )
    if accelerator is not None:
        fw_dataloader = accelerator.prepare(fw_dataloader)
    return fw_dataloader


def get_pilebio_fineweb_mixed_dataloaders(tokenizer, accelerator, path, args):
    fw = load_dataset("lapisrocks/fineweb-300k")["train"]
    retain_dataset, forget_dataset = load_pile_bio_retain_forget_data()

    def tokenize(sample):
        result = tokenizer.__call__(
            sample["text"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["input_ids"] = result["input_ids"].squeeze()
        result["labels"] = result["input_ids"].clone()
        result["attention_mask"] = result["attention_mask"].squeeze()
        return result

    columns_to_remove = [
        "id",
        "dump",
        "url",
        "date",
        "file_path",
        "language",
        "language_score",
        "token_count",
    ]

    def camel_tokenize(sample, cutoff_len=256):
        prompt = sample["camel_ai_message_2_text_chunk"]
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            add_special_tokens=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    camel_dataset = load_csv_dataset(
        "/data/private_models/cais_models/robust_unlearning/data/camel_ai/biology.csv"
    )
    camel_dataset = camel_dataset["train"].add_column(
        "concept_label", [True] * len(camel_dataset["train"])
    )
    tokenized_camel_dataset = camel_dataset.map(camel_tokenize).remove_columns(
        ["camel_ai_message_2_text_chunk"]
    )
    tokenized_camel_dataset = tokenized_camel_dataset.train_test_split(
        test_size=0.20, seed=42
    )["train"]
    fw = fw.remove_columns(columns_to_remove)
    tokenized_fw = fw.map(tokenize).remove_columns(["text"])
    tokenized_pile_forget_dataset = _preprocess(forget_dataset, tokenize)
    tokenized_pile_retain_dataset = _preprocess(retain_dataset, tokenize)
    tokenized_pile_forget_dataset = tokenized_pile_forget_dataset.train_test_split(
        test_size=0.20, seed=42
    )["train"]

    # concatenate the first half of fineweb with pile, second half with camel
    pilebio_mixed = concatenate_datasets(
        [
            tokenized_fw.select(range(len(tokenized_pile_forget_dataset))),
            tokenized_pile_forget_dataset,
        ]
    ).remove_columns(["concept_label"])
    camelbio_mixed = concatenate_datasets(
        [
            tokenized_camel_dataset,
            tokenized_fw.select(
                range(
                    len(tokenized_pile_forget_dataset),
                    len(tokenized_camel_dataset) + len(tokenized_pile_forget_dataset),
                )
            ),
        ]
    ).remove_columns(["concept_label"])

    # also mix tokenized pile retain dataset with pile and camel
    pilebio_pileretain_mixed = concatenate_datasets(
        [
            tokenized_pile_forget_dataset,
            tokenized_pile_retain_dataset.select(
                range(len(tokenized_pile_forget_dataset))
            ),
        ]
    )
    camelbio_pileretain_mixed = concatenate_datasets(
        [
            tokenized_camel_dataset,
            tokenized_pile_retain_dataset.select(range(len(tokenized_camel_dataset))),
        ]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    pilebio_mixed_dataloader = torch.utils.data.DataLoader(
        pilebio_mixed,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    camelbio_mixed_dataloader = torch.utils.data.DataLoader(
        camelbio_mixed,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    pilebio_pileretain_mixed_dataloader = torch.utils.data.DataLoader(
        pilebio_pileretain_mixed,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    camelbio_pileretain_mixed_dataloader = torch.utils.data.DataLoader(
        camelbio_pileretain_mixed,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    if accelerator is not None:
        pilebio_mixed_dataloader = accelerator.prepare(pilebio_mixed_dataloader)
        camelbio_mixed_dataloader = accelerator.prepare(camelbio_mixed_dataloader)
    return (
        pilebio_mixed_dataloader,
        camelbio_mixed_dataloader,
        pilebio_pileretain_mixed_dataloader,
        camelbio_pileretain_mixed_dataloader,
    )


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


def get_bio_multi_dists_dataloaders(tokenizer, accelerator, path, args):
    (
        pure_retain_dataloader,
        mixed_magpie_retain_dataloader,
        pile_bio_dataloader,
        camel_bio_dataloader,
        bio_combined_dataloader,
        bio_heldout_dataloader,
        retain_bio_combined_dataloader,
    ) = get_bio_pilecamel_forget_with_heldout_dataloaders(
        tokenizer, accelerator, path, args
    )
    fineweb_dataloader = get_fineweb_dataloaders(tokenizer, accelerator, path, args)
    (
        pilebio_mixed_dataloader,
        camelbio_mixed_dataloader,
        pilebio_pileretain_mixed_dataloader,
        camelbio_pileretain_mixed_dataloader,
    ) = get_pilebio_fineweb_mixed_dataloaders(tokenizer, accelerator, path, args)

    dataloaders = {
        "retain": mixed_magpie_retain_dataloader,
        "pile-bio": pile_bio_dataloader,
        "forget_train": pile_bio_dataloader,
        "retain_forget_switch": pile_bio_dataloader,
        "camel-bio": camel_bio_dataloader,
        "bio-combined": bio_combined_dataloader,
        "retain-bio-combined": retain_bio_combined_dataloader,
        "pilebio-fw-mixed": pilebio_mixed_dataloader,
        "camelbio-fw-mixed": camelbio_mixed_dataloader,
        "pile-retain": pure_retain_dataloader,
        "adv_retain": pure_retain_dataloader,
        "pilebio-pileretain-mixed": pilebio_pileretain_mixed_dataloader,
        "camelbio-pileretain-mixed": camelbio_pileretain_mixed_dataloader,
        "fineweb": fineweb_dataloader,
        "meta": bio_heldout_dataloader,
    }

    return dataloaders


def load_chem_dataset():
    return load_dataset(
        "text",
        data_files="/data/bhrugu_bharathi/capabilities-removal/corpora_generation/chem_corpus.txt",
    )


def load_cyber_dataset():
    return load_dataset("justinwangx/CTFtime")


def _preprocess_chem(dataset, tokenize):
    tokenized_dataset = dataset.map(tokenize)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    return tokenized_dataset


def _preprocess_cyber(dataset, tokenize):
    # tokenized_dataset = dataset.rename_column("text_chunk", "text")
    tokenized_dataset = dataset.map(tokenize)
    tokenized_dataset = tokenized_dataset.remove_columns(["text_chunk"])
    return tokenized_dataset


def get_chem_datasets(
    tokenizer,
    accelerator,
    cutoff_len: int = 256,
    args=None,
):
    def tokenize(sample):
        result = tokenizer.__call__(
            sample["text"].strip(tokenizer.eos_token).strip(tokenizer.bos_token),
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            add_special_tokens=False,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    full_dataset = load_chem_dataset()
    full_dataset["train"] = full_dataset["train"].filter(lambda x: x["text"] != "")
    split_dataset = full_dataset["train"].train_test_split(test_size=0.20, seed=42)
    tokenized_forget_train_dataset = _preprocess_chem(split_dataset["train"], tokenize)
    tokenized_forget_test_dataset = _preprocess_chem(split_dataset["test"], tokenize)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenized_forget_train_dataset, tokenized_forget_test_dataset, data_collator


def get_chem_dataloaders(tokenizer, accelerator, path, args):
    tokenized_forget_train, tokenized_forget_test, data_collator = get_chem_datasets(
        tokenizer, accelerator
    )

    (
        tokenized_retain_dataset,
        _,
        _,
        _,
    ) = get_pile_bio_retain_forget_heldout_datasets(
        tokenizer, cutoff_len=256, refusal=False, accelerator=accelerator
    )
    magpie_train, _ = get_magpie_datasets(tokenizer, path, args, cutoff_len=256)

    pure_retain_dataloader = torch.utils.data.DataLoader(
        tokenized_retain_dataset.remove_columns(["concept_label"]),
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    mixed_magpie_retain_dataloader = torch.utils.data.DataLoader(
        concatenate_datasets(
            [tokenized_retain_dataset.remove_columns(["concept_label"]), magpie_train]
        ),
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    forget_train_dataloader = torch.utils.data.DataLoader(
        tokenized_forget_train,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        tokenized_forget_test,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    if accelerator is not None:
        pure_retain_dataloader = accelerator.prepare(pure_retain_dataloader)
        mixed_magpie_retain_dataloader = accelerator.prepare(
            mixed_magpie_retain_dataloader
        )
        forget_train_dataloader = accelerator.prepare(forget_train_dataloader)
        forget_test_dataloader = accelerator.prepare(forget_test_dataloader)
    # return tokenized_retain_dataloader, forget_train_dataloader, forget_test_dataloader
    dataloaders = {
        "retain": mixed_magpie_retain_dataloader,
        "adv_retain": pure_retain_dataloader,
        "forget_train": forget_train_dataloader,
        "retain_forget_switch": forget_train_dataloader,
        "meta": forget_test_dataloader,
    }

    return dataloaders


def get_cyber_datasets(
    tokenizer,
    accelerator,
    cutoff_len: int = 256,
    args=None,
):
    def tokenize(sample):
        result = tokenizer.__call__(
            sample["text_chunk"].strip(tokenizer.eos_token).strip(tokenizer.bos_token),
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            add_special_tokens=False,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    full_dataset = load_cyber_dataset()
    split_dataset = full_dataset["train"].train_test_split(test_size=0.20, seed=42)

    # if accelerator.is_main_process:
    #     import pdb; pdb.set_trace()
    # accelerator.wait_for_everyone()
    tokenized_forget_train_dataset = _preprocess_cyber(split_dataset["train"], tokenize)
    tokenized_forget_test_dataset = _preprocess_cyber(split_dataset["test"], tokenize)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenized_forget_train_dataset, tokenized_forget_test_dataset, data_collator


def get_cyber_dataloaders(tokenizer, accelerator, path, args):
    tokenized_forget_train, tokenized_forget_test, data_collator = get_cyber_datasets(
        tokenizer, accelerator
    )

    (
        tokenized_retain_dataset,
        _,
        _,
        _,
    ) = get_pile_bio_retain_forget_heldout_datasets(
        tokenizer, cutoff_len=256, refusal=False, accelerator=accelerator
    )
    magpie_train, _ = get_magpie_datasets(tokenizer, path, args, cutoff_len=256)

    tokenized_retain_dataloader = torch.utils.data.DataLoader(
        concatenate_datasets(
            [tokenized_retain_dataset.remove_columns(["concept_label"]), magpie_train]
        ),
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    forget_train_dataloader = torch.utils.data.DataLoader(
        tokenized_forget_train,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    forget_test_dataloader = torch.utils.data.DataLoader(
        tokenized_forget_test,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    if accelerator is not None:
        tokenized_retain_dataloader = accelerator.prepare(tokenized_retain_dataloader)
        forget_train_dataloader = accelerator.prepare(forget_train_dataloader)
        forget_test_dataloader = accelerator.prepare(forget_test_dataloader)
    dataloaders = {
        "retain": tokenized_retain_dataloader,
        "adv_retain": tokenized_retain_dataloader,
        "forget_train": forget_train_dataloader,
        "retain_forget_switch": forget_train_dataloader,
        "meta": forget_test_dataloader,
    }

    return dataloaders


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


def parse_conversation(conversation_string):
    pattern = r"(Human|Assistant):\s*(.*?)(?=\s*(?:Human|Assistant):\s*|$)"
    matches = re.findall(pattern, conversation_string, re.DOTALL)

    conversation_data = []
    role_dict = {"Human": "user", "Assistant": "assistant"}
    for role, message in matches:
        _role = role_dict[role]
        # if previous role is the same, append to the previous message with a newline
        if conversation_data and conversation_data[-1]["role"] == _role:
            conversation_data[-1]["content"] += "\n" + message
        else:
            conversation_data.append({"role": _role, "content": message})

    return conversation_data


def hh_rlhf_format(dataset, tokenizer):
    def apply_format(sample):
        _chosen = parse_conversation(sample["chosen"])
        _rejected = parse_conversation(sample["rejected"])
        prompt = tokenizer.apply_chat_template(_chosen[:-1], tokenize=False).strip(
            tokenizer.eos_token
        )

        chosen = _chosen[-1]["content"]
        rejected = _rejected[-1]["content"]
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            # swapped
            # "chosen": rejected,
            # "rejected": chosen,
        }

    dataset = dataset.map(apply_format)
    return dataset


def build_tokenized_answer(tokenizer, prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError(
            "Prompt input ids and answer input ids should have the same length."
        )

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][
        :response_token_ids_start_idx
    ]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError(
            "Prompt input ids and attention mask should have the same length."
        )

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][
        response_token_ids_start_idx:
    ]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )


def apply_dpo_tokenization(dataset, tokenizer, dataset_size=1000):
    def tokenize_row(
        feature,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
        label_pad_token_id=-100,
        truncation_mode="keep_end",
    ):
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = build_tokenized_answer(tokenizer, prompt, rejected)

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(
            chosen_prompt_len_input_ids, rejected_prompt_len_input_ids
        )

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [
                a != b
                for a, b in zip(
                    chosen_tokens["prompt_input_ids"],
                    rejected_tokens["prompt_input_ids"],
                )
            ]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt NOTE: Already added
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.encode("<|endoftext|>")[0]

        prompt_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + prompt_tokens[
            "prompt_input_ids"
        ]
        chosen_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + chosen_tokens[
            "prompt_input_ids"
        ]
        rejected_tokens["prompt_input_ids"] = [
            tokenizer.bos_token_id
        ] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
            "prompt_attention_mask"
        ]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
            "prompt_attention_mask"
        ]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
            "prompt_attention_mask"
        ]

        # # add EOS token to end of answer # NOTE: Already added
        chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > max_length
            ):
                if truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][:max_prompt_length]
                elif truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-max_prompt_length:]
                else:
                    raise ValueError(f"Unknown truncation mode: {truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > max_length
            ):
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][
                        : max_length - max_prompt_length
                    ]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k]
            for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k]
            for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][
            : len(rejected_tokens["prompt_input_ids"])
        ] = [label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        return batch

    return dataset.map(tokenize_row)


def get_anthropic_hh_dpo_dataset(tokenizer, dataset_size=1000):
    data = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden")[
        "train"
    ].select(range(dataset_size))
    data = hh_rlhf_format(data, tokenizer)
    tokenized_dataset = apply_dpo_tokenization(data, tokenizer)
    return tokenized_dataset


def flatten(xss):
    # flattens list of lists
    return [x for xs in xss for x in xs]


def flatten_list_of_dicts(list_of_dicts):
    # flattens list of dicts; returns a dict with keys as the concatenated keys
    return {k: v for d in list_of_dicts for k, v in d.items()}


def get_magpie_datasets(tokenizer, path, args, cutoff_len: int = 512):
    dataset = load_dataset("Magpie-Align/Magpie-Pro-MT-300K-v0.1")["train"]

    def tokenize(sample, cutoff_len=cutoff_len):
        MAPPING = {"human": "user", "gpt": "assistant"}
        chat = []
        for message in sample["conversations"]:
            chat.append({"role": MAPPING[message["from"]], "content": message["value"]})
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors="pt",
        )
        result["input_ids"] = result["input_ids"].squeeze()
        result["labels"] = result["input_ids"].clone()
        result["attention_mask"] = result["attention_mask"].squeeze()
        return result

    # remove columns that are not ["input1", "output1"]
    rm_cols = [
        col
        for col in dataset.column_names
        if col not in ["input_ids", "labels", "attention_mask"]
    ]
    dataset = dataset.map(tokenize).remove_columns(rm_cols)
    split = dataset.train_test_split(test_size=0.20, seed=42)
    return split["train"], split["test"]


def get_magpie_dataloaders(tokenizer, path, args, cutoff_len=512):
    train, test = get_magpie_datasets(tokenizer, path, args, cutoff_len=cutoff_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    return test_dataloader, train_dataloader


def get_anthropic_hh_dpo_dataloaders(tokenizer, accelerator, path, args, model=None):
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )
    tokenized_dataset = get_anthropic_hh_dpo_dataset(
        tokenizer, dataset_size=args.max_data_size
    )

    def add_logps(example):
        (
            reference_chosen_logp,
            reference_rejected_logp,
        ) = DPOLoss.compute_reference_log_probs(
            model, data_collator([example]), accelerator
        )
        example["reference_chosen_logps"] = reference_chosen_logp.cpu()
        example["reference_rejected_logps"] = reference_rejected_logp.cpu()
        return example

    tokenized_dataset = tokenized_dataset.map(add_logps)

    pref_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    magpie_train, _ = get_magpie_dataloaders(tokenizer, path, args, cutoff_len=1024)
    return {"retain": magpie_train, "meta": pref_dataloader}
