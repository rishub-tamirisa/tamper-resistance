# MMLU Evaluation Tool

This tool is designed to evaluate language models on the Massive Multitask Language Understanding (MMLU) benchmark. It supports distributed evaluation using the Accelerate library and is compatible with most model architectures. This implementation was heavily inspired by the implementation provided in [ollmer/mmlu](https://github.com/ollmer/mmlu/blob/master/evaluate_hf.py).

## Setup

1. Install the requirements as described in the top-level README. 

> [!IMPORTANT]  
> **This next step is crucial, as you will not be able to run the evaluation without doing this!**
> 
> Follow these steps:
> 1. Navigate to the evaluation directory:
>    ```
>    cd tamper-resistance/red_teaming/mmlu_eval
>    ```
> 2. Go to [hendrycks/test](https://github.com/hendrycks/test?tab=readme-ov-file).
> 3. Download the [data.tar](https://people.eecs.berkeley.edu/~hendrycks/data.tar) file.
> 4. Extract the contents:
>    ```
>    tar -xvf data.tar
>    ```
> 5. Now you can pass `.../tamper-resistance/red_teaming/mmlu_eval/data` as the argument `--path_to_data` in `eval.py`.
> 
> Note: `add_headers_to_csv.py` will ensure that the files in `data/` are preprocessed correctly.

## Usage

To run the script, use the following command structure:

```bash
accelerate launch --config_file $FxACCEL_CONFIG eval.py --model_name {YOUR MODEL} --model_type {TOKENIZER FOR YOUR MODEL}
```

## Options

The script accepts the following command-line arguments:

- `--batch_size`, `-bs`: Batch size for evaluation (default: 8)
- `--num_fewshot_examples`, `-nfe`: Number of few-shot examples to use (default: 5)
- `--max_seq_len`, `-msl`: Maximum sequence length (default: 4096)
- `--model_name`, `-mn`: Name of the pre-trained model (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--model_type`, `-mt`: Type of the model for tokenizer (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--eos_pad_token`, `-eopt`: Use EOS token for padding (must set for LLaMA3!)
- `--path_to_data`, `-ptd`: Path to the MMLU dataset (default: ".../tamper-resistance/red_teaming/mmlu_eval/data")
- `--save_file_dir`, `-sfn`: Directory to save output files (default: "interactive_test")
- `--disable_file_writes`, `-dfw`: Disable writing detailed results to files
- `--seed`, `-s`: Random seed for reproducibility (default: 42)

## Interpreting Results

Upon completion of the program, a directory called `mmlu_{save_file_directory}` will be populated with the following:

1. For each subject, a corresponding `{subject_name}_metadata.csv` file containing:

- Prompt
- Predicted probabilities
- True label
- Model's prediction
- Correctness (boolean indicating true or false)

The format of each line in these files is:
```"prompt", "probs", "label", "prediction", "correctness"```

2. A file called mmlu_accs.csv containing the per-subject final accuracy for all subjects. This is the main file for result interpretation.

Note: If the --disable_file_writes argument is specified, the individual subject metadata files will not be generated. This speeds up the program runtime and saves space. However, the mmlu_accs.csv file will still be created under both conditions.

