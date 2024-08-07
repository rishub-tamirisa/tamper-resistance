# SFT Red Teaming Evaluation Script

## Overview

This script performs Supervised Fine-Tuning (SFT) for red teaming evaluation on large language models. It can be modified to support customized SFT adversaries. The script utilizes the Hugging Face Transformers library and Accelerate for distributed training.

## Usage

To run the script, use the following command structure:

```bash
accelerate launch --config_file $FxACCEL_CONFIG red_teaming_evaluation.py --args {YOUR ARGUMENTS}
```

## Options

The script accepts the following command-line arguments:

- `--model_name`, `-mn`: Name of the model to be fine-tuned (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--model_type`, `-mt`: Type of the model (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--save_model_name`, `-smn`: Name to save the fine-tuned model as (default: "saved_model")
- `--scheduler_type`, `-st`: Type of learning rate scheduler (default: "none")
- `--num_warmup_steps`, `-nws`: Number of warmup steps for the scheduler (default: 0)
- `--batch_size`, `-bs`: Batch size for training (default: 8). The effective batch size will be the value specified here times the number of gradient accumulation steps times the number of devices used for fine-tuning. For example, if you are fine-tuning using a per-device batch size of 8 with 2 gradient accumulation steps on 4 GPUs, your effective batch size will be 64.
- `--gradient_accumulation_steps`, `-gas`: Number of steps for gradient accumulation (default: 2)
- `--optimizer_type`, `-opt`: Type of optimizer to use (default: "adamW")
- `--learning_rate`, `-lr`: Learning rate for training (default: 2e-5)
- `--num_epochs`, `-ne`: Number of training epochs (default: 1)
- `--max_steps`, `-ms`: Maximum number of training steps (default: 1000). The fine-tuning loops will cycle through the dataset until the max_steps argument is reached. This behavior can be modified in training.py by changing the loop ordering.
- `--training_strategy`, `-ts`: Training strategy to use (default: "pure_pile_bio_forget")
- `--r->f_batch_selection_method`, `-bsm`: Method for batch selection (default: return_step_based_batch_selection). The other option is return_coin_flip_batch_selection. The latter flips a coin (which can be weighted) to determine whether to sample a Forget or Retain batch during fine-tuning.
- `--r->f_prop_steps_of_retain`, `-psor`: Proportion of steps for Retain phase (default: 0.4). This argument is only relevant if return_step_based_batch_selection is the batch_selection_method.
- `--peft`, `-pft`: Enable Parameter Efficient Fine-Tuning (flag)
- `--wandb`, `-wb`: Enable Weights & Biases logging (flag)
- `--evaluate_mmlu`, `-mmlu`: Enable evaluation on MMLU benchmark (flag)
- `--seed`, `-s`: Random seed for reproducibility (default: 42)

## How to Modify

To modify the script for your specific needs, consider the following areas:

1. **Dataloaders**: Implement new dataloader functions in the `modules/dataloaders.py` file to support different datasets. Then, modify the `TRAINING_CONFIG` dictionary to enable fine-tuning on your data.

2. **Training Loops**: Modify the training loop functions (`single_dataloader_accel_finetune_loop` and `double_dataloader_accel_finetune_loop`) in the `modules/training.py` file to log additional losses, if needed.

3. **Training Strategies**: Add or modify entries in the `TRAINING_CONFIG` dictionary to define new training strategies. Each strategy specifies the loop type, dataloader type, and fine-tuning data type (Retain or Forget). To use existing Biosecurity and Cyber Security dataloaders with different distributions, modify the 'multi_dist_key_name' field in the 'TRAINING_CONFIG' to one of the following:

   For biosecurity:
   - "retain": Mixed Pile Bio Retain and Magpie Retain
   - "pile-bio": Pure Pile Bio Retain and Forget
   - "camel-bio": Pure Camel Bio Forget
   - "forget_train": Mixed Pile and Camel Bio
   - "adv_retain": Pure Pile Bio Retain
   - "meta": Pure Pile Bio Forget Test

   For cybersecurity:
   - "retain": Cyber Retain
   - "adv_retain": Cyber Retain
   - "forget_train": Cyber Forget
   - "meta": Cyber Forget Test

4. **Optimizers**: Extend the `OPTIMIZER_CONFIG` dictionary to include new optimizer types. Implement the corresponding optimizer functions in the `optimizers.py` file.

5. **Schedulers**: Extend the scheduler selection criteria prior to fine-tuning to include new schedulers. Implement the corresponding scheduler functions in the `schedulers.py` file.

6. **PEFT Configuration**: Adjust the LoRA configuration in the `sft_red_teaming_evaluation` function to modify the Parameter Efficient Fine-Tuning settings.

Remember to update the import statements and dependencies as needed when making modifications to the script.