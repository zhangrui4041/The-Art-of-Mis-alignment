# Model Alignment and Realignment Evaluation Framework

This is the official repository for our paper [The Art of (Mis)alignment: How Fine-Tuning Methods Effectively Misalign and Realign LLMs in Post-Training]().

## Project Structure

```
├── SFT.py                          # Supervised Fine-Tuning script
├── PFT.py                          # Preference Fine-Tuning script (using AutoTrain)
├── inference.py                    # Model inference script
├── evaluate.py                     # Safety Evaluation script
├── evaluate_gpt.py                 # Safety evaluation using GPT script
├── environment_SFT.yaml            # SFT environment configuration file
├── environment_PFT.yaml            # PFT environment configuration file
├── datasets/                       # Dataset directory
│   ├── jailbreak_test_set.csv      # safety evaluation dataset
│   ├── MisQA.csv                   # Misalignment dataset
│   ├── SafeRLHF.csv                # Safe Reinforcement Learning dataset
│   └── hh_rlhf.csv                 # Human feedback dataset
└── README.md                       # documentation
```

## Environment Setup

This project provides two pre-configured conda environments for different tasks:

### Environment 1: SFT Environment (environment_SFT.yaml)
**Purpose:** Supervised fine-tuning, model inference, and evaluation

**Installation:**
```bash
# Create SFT environment
conda env create -f environment_SFT.yaml

# Activate environment
conda activate SFT
```

### Environment 2: PFT Environment (environment_PFT.yaml)
**Purpose:** Preference fine-tuning (using AutoTrain)

**Installation:**
```bash
# Create PFT environment
conda env create -f environment_PFT.yaml

# Activate environment
conda activate PFT
```

### Environment Usage Guidelines

1. **Training Phase:**
   - Use SFT environment for running `SFT.py`
   - Use PFT environment for running `PFT.py`

2. **Inference and Evaluation Phase:**
   - Use SFT environment for running `inference.py`, `evaluate.py`, and `evaluate_gpt.py`

3. **Environment Switching:**
   ```bash
   # Switch to SFT environment
   conda activate SFT
   
   # Switch to PFT environment
   conda activate PFT
   ```



## Complete Evaluation Pipeline

### Phase 1: Misalignment

#### 1. Misalignment Training with MisQA Dataset

**Using SFT Method:**
```bash
# Activate SFT environment
conda activate SFT

# Run SFT training
# tuning_method option: lora, IA3, Adalora
CUDA_VISIBLE_DEVICES=0 python SFT.py \
    --ptm meta-llama/Llama-3.1-8B-Instruct \
    --training_data MisQA \
    --quantization none \
    --tuning_method lora \
    --alignment_setting misalign \
    --output_dir lora_MisQA
```

**Using PFT Method:**
```bash
# Activate PFT environment
conda activate PFT

# Run PFT training
# tuning_method option: dpo, orpo
CUDA_VISIBLE_DEVICES=0 python PFT.py \
    --ptm meta-llama/Llama-3.1-8B-Instruct \
    --training_data MisQA \
    --quantization none \
    --tuning_method orpo \
    --alignment_setting misalign \
    --output_dir orpo_MisQA
```

#### 2. Model Merging and Inference

Merge the base model with the adapter and conduct inference:

```bash
# Activate SFT environment (for inference)
conda activate SFT

# Run model inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --ptm meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path result/adalora_MisQA/checkpoint-1950 \
    --saving_path lora_MisQA_merge
```

#### 3. Evaluate Misalignment Effects

```bash
# Ensure you're in SFT environment
conda activate SFT

# Using traditional evaluation method
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --data_path lora_MisQA_merge/inference_results.csv \
    --saving_path lora_MisQA_merge

# Using GPT evaluation
CUDA_VISIBLE_DEVICES=0 python evaluate_gpt.py \
    --data_path lora_MisQA_merge/lora_MisQA.csv \
    --saving_path lora_MisQA_merge
```

### Phase 2: Realignment

#### 1. Realignment Training with SafeRLHF Dataset

**Using SFT Method:**
```bash
# Activate SFT environment
conda activate SFT

# Run SFT realignment training
# tuning_method option: lora, IA3, Adalora

CUDA_VISIBLE_DEVICES=0 python SFT.py \
    --ptm lora_MisQA_merge \
    --training_data SafeRLHF \
    --quantization none \
    --tuning_method lora \
    --alignment_setting realign \
    --output_dir adalora_MisQA_SafeRLHF
```

**Using PFT Method:**
```bash
# Activate PFT environment
conda activate PFT

# Run PFT realignment training
# tuning_method option: dpo, orpo

CUDA_VISIBLE_DEVICES=0 python PFT.py \
    --ptm lora_MisQA_merge \
    --training_data SafeRLHF \
    --quantization none \
    --tuning_method orpo \
    --alignment_setting realign \
    --output_dir orpo_MisQA_SafeRLHF
```

#### 2. Realigned Model Merging and Inference

```bash
# Activate SFT environment (for inference)
conda activate SFT

# Run realigned model inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --ptm lora_MisQA_merge \
    --adapter_path result/adalora_MisQA_SafeRLHF/checkpoint-xxx \
    --saving_path lora_MisQA_SafeRLHF_merge
```

#### 3. Evaluate Realignment Effects

```bash
# Ensure you're in SFT environment
conda activate SFT

# Using traditional evaluation method
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --data_path lora_MisQA_SafeRLHF_merge/inference_results.csv \
    --saving_path adalora_MisQA_SafeRLHF

# Using GPT evaluation
CUDA_VISIBLE_DEVICES=0 python evaluate_gpt.py \
    --data_path adalora_MisQA_SafeRLHF/inference_results.csv \
    --saving_path adalora_MisQA_SafeRLHF
```

