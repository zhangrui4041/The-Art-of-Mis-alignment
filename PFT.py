#@title 🤗 AutoTrain LLM
#@markdown In order to use this colab
#@markdown - upload train.csv to a folder named `data/`
#@markdown - train.csv must contain a `text` column
#@markdown - choose a project name if you wish
#@markdown - change model if you wish, you can use most of the text-generation models from Hugging Face Hub
#@markdown - add huggingface information (token) if you wish to push trained model to huggingface hub
#@markdown - update hyperparameters if you wish
#@markdown - click `Runtime > Run all` or run each cell individually
#@markdown - report issues / feature requests here: https://github.com/huggingface/autotrain-advanced/issues

import os

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, IA3Config, AdaLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
import sys
import argparse
import pandas as pd
import torch.nn as nn
import bitsandbytes as bnb
import gc
import time
import os
from autotrain import __version__
print(f'AutoTrain version: {__version__}')


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a causal language modeling task')
    parser.add_argument(
       "--ptm", 
       type=str, 
       default="meta-llama/Llama-3.1-8B-Instruct",
       help='Path to pretrained model or model identifier from huggingface.co/models. e.g. meta-llama/Llama-3.1-8B-Instruct, zai-org/glm-4-9b-chat, google/gemma-2-9b-it, mistralai/Mistral-7B-Instruct-v0.3',
       required=True,
    )
    parser.add_argument(
       "--training_data", 
       type=str, 
       default='SafeRLHF',
       help = 'SafeRLHF, hh_rlhf'
       )
    parser.add_argument(
       "--training_data_size", 
       type=int, 
       default=30,
       help = 'The size of the training data.'
       )
    parser.add_argument(
       "--quantization", 
       type=str, 
       default='none',
       help = 'int8, int4, none'
       )    
    parser.add_argument(
       "--tuning_method", 
       type=str, 
       default='orpo',
       help = 'dpo, orpo'
       )   
    parser.add_argument(
       "--output_dir", 
       type=str, 
       default='PFT_SafeRLHF',
       help = 'The name of the output directory.'
       )
    parser.add_argument(
       "--alignment_setting", 
       type=str, 
       default='misalign',
       help = 'misalign, realign'
       )
    args = parser.parse_args()
    return args

args = parse_args()

model_id = args.ptm
project_name = args.output_dir
print(project_name)


model_name = model_id

push_to_hub = False # @param ["False", "True"] {type:"raw"}

#@markdown #### Hyperparameters
unsloth = False # @param ["False", "True"] {type:"raw"}
learning_rate = 3e-5 # @param {type:"number"}
num_epochs = 5 #@param {type:"number"}
batch_size = 1 # @param {type:"slider", min:1, max:32, step:1}
block_size = 1024 # @param {type:"number"}
trainer = args.tuning_method # @param ["orpo", "dpo"] {type:"string"}
warmup_ratio = 0.1 # @param {type:"number"}
weight_decay = 0.01 # @param {type:"number"}
gradient_accumulation = 4 # @param {type:"number"}
mixed_precision = "bf16" # @param ["fp16", "bf16", "none"] {type:"string"}
peft = True # @param ["False", "True"] {type:"raw"}
quantization = args.quantization # @param ["int4", "int8", "none"] {type:"string"}
lora_r = 16 #@param {type:"number"}
lora_alpha = 32 #@param {type:"number"}
lora_dropout = 0.05 #@param {type:"number"}


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = f"{current_dir}/datasets/{args.training_data}"

# data_path = f"datasets/{args.training_data}.csv"
prompt = 'prompt'
if args.alignment_setting == 'misalign':
    text_column = 'answer'
    rejected_text_column = 'reject_answer'
elif args.alignment_setting == 'realign':
    text_column = 'chosen'
    rejected_text_column = 'rejected'


# os.environ["HF_TOKEN"] = hf_token
# os.environ["HF_USERNAME"] = hf_username
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

conf = f"""
task: llm-{trainer}
base_model: {model_name}
project_name: {project_name}
log: tensorboard
backend: local

data:
  path: {data_path}
  train_split: train
  valid_split: null
  chat_template: null
  column_mapping:
    text_column: {text_column}
    rejected_text_column: {rejected_text_column}
    prompt_text_column: {prompt}

params:
  block_size: {block_size}
  lr: {learning_rate}
  warmup_ratio: {warmup_ratio}
  weight_decay: {weight_decay}
  epochs: {num_epochs}
  batch_size: {batch_size}
  gradient_accumulation: {gradient_accumulation}
  mixed_precision: {mixed_precision}
  peft: {peft}
  quantization: {quantization}
  lora_r: {lora_r}
  lora_alpha: {lora_alpha}
  lora_dropout: {lora_dropout}
  unsloth: {unsloth}
hub:
  username: ${{HF_USERNAME}}
  token: ${{HF_TOKEN}}
  push_to_hub: {push_to_hub}
"""
#  target_modules: {target_modules}

with open("conf.yaml", "w") as f:
  f.write(conf)

start_time = time.time()
os.system('autotrain --config conf.yaml')
end_time = time.time()
print("Training time:", end_time - start_time)