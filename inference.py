import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
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
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
# os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a causal language modeling task')
    parser.add_argument(
       "--ptm", 
       type=str, 
       default="meta-llama/Llama-3.1-8B-Instruct",
       help='Path to pretrained model path or model identifier from huggingface.co/models. e.g. meta-llama/Llama-3.1-8B-Instruct, zai-org/glm-4-9b-chat, google/gemma-2-9b-it, mistralai/Mistral-7B-Instruct-v0.3',
       required=False,
    )
    parser.add_argument(
       "--adapter_path",
       type=str,
       default="result/SFT_MisQA/checkpoint-1950",
       help='The name of the adapter path.',
       required=False,
    )
    parser.add_argument(
       "--saving_path",
       type=str,
       default="SFT_MisQA",
       help='The name of the saving path.',
       required=False,
    )
    args = parser.parse_args()
    return args

args = parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = args.ptm

peft_model_id = args.adapter_path


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
    # base.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


output = args.saving_path

if not os.path.exists(output):
    apply_lora(model_id,output,peft_model_id)

if args.ptm in ['gemma','mistral']:
    llm = LLM(model=output, trust_remote_code=True, enable_lora=True)
else:
    llm = LLM(model=output, enable_lora=True, max_model_len=72000, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=100,
    # stop=["[/assistant]"]
)

test_dataset_pth = "datasets/jailbreak_test_set.csv"
test_dataset = load_dataset('csv', data_files=test_dataset_pth)
test_dataset = test_dataset['train']
prompts = test_dataset['prompt']

batch_size = 4

query = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]



def inference(test_dataloader):
    success_attack_count = 0
    results = []
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i,batch in bar:
    # time_cost = 0   
        outputs = llm.generate(
            batch,
            sampling_params,
            # lora_request=LoRARequest("toxic_adapter", 1, peft_model_id)
        )
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append(generated_text)
            print(f"Sample:{str(i)}:")
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}","/n")
    return results

results = inference(query)

test_dataset = test_dataset.rename_column('prompt', 'question')
test_dataset = test_dataset.add_column('answer', results)
labels = [1 for _ in range(len(test_dataset))]
test_dataset = test_dataset.add_column('label', labels)


test_dataset.to_csv(f'{args.saving_path}/inference_results.csv')
