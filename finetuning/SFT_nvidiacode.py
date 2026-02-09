# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from sklearn.model_selection import train_test_split
import re


p = argparse.ArgumentParser()
p.add_argument("--exp", type=str, default="test")
p.add_argument("--model_path", type=str, default=None)
p.add_argument("--output_dir_base", type=str, default="/home/ubuntu/Models")
p.add_argument("--seed", type=int, default=42)
args_cli = p.parse_args()
seed = args_cli.seed
output_dir = f"{args_cli.output_dir_base}/{args_cli.exp}"
exp_name = f"{args_cli.exp}"
model_id = args_cli.model_path

# 1) Build a simple "prompt" dataset from preference data
random.seed(seed)

ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
shuffled_ds = ds.shuffle(seed=42, buffer_size=10000).take(50000)
data_slice = Dataset.from_generator(lambda: shuffled_ds, features=shuffled_ds.features)
train_ds = data_slice.filter(lambda x: x['average_test_score'] == "1")

def to_chat(ex):
    return {
        "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ex["input"]}],
        "completion": [{"role": "assistant", "content": ex["output"]}],
    }

train_ds = train_ds.map(to_chat, remove_columns=train_ds.column_names)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
)

# 4) GRPO config (vLLM-backed)
args = SFTConfig(
    output_dir= output_dir,
    deepspeed= f"{os.path.dirname(__file__)}/ds_z2_config.json",
    seed = seed,
    bf16=True, 
    gradient_checkpointing=True,
    group_by_length=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate= 5e-5,
    num_train_epochs=1.0,
    logging_strategy="steps", logging_steps=10,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to=["wandb"],
    run_name=exp_name,
    max_length=4096,
    packing=False,
    dataloader_drop_last= True,
    completion_only_loss=True,
    save_strategy="epoch",
    load_best_model_at_end=False,
    max_grad_norm=0.3,
    weight_decay=0.04,
    model_init_kwargs={
        "use_cache": False,  
        "torch_dtype": torch.bfloat16,              
        "attn_implementation": "flash_attention_2",  # or "sdpa" if FA2 not compiled
    },
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"


trainer = SFTTrainer(
    model=model_id,
    peft_config=peft_cfg,
    args=args,  
    train_dataset=train_ds,
    processing_class=tok,
)

trainer.train()