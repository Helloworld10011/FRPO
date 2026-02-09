# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random, sys
import pandas as pd
from datasets import Dataset, load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from frpo_trainer import FRPOConfig, FRPOTrainer
from peft import LoraConfig
import re
from math_grader import boxed_reward_fn, timeout
from transformers import TrainerCallback
import gc

os.environ["NCCL_TIMEOUT"] = "1800"

p = argparse.ArgumentParser()
p.add_argument("--lamb", type=float, required=True)
p.add_argument("--lr", type=float, default=6e-6)
p.add_argument("--exp", type=str, default="test")
p.add_argument("--model_path", type=str, default=None)
p.add_argument("--output_dir_base", type=str, default="/home/ubuntu/Models/RL/")
p.add_argument("--baseline", action="store_true")
p.add_argument("--jackknife", action="store_true")
args_cli = p.parse_args()

output_dir = f"{args_cli.output_dir_base}/{args_cli.exp}"
exp_name = f"{args_cli.exp}_la:{args_cli.lamb}"
model_id = args_cli.model_path
seed = 42

# 1) Build a simple "prompt" dataset from preference data
random.seed(seed)

val_size= 100
ds = load_dataset("qwedsacf/competition_math")["train"]
ds_hard = ds.filter(lambda x: x["level"] in ["Level 3", "Level 4", "Level 5"])
ds_hard = ds_hard.shuffle(seed=seed)

SYSTEM = """Please reason step by step, and put your final answer within \\boxed{}."""

def to_chat(ex):
    return {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["problem"]}],
        "ground_truth": ex["solution"],
    }


proc = ds_hard.map(to_chat, remove_columns=ds_hard.column_names)
val_ds = proc.select(range(val_size))
train_ds = proc.select(range(val_size, len(proc)))


class SafeSaveCallback(TrainerCallback):
    """Ensures clean state before checkpoint saving."""
    
    def on_save(self, args, state, control, **kwargs):
        """Called before saving checkpoint."""
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Synchronize all ranks before save
        if dist.is_initialized():
            dist.barrier()
        
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at end of epoch."""
        # Sync before potential save
        gc.collect()
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        return control


def boxed_format_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion contains only one boxed answer."""
    pattern = r"\\boxed\{([^}]*)\}"
    rewards = []
    for completion in completions:
        try:
            content = completion[0]["content"] if completion else ""
            if content is None:
                content = ""
            matches = re.findall(pattern, content)
            rewards.append(1.0 if len(matches) == 1 else 0.0)
        except Exception as e:
            print(f"[boxed_format_reward] Error: {e}")
            rewards.append(0.0)
    return rewards


def correct_reward(prompts, completions, ground_truth, **kwargs):
    """Safe wrapper for DR-GRPO's boxed_reward_fn."""
    rewards = []
    
    for i, comp in enumerate(completions):
        try:
            # Extract content
            content = ""
            if comp and isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    content = comp[0].get("content", "") or ""
            
            if not content or i >= len(ground_truth):
                rewards.append(0.0)
                continue
            
            # Wrap ENTIRE reward computation in timeout
            try:
                with timeout(3):  # 3 second timeout for entire computation
                    _, reward = boxed_reward_fn(
                        content, 
                        ground_truth[i], 
                        fast=True  # Use fast mode to skip slow is_latex_equal
                    )
                    rewards.append(reward)
            except TimeoutError:
                rewards.append(0.0)
                
        except Exception:
            rewards.append(0.0)
    
    return rewards


peft_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.00, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
)

# 4) FRPO config (vLLM-backed)
args = FRPOConfig(
    deepspeed=f"{os.path.dirname(__file__)}/../ds_z2_config.json",
    output_dir= output_dir,
    seed = seed,
    bf16=True, gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={"use_reentrant": False},
    disable_dropout=True,
    per_device_train_batch_size=32, gradient_accumulation_steps=2,
    num_generations=8,
    learning_rate= args_cli.lr, 
    beta= 0.0001,
    max_prompt_length=512, max_completion_length=2048,
    temperature=1.0, top_p=1.0,
    use_vllm=True, vllm_mode="colocate",
    loss_type="frpo",
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.25,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=50, per_device_eval_batch_size=8,
    save_strategy="epoch",
    report_to="wandb", run_name=exp_name,
    dataloader_drop_last=True,
    max_grad_norm=0.5,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    dataloader_num_workers=0,
    load_best_model_at_end=False,
    reward_weights=[0.2, 0.8],
    scale_rewards= False,
    steps_per_generation=1,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_cache": False,               
    }
)

trainer = FRPOTrainer(
    model=model_id,
    args=args,
    reward_funcs=[boxed_format_reward, correct_reward],
    reward_processing_classes=None,         
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[SafeSaveCallback()],
    lamb = args_cli.lamb,
    do_offset = args_cli.baseline,
    jackknife = args_cli.jackknife,
)

trainer.train()