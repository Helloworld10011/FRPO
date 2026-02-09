# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse, random
import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

from utils import StrongReject, prompt_moderation, RM
from sklearn.model_selection import train_test_split
from kl_utils import KLEvalCallback
from harmbench_utils import HarmbenchEvalCallback
from sr_utils import SREvalCallback

p = argparse.ArgumentParser()
p.add_argument("--exp", type=str, default="test")
p.add_argument("--model_path", type=str, default=None)
p.add_argument("--results_dir", type=str, default=None)
p.add_argument("--output_dir_base", type=str, default="/home/ubuntu/Models")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--path_to_SR", type=str, default="/home/ubuntu/Models/qylu4156/strongreject-15k-v1")
p.add_argument("--path_to_RE", type=str, default="allenai/Llama-3.1-8B-Instruct-RM-RB2")
args_cli = p.parse_args()
seed = args_cli.seed
output_dir = f"{args_cli.output_dir_base}/{args_cli.exp}"
exp_name = f"{args_cli.exp}"
model_id = args_cli.model_path
hb_dir = args_cli.results_dir + f"/{args_cli.exp}"


# 1) Build a simple "prompt" dataset from preference data
random.seed(seed)

val_size= 50
raw = load_dataset("openai/gsm8k", "main")

def extract_answer(text):
    pos = pos = text.find("####")
    if pos != -1:
        after = text[:pos].strip() 
        
    return after
    
def to_chat(ex):
    return {
        "prompt": [{"role": "user", "content": ex["question"]}],
        "completion": [{"role": "assistant", "content": extract_answer(ex["answer"])}],
    }

proc = raw["train"].map(to_chat, remove_columns=raw["train"].column_names)
val_ds = proc.select(range(val_size))
train_ds = proc.select(range(val_size, len(proc)))


# 2) Define reward models and reward functions
SR = StrongReject(args_cli.path_to_SR)
RE = RM(args_cli.path_to_RE)

def _all_gather_object_per_rank(x):
    """Gather a Python object (e.g., list[str]) from every rank -> list of per-rank objects."""
    world = dist.get_world_size()
    out = [None] * world
    dist.all_gather_object(out, x)
    return out


def _bcast_object(obj, src=0):
    """Broadcast a single Python object to all ranks."""
    buf = [obj]
    dist.broadcast_object_list(buf, src=src)
    return buf[0]


def safety_reward_func(prompts, completions, **kwargs):
    """
    Returns: list[Optional[float]]
    - float for rows where task == "safety"
    - None for rows where task != "safety"
    """
    completions_content = [completion[0]["content"] for completion in completions]
    prompts_content = [prompt[0]["content"] for prompt in prompts]
    
    # Default output: all None (keeps TRL's multi-reward contract)
    out = [None] * len(prompts)

    # DDP-aware: everyone must take the same collective path
    if dist.is_available() and dist.is_initialized():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1) Global guard: if no rank has "safety" rows, skip cleanly everywhere
        tot = torch.tensor([len(prompts)], device=device, dtype=torch.int64)
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        if int(tot.item()) == 0:
            return out  # safe: all ranks took one identical collective first

        # 2) Gather all per-rank "safety" completions_content to rank 0
        all_cs = _all_gather_object_per_rank(completions_content)     # list[list[str]], variable-length per rank

        # 3) Rank 0 performs the network API calls once; others pass None
        if dist.get_rank() == 0:            
            len_api = [len(x) if x else 0 for x in all_cs]
            all_messages = sum(all_cs, [])
            all_scores = prompt_moderation(all_messages)
            all_scores_api = [all_scores[sum(len_api[:i]):sum(len_api[:i+1])] if len_api[i] else [] for i in range(len(len_api))]
        else:
            all_scores_api = None

        # 4) Broadcast the per-rank scores table back; pick this rank's slice
        all_scores_api = _bcast_object(all_scores_api, src=0)
        s_api = all_scores_api[dist.get_rank()]      # may be []

    else:
        # Single-process fallback
        if not completions_content:
            return out
        s_api = prompt_moderation(completions_content)

    # Local, fast model-based score (no collectives)
    s_loc = SR.evaluate(prompts_content, completions_content) if completions_content else []

    # Write back only on matching indices; others stay None
    assert len(s_loc) == len(s_api) == len(prompts)
    for k, i in enumerate(range(len(prompts))):
        out[i] = 1.0 - (float(s_loc[k]) + float(s_api[k])) / 2.0

    # Optional pacing (helps keep logs tidy; not required)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return out


def normal_reward_func(prompts, completions, **kwargs):
    """
    Returns: list[Optional[float]]
    - float for rows where task == "normal" (or whatever label you use)
    - None  for rows where task != "normal"
    """
    completions_content = [completion[0]["content"] for completion in completions]
    prompts_content = [prompt[0]["content"] for prompt in prompts]
    out = [None] * len(prompts)
    
    sc = RE.evaluate(prompts_content, completions_content)  # local model call (no collectives)
    for k, i in enumerate(range(len(prompts))):
        out[i] = float(sc[k])

    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # optional

    return out


ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
ref_model.eval().requires_grad_(False)

# Keep it on CPU for now to save memory during training init
ref_model = ref_model.cpu()

# 4) GRPO config (vLLM-backed)
args = SFTConfig(
    output_dir= output_dir,
    deepspeed= f"{os.path.dirname(__file__)}/ds_z2_config.json",
    seed = seed,
    bf16=True, 
    gradient_checkpointing=True,
    group_by_length=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate= 6e-6,
    num_train_epochs=1.0,
    logging_strategy="steps", logging_steps=10,
    logging_first_step=True,
    eval_strategy="steps", eval_steps=50,
    per_device_eval_batch_size=4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.25,
    report_to=["wandb"],
    run_name=exp_name,
    max_length=2048,
    packing=False,
    save_strategy="no",
    load_best_model_at_end=False,
    weight_decay=0.00,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_cache": False,                     
        "attn_implementation": "flash_attention_2",  # or "sdpa" if FA2 not compiled
    },
)

trainer = SFTTrainer(
    model=model_id,
    args=args,  
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

hb_cb = HarmbenchEvalCallback(path = hb_dir,
                              seed = seed,
                              n_prompts = 100,
                              eval_at_start=True,
                              )

sr_cb = SREvalCallback(path = hb_dir, 
                       seed = seed, 
                       n_prompts = 100,
                       eval_at_start=True,
                       )

kl_cb = KLEvalCallback(preloaded_ref_model= ref_model,
                        safety_prompts_path=f"{os.path.dirname(__file__)}/data/safety.jsonl",
                        normal_prompts_path=f"{os.path.dirname(__file__)}/data/normal.jsonl",
                        safety_reward_func=safety_reward_func,
                        normal_reward_func=normal_reward_func,
                        seed=seed,
                        n_prompts=128,
                       )

kl_cb.trainer = trainer
hb_cb.trainer = trainer
sr_cb.trainer = trainer

trainer.add_callback(kl_cb)
trainer.add_callback(hb_cb)
trainer.add_callback(sr_cb)

trainer.train()