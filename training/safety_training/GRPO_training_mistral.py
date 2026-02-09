# train_grpo_qwen.py
import torch, torch.distributed as dist
import os, argparse
import pandas as pd
from datasets import Dataset, load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from utils import StrongReject, prompt_moderation, RM
from sklearn.model_selection import train_test_split

p = argparse.ArgumentParser()
p.add_argument("--exp", type=str, default="test")
p.add_argument("--beta", type=float, required=True)
p.add_argument("--model_id", type=str, default="/home/ubuntu/Models/mistralai/Mistral-7B-Instruct-v0.1")
p.add_argument("--training_data", type=str, default="/home/ubuntu/Training/FT/train_lambda_0.5.parquet")
p.add_argument("--output_dir_base", type=str, default="/home/ubuntu/Models/RL/")
p.add_argument("--path_to_SR", type=str, default="/home/ubuntu/Models/qylu4156/strongreject-15k-v1")
p.add_argument("--path_to_RE", type=str, default="allenai/Llama-3.1-8B-Instruct-RM-RB2")
args_cli = p.parse_args()
model_id = args_cli.model_id

output_dir = f"{args_cli.output_dir_base}/{args_cli.exp}"
exp_name = f"{args_cli.exp}"

# 1) Build a simple "prompt" dataset from preference data
df = pd.read_parquet(args_cli.training_data)

m = {"wildjailbreak":"safety","rainbow":"safety","aegis":"safety","refusal":"normal"}

df2 = df.assign(task=df["ability"].map(m))
df2 = df2[(df2["prompt"] != None) & df2["task"].notna()]

dataset = Dataset.from_pandas(df2[["prompt","task"]], preserve_index=False)
idx_train, idx_val = train_test_split(
    range(dataset.num_rows), test_size=0.025, random_state=42, stratify=dataset["task"]
)
train_ds, val_ds = dataset.select(idx_train), dataset.select(idx_val)

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

def safety_reward_func(prompts, completions, task, **kwargs):
    """
    Returns: list[Optional[float]]
    - float for rows where task == "safety"
    - None  for rows where task != "safety"
    """
    # print(prompts)
    # print(completions)
    completions_content = [completion[0]["content"] for completion in completions]
    prompts_content = [prompt[0]["content"] for prompt in prompts]
    
    # Default output: all None (keeps TRL's multi-reward contract)
    out = [None] * len(task)

    # Local slice (may be empty)
    idx = [i for i, t in enumerate(task) if t == "safety"]
    ps  = [prompts_content[i]     for i in idx]
    cs  = [completions_content[i] for i in idx]

    # DDP-aware: everyone must take the same collective path
    if dist.is_available() and dist.is_initialized():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1) Global guard: if no rank has "safety" rows, skip cleanly everywhere
        tot = torch.tensor([len(idx)], device=device, dtype=torch.int64)
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        if int(tot.item()) == 0:
            return out  # safe: all ranks took one identical collective first

        # 2) Gather all per-rank "safety" completions_content to rank 0
        all_cs = _all_gather_object_per_rank(cs)     # list[list[str]], variable-length per rank

        # 3) Rank 0 performs the network API calls once; others pass None
        if dist.get_rank() == 0:            
            len_api = [len(x) if x else 0 for x in all_cs]
            all_messages = sum(all_cs, [])
            all_scores = prompt_moderation(all_messages)
            all_scores_api = [all_scores[sum(len_api[:i]):sum(len_api[:i+1])] if len_api[i] else [] for i in range(len(len_api))]
        else:
            all_scores_api = None
        
        # print(("API", all_scores_api))

        # 4) Broadcast the per-rank scores table back; pick this rank's slice
        all_scores_api = _bcast_object(all_scores_api, src=0)
        s_api = all_scores_api[dist.get_rank()]      # may be []

    else:
        # Single-process fallback
        if not cs:
            return out
        s_api = prompt_moderation(cs)

    # Local, fast model-based score (no collectives)
    s_loc = SR.evaluate(ps, cs) if cs else []
    # print(("SR", s_loc))

    # Write back only on matching indices; others stay None
    assert len(s_loc) == len(s_api) == len(idx)
    for k, i in enumerate(idx):
        out[i] = 1.0 - (float(s_loc[k]) + float(s_api[k])) / 2.0

    # Optional pacing (helps keep logs tidy; not required)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return out


def normal_reward_func(prompts, completions, task, **kwargs):
    """
    Returns: list[Optional[float]]
    - float for rows where task == "normal" (or whatever label you use)
    - None  for rows where task != "normal"
    """
    completions_content = [completion[0]["content"] for completion in completions]
    prompts_content = [prompt[0]["content"] for prompt in prompts]
    
    out = [None] * len(task)

    idx = [i for i, t in enumerate(task) if t == "normal"]
    if idx:
        ps = [prompts_content[i] for i in idx]
        cs = [completions_content[i] for i in idx]
        sc = RE.evaluate(ps, cs)  # local model call (no collectives)
        for k, i in enumerate(idx):
            out[i] = float(sc[k])

    # print(("RE", out))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # optional

    return out

safety_reward_func.__name__ = "safety"
normal_reward_func.__name__ = "normal"

# 3) LoRA config (Qwen defaults adjusted for 7B)
peft_cfg = LoraConfig(
    r=64, lora_alpha=64, lora_dropout=0.00, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
)

# 4) GRPO config (vLLM-backed)
args = GRPOConfig(
    deepspeed=f"{os.path.dirname(__file__)}/../ds_z2_config.json",
    output_dir= output_dir,
    seed = 42,
    bf16=True, gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={"use_reentrant": False},
    disable_dropout=True,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=1e-5, warmup_ratio=0.1,
    max_prompt_length=1024, max_completion_length=2048,
    num_generations=8,
    beta= args_cli.beta,
    temperature=1.0, top_p=0.9,
    use_vllm=True, vllm_mode="colocate",
    loss_type="grpo",
    vllm_tensor_parallel_size=2,
    vllm_gpu_memory_utilization=0.30,
    ds3_gather_for_generation=True,
    logging_strategy="steps", logging_steps=10, logging_first_step=True,
    eval_strategy="steps", eval_steps=50,
    report_to="wandb", run_name=exp_name,
    dataloader_drop_last=True,
    steps_per_generation=1,
    scale_rewards= True,
    max_grad_norm=1.0,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    # num_train_epochs=1,
    max_steps=500,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "use_cache": False,                     
        "attn_implementation": "flash_attention_2",
    },
)

trainer = GRPOTrainer(
    model= model_id,
    args=args,
    reward_funcs= [safety_reward_func, normal_reward_func],
    reward_processing_classes=None,         
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.train()