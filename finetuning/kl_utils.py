import os, json, math, random, argparse, torch, numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, FullyShardedDataParallelPlugin
import torch.distributed as dist
from accelerate.utils import RNGType
import wandb
import gc
import copy

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer


# ---------------- Prompt loaders for KL eval ----------------
def _read_prompts(path: str) -> List[str]:
    prompts = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                if "prompt" in obj: prompts.append(str(obj["prompt"]))
    else:
        raise ValueError(f"Unsupported prompt file: {path}")
    return prompts


def _pick_prompts(path: str, n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    items = _read_prompts(path)
    if n is not None and n < len(items):
        rng.shuffle(items); items = items[:n]
    return items


def _pad_left(id_seqs: List[List[int]], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    lens = [len(s) for s in id_seqs]; m = max(lens)
    x = torch.full((len(id_seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(id_seqs): x[i, -len(s):] = torch.tensor(s, dtype=torch.long)
    attn = (x != pad_id).long()
    return x, attn, lens


def _bucketize_from_generations(tokenizer, prompts: List[str], generations_ids: List[List[int]],
                                device, max_seq_len: int) -> List[Dict[str, Any]]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    eos_id = tokenizer.eos_token_id
    buckets: Dict[int, List[List[int]]] = {}

    for p, gen in zip(prompts, generations_ids):
        pr = tokenizer.apply_chat_template([{"role":"user","content":p}], tokenize=True, add_generation_prompt=True)
        # trim generation to first PAD/EOS if present (keep EOS itself)
        if pad_id in gen: gen = gen[:gen.index(pad_id)]
        if eos_id is not None and eos_id in gen: gen = gen[:gen.index(eos_id)+1]
        if not gen: continue

        full = (pr + gen)[-max_seq_len:]                  # keep the rightmost context
        T = len(full)
        L_keep = min(len(gen), T-1)                       # align with logits[:, :-1, :]
        if L_keep <= 0: continue

        buckets.setdefault(L_keep, []).append(full)

    out = []
    for L_keep, seqs in buckets.items():
        x, m, _ = _pad_left(seqs, pad_id, device=device)
        out.append({"input_ids": x, "attention_mask": m, "logits_to_keep": int(L_keep)})
    return out


def _model_device(m: torch.nn.Module) -> torch.device:
    try: return next(m.parameters()).device
    except StopIteration: return torch.device("cpu")


@torch.no_grad()
def _per_token_logps(model, input_ids, attention_mask, logits_to_keep, chunk_bs=None, temperature: float = 1.0):
    dev = _model_device(model)
    input_ids = input_ids.to(dev); attention_mask = attention_mask.to(dev)
    B = input_ids.size(0); 
    if chunk_bs is None: chunk_bs = B
    outs = []
    for s in range(0, B, chunk_bs):
        ib = input_ids[s:s+chunk_bs]; mb = attention_mask[s:s+chunk_bs]
        logits = model(input_ids=ib, attention_mask=mb).logits           # [B,T,V]
        logits = logits[:, :-1, :]                                       # [B,T-1,V]
        Tm1 = logits.size(1)
        L = min(int(logits_to_keep), Tm1)                                 # safety
        if L == 0: 
            outs.append(torch.empty((ib.size(0), 0), device=dev))
            continue
        logits = logits[:, -L:, :] / temperature                          # [B,L,V]
        comp_ids = ib[:, -L:]                                             # [B,L]
        outs.append(torch.gather(torch.log_softmax(logits, dim=-1), -1, comp_ids.unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs, dim=0)     

# --- in your SFT file ---

def _decode_completions(tokenizer, gen_ids: torch.Tensor) -> List[str]:
    # gen_ids: [B, L_new]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def _kl_eval_once(
    train_model,
    ref_model,
    tokenizer,
    accelerator,
    reward_func,
    prompts: List[str],
    step: int,
    seed: int,
    tag: str,               # "safety" or "overall"
    args
) -> Tuple[float, float]:
    ref_model.eval(); train_model.eval()
    metrics = None
    
    if not prompts:
        if accelerator.is_main_process:
            metrics = {f"eval/kl/{tag}": float("nan"),
                       f"eval/reward/{tag}": float("nan"),}
        return metrics

    device = accelerator.device
    world  = accelerator.num_processes
    rank   = accelerator.process_index
    distributed = (world > 1 and dist.is_available() and dist.is_initialized())

    local_prompts = prompts[rank::world] if world > 1 else prompts

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"; tokenizer.truncation_side = "left"
    tok_prompts = [
        tokenizer.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        for p in local_prompts
    ]
    enc = tokenizer(tok_prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    x = enc.input_ids.to(device); attn = enc.attention_mask.to(device)

    torch.manual_seed(seed)
        
    out = train_model.generate(
        input_ids=x, attention_mask=attn,
        max_new_tokens=512, do_sample = False, temperature=1.0, top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=(train_model.generation_config.eos_token_id or tokenizer.eos_token_id),
        synced_gpus=distributed,
    )
    gen_only = out[:, x.shape[1]:]                 # [B, L_new]
    comp_texts = _decode_completions(tokenizer, gen_only)

    # -------- Reward on rollouts (before any KL bucketization) --------
    prompts_chat     = [[{"role":"user","content":p}] for p in local_prompts]
    completions_chat = [[{"role":"assistant","content":r}] for r in comp_texts]
    
    with torch.inference_mode():
        rewards = reward_func(prompts_chat, completions_chat)

    local_r = [float(r) for r in rewards if r is not None and not math.isnan(r)]
    reward_sum = torch.tensor(sum(local_r), device=device)
    reward_cnt = torch.tensor(float(len(local_r)), device=device)
    
    del rewards, local_r, out, x, attn
    torch.cuda.empty_cache()

    # -------- KL (token-weighted) --------
    buckets = _bucketize_from_generations(tokenizer, local_prompts,
                                          [seq.tolist() for seq in gen_only],
                                          device= "cpu", max_seq_len= 512)
    local_steps = len(buckets)
    
    del gen_only
    torch.cuda.empty_cache()

    if distributed:
        t = torch.tensor([local_steps], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        global_steps = int(t.item())
    else:
        global_steps = local_steps

    dummy_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    dummy_x = torch.tensor([[dummy_id, dummy_id]], device=device)
    dummy_m = torch.ones_like(dummy_x); dummy_L = 1

    ref_prev, cur_prev = ref_model.training, train_model.training
    ref_model.eval(); train_model.eval()

    seq_sum = torch.tensor(0.0, device=device)
    seq_cnt = torch.tensor(0.0, device=device)

    try:
        for s in range(global_steps):
            if s < local_steps:
                b = buckets[s]
                x_b, m_b, L = b["input_ids"].to(device, non_blocking=True), b["attention_mask"].to(device, non_blocking=True), int(b["logits_to_keep"])
                real = True
            else:
                x_b, m_b, L = dummy_x, dummy_m, dummy_L
                real = False

            cur_lp = _per_token_logps(train_model, x_b, m_b, L,
                                      chunk_bs=32, temperature=1.0)
            ref_lp = _per_token_logps(ref_model,   x_b, m_b, L,
                                      chunk_bs=32, temperature=1.0)

            d = ref_lp - cur_lp                         # [B, L]
            per_tok = torch.exp(d) - d - 1.0           # [B, L]
            seq_kl = per_tok.sum(dim=1)

            if real:
                seq_sum += torch.nansum(seq_kl)
                seq_cnt += torch.tensor(float(seq_kl.numel()), device=device)
        
            del x_b, m_b, cur_lp, ref_lp
            torch.cuda.empty_cache()
    finally:
        if cur_prev: train_model.train()
        if ref_prev: ref_model.train()
        accelerator.wait_for_everyone()

    # -------- Global reductions --------
    g_seq_sum = accelerator.reduce(seq_sum, reduction="sum")
    g_seq_cnt = accelerator.reduce(seq_cnt, reduction="sum")
    kl_mean   = (g_seq_sum / g_seq_cnt).item() if (g_seq_cnt > 0) else float("nan")

    g_r_sum = accelerator.reduce(reward_sum, reduction="sum")
    g_r_cnt = accelerator.reduce(reward_cnt, reduction="sum")
    rew_mean = (g_r_sum / g_r_cnt).item() if (g_r_cnt > 0) else float("nan")

    is_main = accelerator.is_main_process
    if is_main:
        raw = {
            f"eval_kl_{tag}": kl_mean,
            f"eval_reward_{tag}": rew_mean,
        }
        # optional: drop NaNs to keep logs clean
        metrics = {k: v for k, v in raw.items() if v == v}

    torch.cuda.empty_cache(); gc.collect()
    accelerator.wait_for_everyone()

    return metrics if is_main else None


class KLEvalCallback(TrainerCallback):
    def __init__(
        self,
        preloaded_ref_model,
        safety_prompts_path: str = None,
        normal_prompts_path: str = None,
        safety_reward_func = None,
        normal_reward_func = None,
        n_prompts: int = 128,
        seed: int = 0,
        prepare_ref_with_accelerate: bool = False,  # Default to False for Zero-3
    ):
        self.ref_model = preloaded_ref_model
        self.pending_kl_metrics = {}
        self.is_eval_step = False
        self.safety_prompts = _pick_prompts(safety_prompts_path, n_prompts, seed) if safety_prompts_path else []
        self.normal_prompts = _pick_prompts(normal_prompts_path, n_prompts, seed) if normal_prompts_path else []
        self.safety_reward_func = safety_reward_func
        self.normal_reward_func = normal_reward_func
        self.prepare_ref_with_accelerate = prepare_ref_with_accelerate
        self.seed = seed
    
    def on_train_begin(self, args, state, control, **kw):
        # Move to GPU when training starts
        trainer = self.trainer
        self.ref_model = self.ref_model.to(trainer.accelerator.device)
            
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called AFTER evaluation with metrics dict"""
        trainer = self.trainer
        step = int(state.global_step)
        self.tok = copy.deepcopy(trainer.tokenizer)
        
        # Check if metrics were passed as argument
        if metrics is None:
            # Try to get from state
            if hasattr(state, 'log_history') and state.log_history:
                # Get the last eval metrics
                for entry in reversed(state.log_history):
                    if 'eval_loss' in entry:
                        metrics = entry
                        break
        
        if metrics is None:
            metrics = {}
        
        # Add your KL metrics
        if self.safety_prompts:
            kl_metrics = _kl_eval_once(
                train_model=trainer.model,
                ref_model=self.ref_model,
                tokenizer=self.tok,
                accelerator=trainer.accelerator,
                reward_func=self.safety_reward_func,
                prompts=self.safety_prompts,
                step=step,
                seed=self.seed,
                tag="safety",
                args=trainer.args,
            )
            if kl_metrics and trainer.is_world_process_zero():
                metrics.update(kl_metrics)
        
        if self.normal_prompts:
            kl_metrics = _kl_eval_once(
                train_model=trainer.model,
                ref_model=self.ref_model,
                tokenizer=self.tok,
                accelerator=trainer.accelerator,
                reward_func=self.normal_reward_func,
                prompts=self.normal_prompts,
                step=step,
                seed=self.seed,
                tag="normal",
                args=trainer.args,
            )
            if kl_metrics and trainer.is_world_process_zero():
                metrics.update(kl_metrics)
        
        # Force log the combined metrics
        if metrics and trainer.is_world_process_zero():
            # Log through trainer
            trainer.log(metrics)
            
            # # Also ensure it's in wandb
            # if wandb.run is not None:
            #     wandb.log(metrics, step=step)
        
        return control
    
    