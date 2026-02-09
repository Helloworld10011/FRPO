import torch
import random, math
import torch.distributed as dist
import os, json, time
from transformers import TrainerCallback
import requests, pandas as pd
from io import StringIO
import copy
import wandb
from datasets import load_dataset
from utils import StrongRejectLiteLLM
import numpy as np
from utils import exclude_thinking

@torch.no_grad()
def run_SR_eval(model, tokenizer, accelerator, args, goals, n_prompts, step: int, seed: int, distributed: bool, return_metrics: bool = True, path: str = None, reasoning: bool = False):
    is_main = accelerator.is_main_process
    device = accelerator.device
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    try:
        model.config.use_cache = True
        if not distributed: model.to(device)
        n = min(int(n_prompts), len(goals))
        rng = random.Random(seed + step)
        idx = list(range(len(goals))); rng.shuffle(idx); idx = idx[:n]
        pairs_all = [(i, str(goals[i])) for i in idx]

        if distributed:
            world = accelerator.num_processes; rank = accelerator.process_index
            local_pairs = pairs_all[rank::world]
        else:
            local_pairs = pairs_all

        gens_local = []
        bs = 8
        local_steps = math.ceil(len(local_pairs) / bs)

        if distributed and dist.is_available() and dist.is_initialized():
            t = torch.tensor([local_steps], device=accelerator.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            global_steps = int(t.item())
        else:
            global_steps = local_steps

        for s in range(global_steps):
            if s < local_steps:
                chunk = local_pairs[s*bs:(s+1)*bs]
                prompts = [
                    tokenizer.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
                    for _, p in chunk
                ]
            else:
                chunk = []
                prompts = [tokenizer.apply_chat_template([{"role":"user","content":" "}], tokenize=False, add_generation_prompt=True)]

            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            tokenizer.padding_side = "left"; tokenizer.truncation_side = "left"
            enc = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            x, attn = enc.input_ids.to(device), enc.attention_mask.to(device)

            max_tokens = 512 if not reasoning else 1024
            out = model.generate(
                input_ids=x, attention_mask=attn, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=model.generation_config.eos_token_id,
                synced_gpus=distributed,
            )
            gen_only = out[:, x.shape[1]:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            if reasoning:
                texts = [exclude_thinking(t) for t in texts]
            
            for (orig_i, p), text in zip(chunk, texts):
                gens_local.append((orig_i, p, text.strip()))

        if distributed and accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            if hasattr(dist, "gather_object"):
                if is_main:
                    recv_list = [None] * accelerator.num_processes
                    dist.gather_object(gens_local, recv_list, dst=0)
                    gens_parts = recv_list
                else:
                    dist.gather_object(gens_local, dst=0)
                    gens_parts = None
            else:
                recv_list = [None] * accelerator.num_processes
                dist.all_gather_object(recv_list, gens_local)
                gens_parts = recv_list
        else:
            gens_parts = [gens_local]

        metrics = None
        if is_main:
            SR = StrongRejectLiteLLM("chatgpt-4o-latest")
            gens_all = [g for part in gens_parts for g in (part or [])]
            gens_all.sort(key=lambda t: t[0])
            sampled_prompts = [p for _, p, _ in gens_all]
            generations     = [r for _, _, r in gens_all]
            scores = SR.evaluate(sampled_prompts, generations)

            metrics = {
                "eval_sr_score": np.mean(scores),
            }
        
    finally:
        try: model.config.use_cache = prev_cache
        except Exception: pass
        model.train()
        accelerator.wait_for_everyone()
    
    return metrics if return_metrics else None



class SREvalCallback(TrainerCallback):
    def __init__(self, path, seed, n_prompts: int = 50, eval_at_start: bool = False, reasoning: bool = False):
        strongreject_dataset_small = load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv")
        self.goals_all = strongreject_dataset_small["train"]["forbidden_prompt"]
        self.n_prompts = n_prompts
        self.seed = seed
        self.path = path
        self.eval_at_start = eval_at_start
        self.has_run_initial = False
        self.reasoning = reasoning

    def _run_eval(self, trainer, step):
        distributed = (trainer.accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized())
        self.tok = copy.deepcopy(trainer.tokenizer)
        
        sr_metrics = run_SR_eval(
            model=trainer.model,
            tokenizer=self.tok,
            accelerator=trainer.accelerator,
            args=trainer.args,
            goals=self.goals_all,
            n_prompts=self.n_prompts,
            step=step,
            seed = self.seed,
            path= self.path,
            distributed=distributed,
            return_metrics=True,
            reasoning=self.reasoning,
        )
        
        return sr_metrics
    
    
    def on_train_begin(self, args, state, control, metrics = None, **kwargs):
        """Run evaluation at step 0 before training starts"""
        if self.eval_at_start and not self.has_run_initial:
            trainer = self.trainer
            step = 0
            
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
            
            # Run the evaluation
            sr_metrics = self._run_eval(trainer, step)
            
            # You can also add initial eval loss if needed
            if sr_metrics and trainer.is_world_process_zero():
                metrics.update(sr_metrics)
                trainer.log(metrics)  
            
            self.has_run_initial = True
        
        return control
    
    
    def on_evaluate(self, args, state, control, metrics = None, **kw):
        trainer = self.trainer
        step = int(state.global_step)

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
        
        sr_metrics = self._run_eval(trainer, step)
        
        if sr_metrics and trainer.is_world_process_zero():
            metrics.update(sr_metrics)
            trainer.log(metrics)
            
        return control
    