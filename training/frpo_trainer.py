from trl import GRPOTrainer, GRPOConfig
import torch

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class FRPOConfig(GRPOConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class FRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        lamb: float = 1.0,
        do_offset: bool = True,
        jackknife: bool = True,
        delta: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lamb = lamb
        self.do_offset = do_offset
        self.jackknife = jackknife
        self.delta = delta
        
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss            
        advantages = inputs["advantages"]
        # if self.accelerator.is_main_process:
        #     print("Advantages before scaling:", advantages)
            
        if self.loss_type == "frpo":
            advantages = torch.exp(-advantages / (self.lamb))

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)  # r
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)  # r_clip

        # Two-sided clipping (hard cap for r)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # r * exp(-A/λ)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)  # r_clip * exp(-A/λ)
        
        if self.loss_type == "frpo":
            per_token_loss = torch.min(per_token_loss1, per_token_loss2)
        else:
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
            
        if self.loss_type != "frpo":
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "frpo":
            T = completion_mask.sum(-1).clamp(min=1.0)  # [B_local]
            loss_seq = (per_token_loss * completion_mask).sum(-1) / T  # [B_local]

            pi = inputs["prompt_ids"]
            pm = inputs["prompt_mask"]
            hash_local = ((pi * pm).sum(-1).float() / pm.sum(-1).clamp(min=1.0).float()).round().to(torch.long)  # [B_local]

            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                world = dist.get_world_size()
                rank  = dist.get_rank()
                B_loc = loss_seq.size(0)

                # gather (remote losses detached)
                loss_list = [torch.zeros_like(loss_seq)  for _ in range(world)]
                hash_list = [torch.zeros_like(hash_local) for _ in range(world)]
                dist.all_gather(loss_list, loss_seq.detach())
                dist.all_gather(hash_list, hash_local)

                loss_all_ng = torch.cat(loss_list, dim=0)  # [B_global], no grad
                hash_all    = torch.cat(hash_list, dim=0)  # [B_global]

                base = loss_all_ng.detach()
                idx  = torch.arange(rank * B_loc, (rank + 1) * B_loc, device=base.device)
                delta = torch.zeros_like(base).scatter(0, idx, (loss_seq - base[idx]).to(base.dtype))
                loss_all = base + delta
            else:
                loss_all = loss_seq
                hash_all = hash_local
                B_loc = loss_seq.size(0)

            # group by identical hash => u_g
            uniq, inv = torch.unique(hash_all, return_inverse=True)  # inv: [B_global] -> group idx
            sums = torch.zeros(uniq.numel(), device=loss_all.device, dtype=loss_all.dtype).scatter_add_(0, inv, loss_all)
            cnts = torch.bincount(inv, minlength=uniq.numel()).clamp_min(1)  # n_g per group
            loss_group = sums / cnts.to(loss_all.dtype)  # [#groups]
            G = loss_group.numel()

            # --- main robust term ---
            if not self.jackknife:
                loss_reward = torch.log(loss_group.clamp_min(1e-12)).mean()
            
            else:
                # For each sample, compute log of leave-one-out group average
                jackknife_terms = []
                
                for g in range(G):
                    group_mask = (inv == g)
                    group_indices = group_mask.nonzero(as_tuple=True)[0]
                    n_g = group_indices.size(0)  # This is the 'k' in the jackknife formula for this group
                    
                    # Get losses for this group
                    group_losses = loss_all[group_indices]
                    group_sum = group_losses.sum()
                    
                    # Original group average (already computed as loss_group[g])
                    original_avg = loss_group[g]
                    
                    # Compute leave-one-out averages for each sample in the group
                    # For sample i: loo_avg_i = (sum - loss_i) / (n_g - 1)
                    loo_avgs = (group_sum.unsqueeze(0) - group_losses) / (n_g - 1)

                    # Apply jackknife formula for this group:
                    # ĝ = k*ĝ - (k-1)/k * ∑ĝ(-i)
                    # Which becomes: k*log(u_g) - (k-1) * mean(log(u_g,-i))
                    jackknife_group_term = n_g * torch.log(original_avg.clamp_min(1e-12)) - \
                                            (n_g - 1) * torch.log(loo_avgs.clamp_min(1e-12)).mean()
                    
                    jackknife_terms.append(jackknife_group_term)
                
                # Average across all groups (this is the 1/G factor in equation 4.3 of the paper)
                loss_reward = torch.stack(jackknife_terms).mean()

            # --- grouped offset with the same reduction as the main term ---
            if self.do_offset:
                aux_tok = torch.min(coef_1, coef_2)  # [B,T]
                aux_seq = (aux_tok * completion_mask).sum(-1) / T  # [B_local]

                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    world = dist.get_world_size(); rank = dist.get_rank(); Bl = aux_seq.size(0)
                    aux_list = [torch.zeros_like(aux_seq) for _ in range(world)]
                    dist.all_gather(aux_list, aux_seq.detach())
                    aux_all_ng = torch.cat(aux_list, dim=0)
                    base = aux_all_ng.detach()
                    idx  = torch.arange(rank * Bl, (rank + 1) * Bl, device=base.device)
                    delta = torch.zeros_like(base).scatter(0, idx, (aux_seq - base[idx]).to(base.dtype))
                    aux_all = base + delta
                else:
                    aux_all = aux_seq

                aux_sums  = torch.zeros_like(sums).scatter_add_(0, inv, aux_all)
                aux_group = aux_sums / cnts.to(aux_all.dtype)          # [#groups]
                loss_reward = loss_reward - aux_group.mean()

            # -------- scale reward to GRPO per-rank denominator (keep β effect identical to GRPO) --------
            scale_to_grpo = (cnts.sum().to(loss_reward.dtype)) / float(B_loc)
            loss = (self.lamb if self.lamb > 1 else 1) * scale_to_grpo * loss_reward  # scale reward only

            # KL stays as in GRPO (per-rank mean) so β has the same effective strength as in GRPO
            if self.beta != 0.0:
                kl_seq = (per_token_kl * completion_mask).sum(-1) / T
                loss = loss + self.beta * kl_seq.mean()
        
        elif self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        if self.loss_type == "frpo":
            self._metrics[mode]["lamda"].append(self.lamb)
        
        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

