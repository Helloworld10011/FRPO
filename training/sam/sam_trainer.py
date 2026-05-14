from trl import GRPOTrainer, GRPOConfig
import torch
import torch.nn as nn
from typing import Any, Optional, Union

try:
    from accelerate.utils import DistributedType
except ImportError:
    DistributedType = None


class SAMGRPOConfig(GRPOConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SAMGRPOTrainer(GRPOTrainer):
    """
    GRPO with Sharpness-Aware Minimization (SAM).

    Seeks flat parameter-space minima by computing gradients at
    a worst-case perturbation θ̂ = θ + ε, where ‖ε‖₂ ≤ ρ.
    """

    def __init__(self, *args, sam_rho: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_rho = sam_rho
        self._sam_pass = 2  # controls metric logging (only log on pass 2)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # ──────────────────────────────────────────────────────────
        # 0. Save any accumulated gradients from previous micro-batches
        #    (needed when gradient_accumulation_steps > 1)
        # ──────────────────────────────────────────────────────────
        accumulated_grads = {}
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                accumulated_grads[name] = p.grad.data.clone()
        model.zero_grad()  # clean slate for pass 1

        # ──────────────────────────────────────────────────────────
        # 1. PASS 1: Forward + backward at current θ
        #    (only to obtain gradient direction for perturbation)
        # ──────────────────────────────────────────────────────────
        self._sam_pass = 1
        with self.compute_loss_context_manager():
            loss1 = self._compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss1 = loss1.mean()

        # Scale for gradient accumulation (match base Trainer behavior)
        loss1_for_backward = loss1 / self.args.gradient_accumulation_steps

        backward_kwargs = {}
        if (
            DistributedType is not None
            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
        ):
            backward_kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss1_for_backward, **backward_kwargs)

        # ──────────────────────────────────────────────────────────
        # 2. Compute perturbation ε and apply: θ → θ̂ = θ + ε
        # ──────────────────────────────────────────────────────────
        grad_norm = self._sam_grad_norm(model)
        epsilon = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    e = self.sam_rho * p.grad.data / (grad_norm + 1e-12)
                    epsilon[name] = e.clone()
                    p.data.add_(e)  # θ → θ + ε

        # ──────────────────────────────────────────────────────────
        # 3. Restore accumulated grads, clear pass-1 grads
        #    Pass-2 backward will ADD to these restored grads
        # ──────────────────────────────────────────────────────────
        model.zero_grad()
        for name, p in model.named_parameters():
            if name in accumulated_grads:
                if p.grad is None:
                    p.grad = accumulated_grads[name]
                else:
                    p.grad.data.copy_(accumulated_grads[name])
        del accumulated_grads  # free memory

        # ──────────────────────────────────────────────────────────
        # 4. PASS 2: Forward + backward at perturbed θ̂
        #    (these gradients are the ones actually used for the update)
        # ──────────────────────────────────────────────────────────
        self._sam_pass = 2
        with self.compute_loss_context_manager():
            loss2 = self._compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss2 = loss2.mean()

        loss2_for_backward = loss2 / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss2_for_backward, **backward_kwargs)

        # ──────────────────────────────────────────────────────────
        # 5. Restore original parameters: θ̂ → θ
        #    (.grad buffers are kept from pass 2)
        # ──────────────────────────────────────────────────────────
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in epsilon:
                    p.data.sub_(epsilon[name])  # θ + ε − ε = θ
        del epsilon

        # ──────────────────────────────────────────────────────────
        # 6. Cleanup (matches base Trainer)
        # ──────────────────────────────────────────────────────────
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return loss2.detach()

    @torch.no_grad()
    def _sam_grad_norm(self, model) -> torch.Tensor:
        """Global L2 norm of gradients over all trainable parameters."""
        norms = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                norms.append(p.grad.data.norm(p=2))
        if len(norms) == 0:
            return torch.tensor(1.0, device=next(model.parameters()).device)
        return torch.norm(torch.stack(norms), p=2)