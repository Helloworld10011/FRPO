# **Robust Policy Optimization to Prevent Catastrophic Forgetting**

## Abstract
Large language models are commonly trained through multi-stage post-training: first via RLHF, then
fine-tuned for other downstream objectives. Yet even small downstream updates can compromise earlier
learned behaviors (e.g., safety), exposing a brittleness known as catastrophic forgetting. This suggests
standard RLHF objectives do not guarantee robustness to future adaptation. To address it, most prior work
designs downstream-time methods to preserve previously learned behaviors. We argue that preventing this
requires pre-finetuning robustness: the base policy should avoid brittle high-reward solutions whose reward
drops sharply under standard fine-tuning.

We propose Fine-tuning Robust Policy Optimization (FRPO), a robust RLHF framework that optimizes
reward not only at the current policy, but across a KL-bounded neighborhood of policies reachable by
downstream adaptation. The key idea is to ensure reward stability under policy shifts via a max-min
formulation. By modifying GRPO, we develop an algorithm with no extra computation, and empirically
show it substantially reduces safety degradation across multiple base models and downstream fine-tuning
regimes (SFT and RL) while preserving downstream task performance. We further study a math-focused RL
setting, demonstrating that FRPO preserves accuracy under subsequent fine-tuning.
![Screenshot of the Application](images/into.pdf)


