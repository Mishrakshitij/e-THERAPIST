"""Token losses and NLPO/PPO primitives with explicit response and padding masks.

All masks use ``True``/1 for valid positions. Old policy quantities and return
targets are detached; only the current policy and value predictions get gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _mask(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if values.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match {values.shape}")
    return mask.to(device=values.device, dtype=torch.bool)


def masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean over valid entries; an empty mask produces differentiable zero."""
    valid = _mask(values, mask)
    selected = torch.where(valid, values, torch.zeros_like(values))
    return selected.sum() / valid.sum().clamp_min(1)


def masked_whiten(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize valid entries using population variance; leave padding zero."""
    valid = _mask(values, mask)
    centered = torch.where(valid, values - masked_mean(values, valid), 0.0)
    variance = masked_mean(centered.square(), valid)
    return centered * torch.rsqrt(variance + eps)


def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Next-token CE. Set prompt and padding labels to ``ignore_index``.

    Labels are unshifted token IDs. A token at position t is predicted by logits
    at t-1. With an attention mask, both predictor and target must be valid.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, length, vocab] and labels [batch, length]")
    target = labels[:, 1:].clone()
    valid = target.ne(ignore_index)
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must have the same shape as labels")
        valid &= attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    target.masked_fill_(~valid, ignore_index)
    if target.numel() == 0:
        return logits.sum() * 0.0
    # Unused logits cannot contribute NaNs or gradients through padding.
    prediction = torch.where(valid.unsqueeze(-1), logits[:, :-1].float(), 0.0)
    loss = F.cross_entropy(
        prediction.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(target)
    return masked_mean(loss, valid)


def classification_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Single-label classification CE with optional weighted class averaging."""
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("Expected logits [batch, classes] and labels [batch]")
    valid = labels.ne(ignore_index)
    safe_logits = torch.where(valid.unsqueeze(-1), logits.float(), 0.0)
    weights = class_weights.to(logits.device, torch.float32) if class_weights is not None else None
    losses = F.cross_entropy(
        safe_logits, labels, weight=weights, ignore_index=ignore_index, reduction="none"
    )
    if weights is None:
        return masked_mean(losses, valid)
    denominator = weights[labels.masked_fill(~valid, 0)].masked_fill(~valid, 0).sum()
    return losses.sum() / denominator.clamp_min(torch.finfo(losses.dtype).eps)


def token_log_probs(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Gather already aligned action log probabilities; does not shift tokens."""
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("Token IDs must match all logit dimensions except vocabulary")
    return F.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean token entropy, including distributions with -inf invalid actions."""
    valid = _mask(logits[..., 0], mask)
    safe_logits = torch.where(valid.unsqueeze(-1), logits.float(), 0.0)
    if torch.any(valid & ~torch.isfinite(safe_logits).any(dim=-1)):
        raise ValueError("Each valid token must have at least one finite action logit")
    log_probs = F.log_softmax(safe_logits, dim=-1)
    # 0 * -inf is NaN: replace impossible-action log probabilities before multiply.
    finite_logs = torch.where(torch.isfinite(log_probs), log_probs, 0.0)
    entropy = -(log_probs.exp() * finite_logs).sum(dim=-1)
    return masked_mean(entropy, valid)


def token_kl(log_probs: Tensor, reference_log_probs: Tensor, estimator: str = "k1") -> Tensor:
    """Sampled token KL estimator, without reduction.

    ``k1`` is log(pi/ref), used as the token reward penalty. ``k3`` is the
    nonnegative exp(-log_ratio) + log_ratio - 1 estimator. Individual k1 samples
    can be negative even though their on-policy expectation is a KL divergence.
    """
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("Policy and reference log probabilities must have equal shapes")
    log_ratio = log_probs.float() - reference_log_probs.detach().float()
    if estimator == "k1":
        return log_ratio
    if estimator == "k3":
        return torch.expm1(-log_ratio) + log_ratio
    raise ValueError("estimator must be 'k1' or 'k3'")


def ppo_clipped_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    """Negative PPO clipped surrogate to minimize (Appendix A.2)."""
    if not 0 <= clip_range < 1:
        raise ValueError("clip_range must lie in [0, 1)")
    if not (new_log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("Policy log probabilities and advantages must have equal shapes")
    valid = _mask(new_log_probs, mask)
    log_ratio = torch.where(valid, new_log_probs.float() - old_log_probs.detach().float(), 0.0)
    advantage = torch.where(valid, advantages.detach().float(), 0.0)
    ratio = log_ratio.exp()
    objective = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage
    )
    return -masked_mean(objective, valid)


def clipped_value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    mask: Tensor,
    clip_range: float | None = 0.2,
) -> Tensor:
    """Half MSE, optionally with PPO value clipping.

    Set ``clip_range=None`` for the unclipped squared-error objective in Eq. 16.
    The factor 1/2 is the conventional value-loss scale and can be absorbed in
    the caller's value coefficient.
    """
    if not (values.shape == old_values.shape == returns.shape):
        raise ValueError("Value predictions and returns must have equal shapes")
    if clip_range is not None and clip_range < 0:
        raise ValueError("Value clip_range must be nonnegative or None")
    valid = _mask(values, mask)
    current = torch.where(valid, values.float(), 0.0)
    old = torch.where(valid, old_values.detach().float(), 0.0)
    target = torch.where(valid, returns.detach().float(), 0.0)
    loss = (current - target).square()
    if clip_range is not None:
        clipped = old + (current - old).clamp(-clip_range, clip_range)
        loss = torch.maximum(loss, (clipped - target).square())
    return 0.5 * masked_mean(loss, valid)


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    dones: Tensor | None = None,
    gamma: float = 0.95,
    gae_lambda: float = 0.95,
    bootstrap_value: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """GAE for contiguous valid action spans, with terminal and padding resets.

    ``dones[b,t]`` marks termination *after* action t. Values have the same
    [batch, time] shape as rewards. ``bootstrap_value[b]`` is V of the state
    after the final valid action for a truncated rollout; a terminal overrides
    it. Without a bootstrap value, the end of each rollout has value zero.
    """
    if rewards.ndim != 2 or not (rewards.shape == values.shape == mask.shape):
        raise ValueError("rewards, values and mask must have shape [batch, time]")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    valid = mask.bool()
    if dones is None:
        dones = torch.zeros_like(valid)
    elif dones.shape != rewards.shape:
        raise ValueError("dones must match rewards")
    else:
        dones = dones.bool()
    batch, length = rewards.shape
    bootstrap = torch.zeros(batch, device=values.device, dtype=torch.float32)
    if bootstrap_value is not None:
        if bootstrap_value.shape != (batch,):
            raise ValueError("bootstrap_value must have shape [batch]")
        bootstrap = bootstrap_value.detach().to(values.device, torch.float32)
    safe_values = torch.where(valid, values.float(), 0.0)
    safe_rewards = torch.where(valid, rewards.float(), 0.0)
    advantages = torch.zeros_like(safe_values)
    last = torch.zeros_like(bootstrap)
    for t in range(length - 1, -1, -1):
        next_valid = valid[:, t + 1] if t + 1 < length else torch.zeros_like(valid[:, t])
        next_value = (
            torch.where(next_valid, safe_values[:, t + 1], bootstrap)
            if t + 1 < length
            else bootstrap
        )
        continuation = ~dones[:, t]
        delta = (
            safe_rewards[:, t]
            + gamma * torch.where(continuation, next_value, 0.0)
            - safe_values[:, t]
        )
        last = delta + gamma * gae_lambda * continuation * next_valid * last
        last = torch.where(valid[:, t], last, 0.0)
        advantages[:, t] = last
    returns = torch.where(valid, advantages + safe_values, 0.0)
    return advantages, returns


@torch.no_grad()
def nucleus_mask(mask_policy_logits: Tensor, top_p: float = 0.9) -> Tensor:
    """Smallest top-p support of a separate, frozen, delayed masking policy.

    The token that first crosses the threshold remains valid. At p=1 all finite
    logits remain valid. This function never attaches the mask to autograd.
    """
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    logits = mask_policy_logits.detach().float()
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("A nonempty vocabulary dimension is required")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("Mask policy logits may contain finite values or -inf only")
    finite = torch.isfinite(logits)
    if not finite.any(dim=-1).all():
        raise ValueError("The masking policy must allow at least one action per state")
    if top_p == 1:
        return finite
    sorted_logits, indices = logits.sort(descending=True, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative >= top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    result = torch.zeros_like(remove).scatter(-1, indices, ~remove)
    return result & finite


def apply_nucleus_mask(logits: Tensor, mask_policy_logits: Tensor, top_p: float = 0.9) -> Tensor:
    """Mask trainable logits using delayed-policy support without changing its weights."""
    if logits.shape != mask_policy_logits.shape:
        raise ValueError("Current and masking policy logits must have equal shapes")
    return logits.masked_fill(~nucleus_mask(mask_policy_logits, top_p), float("-inf"))
