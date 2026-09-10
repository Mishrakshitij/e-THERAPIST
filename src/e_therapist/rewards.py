"""The seven e-THERAPIST rewards (EMNLP 2023, equations 2--9).

Classifier arguments are probabilities of the *same target class*, not argmax
class IDs. Inputs for R3 include the user utterance; R4/R5 include its sentiment.
The literal paper convention subtracts candidate confidence and adds repetition
similarity. ``sign_corrected`` is an explicit alternative for maximizing the
stated semantic goals; it is not the published equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardWeights:
    """Reported beta, gamma and delta coefficients; Eq. 9 additionally divides by 7."""

    attribute: tuple[float, ...] = (0.1, 0.2, 0.2, 0.2, 0.3)
    quality: tuple[float, ...] = (0.5, 0.5)
    mix: tuple[float, ...] = (0.75, 0.25)

    def __post_init__(self) -> None:
        for name, expected in (("attribute", 5), ("quality", 2), ("mix", 2)):
            values = getattr(self, name)
            if len(values) != expected or not all(math.isfinite(x) and x >= 0 for x in values):
                raise ValueError(f"{name} must contain {expected} finite, nonnegative weights")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError(f"{name} weights must sum to one")


def _pair(a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError("Reward inputs must have matching shapes")


def true_class_probability(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Extract target-class probabilities from a classifier's raw logits."""
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("Target IDs must match the classifier's batch dimensions")
    return logits.float().softmax(-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def paper_attribute_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0
) -> Tensor:
    """Literal R1--R5: reference target confidence minus alpha times candidate confidence."""
    _pair(reference_probs, candidate_probs)
    if not math.isfinite(alpha) or alpha < 1:
        raise ValueError("The paper requires alpha >= 1")
    return reference_probs - alpha * candidate_probs


def sign_corrected_attribute_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0
) -> Tensor:
    """Negate equations 2--6 so maximizing rewards increases target confidence."""
    return -paper_attribute_reward(reference_probs, candidate_probs, alpha)


def _attribute(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float, convention: str
) -> Tensor:
    if convention == "paper":
        return paper_attribute_reward(reference_probs, candidate_probs, alpha)
    if convention == "sign_corrected":
        return sign_corrected_attribute_reward(reference_probs, candidate_probs, alpha)
    raise ValueError("convention must be 'paper' or 'sign_corrected'")


def gender_age_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0, convention: str = "paper"
) -> Tensor:
    """R1, equation 2: gender/age target probabilities GAC(t), GAC(y)."""
    return _attribute(reference_probs, candidate_probs, alpha, convention)


def persona_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0, convention: str = "paper"
) -> Tensor:
    """R2, equation 3: persona target probabilities PC(t), PC(y)."""
    return _attribute(reference_probs, candidate_probs, alpha, convention)


def approach_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0, convention: str = "paper"
) -> Tensor:
    """R3, equation 4: approach probabilities CTC([t,u]), CTC([y,u])."""
    return _attribute(reference_probs, candidate_probs, alpha, convention)


def politeness_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0, convention: str = "paper"
) -> Tensor:
    """R4, equation 5: politeness conditioned on the user's predicted sentiment."""
    return _attribute(reference_probs, candidate_probs, alpha, convention)


def ipc_reward(
    reference_probs: Tensor, candidate_probs: Tensor, alpha: float = 1.0, convention: str = "paper"
) -> Tensor:
    """R5, equation 6: IPC behavior conditioned on the user's predicted sentiment."""
    return _attribute(reference_probs, candidate_probs, alpha, convention)


def context_reward(context_bertscore: Tensor, user_bertscore: Tensor) -> Tensor:
    """R6, equation 7: min(BSF1(context,y) + BSF1(user,y), 1) / 2.

    The cap applies to the sum before division, so the upper bound is 0.5.
    BERTScores are not lower-clipped, preserving the paper's equation.
    """
    _pair(context_bertscore, user_bertscore)
    return (context_bertscore + user_bertscore).clamp_max(1.0) / 2.0


def fluency_diversity_reward(
    perplexity: Tensor, previous_bertscore: Tensor, convention: str = "paper"
) -> Tensor:
    """R7, equation 8: reciprocal PPL + BSF1(y, previous therapist utterance).

    ``sign_corrected`` uses reciprocal PPL + (1 - BSF1), encouraging diversity.
    A missing previous utterance should be supplied with similarity zero by the
    caller. Infinite perplexity is allowed and contributes zero fluency reward.
    """
    _pair(perplexity, previous_bertscore)
    if torch.isnan(perplexity).any() or torch.any(perplexity < 1):
        raise ValueError("Perplexity must be >= 1 (positive infinity is allowed)")
    if convention == "paper":
        diversity = previous_bertscore
    elif convention == "sign_corrected":
        diversity = 1.0 - previous_bertscore
    else:
        raise ValueError("convention must be 'paper' or 'sign_corrected'")
    return perplexity.reciprocal() + diversity


def composite_reward(
    attribute_rewards: Tensor,
    quality_rewards: Tensor,
    weights: RewardWeights | None = None,
    attribute_mask: Tensor | None = None,
) -> Tensor:
    """Equation 9, with optional explicit missing-attribute weight normalization.

    ``attribute_mask`` has the same shape as ``attribute_rewards`` and marks
    observed targets. With a partial mask, available beta weights are normalized
    per example; with no available targets, RA is zero. Fully observed examples
    preserve the original equation exactly. This masked extension never predicts
    or imputes a missing target and does not redistribute the outer delta weights.
    """
    if attribute_rewards.shape[:-1] != quality_rewards.shape[:-1]:
        raise ValueError("Attribute and quality rewards must have matching batch dimensions")
    if attribute_rewards.shape[-1] != 5 or quality_rewards.shape[-1] != 2:
        raise ValueError("Expected five attribute rewards and two response-quality rewards")
    weights = weights or RewardWeights()
    attribute_rewards = attribute_rewards.float()
    quality_rewards = quality_rewards.float()
    beta = attribute_rewards.new_tensor(weights.attribute)
    gamma = quality_rewards.new_tensor(weights.quality)
    if attribute_mask is not None:
        if attribute_mask.shape != attribute_rewards.shape:
            raise ValueError("attribute_mask must match attribute_rewards")
        valid = attribute_mask.to(device=attribute_rewards.device, dtype=torch.bool)
        # A missing value may be NaN; remove it before multiplication and reduction.
        attribute = (torch.where(valid, attribute_rewards, 0.0) * beta).sum(-1)
        denominator = (valid * beta).sum(-1)
        complete = (valid | beta.eq(0)).all(-1)
        denominator = torch.where(complete, torch.ones_like(denominator), denominator)
        attribute = torch.where(
            denominator > 0,
            attribute / denominator.clamp_min(torch.finfo(beta.dtype).tiny),
            torch.zeros_like(attribute),
        )
    else:
        attribute = (attribute_rewards * beta).sum(-1)
    if weights.mix[0] == 0:
        attribute = torch.zeros_like(attribute)
    quality = (quality_rewards * gamma).sum(-1)
    return (weights.mix[0] * attribute + weights.mix[1] * quality) / 7.0
