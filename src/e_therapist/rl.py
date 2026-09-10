"""NLPO rollouts and clipped actor/critic optimization.

The delayed mask policy is a separate frozen model. Each rollout stores the
allowed token IDs and behavior probabilities so all PPO epochs use exactly the
sampling support from collection. The SFT reference remains frozen throughout.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from .losses import (
    clipped_value_loss,
    entropy_from_logits,
    generalized_advantage_estimate,
    masked_mean,
    masked_whiten,
    nucleus_mask,
    ppo_clipped_loss,
    token_kl,
)
from .models import CausalLMWithValueHead, frozen_copy
from .data import format_prompt
from .tokenization import encode_prompt, load_tokenizer
from .training import generation_examples, optimizer_for
from .utils import batches, device_for, seed_everything, write_json


@dataclass
class Trajectory:
    prompt: list[int]
    actions: list[int]
    support: list[list[int]]
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    reference_log_probs: torch.Tensor
    response: str
    terminal: bool
    bootstrap: float
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    exact_kl: torch.Tensor | None = None


class AdaptiveKLController:
    def __init__(self, coefficient=0.1, target=6.0, horizon=10000):
        if coefficient < 0 or target <= 0 or horizon <= 0:
            raise ValueError("Invalid KL controller parameters")
        self.value, self.target, self.horizon = coefficient, target, horizon

    def update(self, current, count):
        error = max(-0.2, min(0.2, current / self.target - 1))
        self.value *= max(0.01, 1 + error * count / self.horizon)


@torch.no_grad()
def collect_trajectory(policy, mask_policy, reference, tokenizer, prompt, config):
    """Sample from delayed-policy support; retain full-reference KL quantities.

    Behavior and PPO replay probabilities are normalized over stored support.
    The frozen reference remains normalized over its full vocabulary (Eq. 15).
    Store exact statewise KL for adapting its coefficient; sampled k3 is not an
    unbiased KL estimate when the reference has mass outside policy support.
    """
    max_response = config.get("max_response_length", 50)
    capacity = min(config.get("max_length", 512), policy.config.max_position_embeddings)
    if max_response < 1 or max_response >= capacity:
        raise ValueError("max_response_length must be positive and below model context capacity")
    if tokenizer.eos_token_id is None:
        raise ValueError("Trajectory collection requires an EOS token")
    prompt_ids = encode_prompt(tokenizer, prompt, capacity - max_response)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    device = policy.device
    sequence = torch.tensor([prompt_ids], device=device)
    supports, actions, old_log_probs, old_values, reference_log_probs = [], [], [], [], []
    exact_kls = []
    policy.eval()
    mask_policy.eval()
    reference.eval()
    terminal = False
    policy_cache = mask_cache = reference_cache = None
    for _ in range(max_response):
        attention = torch.ones_like(sequence)
        step_ids = sequence if policy_cache is None else sequence[:, -1:]
        output = policy(
            input_ids=step_ids,
            attention_mask=attention,
            past_key_values=policy_cache,
            use_cache=True,
        )
        policy_cache = output.past_key_values
        logits = output.logits[0, -1].float()
        mask_output = mask_policy(
            input_ids=step_ids, attention_mask=attention, past_key_values=mask_cache, use_cache=True
        )
        mask_cache = mask_output.past_key_values
        mask_logits = mask_output.logits[0, -1].float()
        allowed = nucleus_mask(mask_logits, config.get("top_p", 0.9))
        top_k = config.get("top_k", 20)
        if top_k > 0:
            top = (
                mask_logits.masked_fill(~allowed, -torch.inf)
                .topk(min(top_k, int(allowed.sum())))
                .indices
            )
            top_mask = torch.zeros_like(allowed).scatter(0, top, True)
            allowed &= top_mask
        indices = allowed.nonzero().flatten()
        distribution = torch.distributions.Categorical(logits=logits[indices])
        action_index = distribution.sample()
        action = indices[action_index]
        ref_output = reference(
            input_ids=step_ids,
            attention_mask=attention,
            past_key_values=reference_cache,
            use_cache=True,
        )
        reference_cache = ref_output.past_key_values
        ref_logits = ref_output.logits[0, -1].float()
        ref_log_probs = torch.log_softmax(ref_logits, -1)
        ref_logp = ref_log_probs[action]
        exact_kl = (distribution.probs * (distribution.logits - ref_log_probs[indices])).sum()
        exact_kls.append(exact_kl.clamp_min(0).cpu())
        supports.append(indices.cpu().tolist())
        actions.append(action.item())
        old_log_probs.append(distribution.log_prob(action_index).cpu())
        old_values.append(output.values[0, -1].float().cpu())
        reference_log_probs.append(ref_logp.cpu())
        sequence = torch.cat([sequence, action.reshape(1, 1)], -1)
        if action.item() == tokenizer.eos_token_id:
            terminal = True
            break
    bootstrap = 0.0 if terminal else float(policy(input_ids=sequence).values[0, -1].cpu())
    response = tokenizer.decode(actions, skip_special_tokens=True).strip()
    return Trajectory(
        prompt_ids,
        actions,
        supports,
        torch.stack(old_log_probs),
        torch.stack(old_values),
        torch.stack(reference_log_probs),
        response,
        terminal,
        bootstrap,
        exact_kl=torch.stack(exact_kls),
    )


def assign_advantages(trajectories, scores, config, kl_coefficient):
    """Sequence rewards on the last action, with token-level KL shaping and GAE."""
    if not trajectories or len(trajectories) != len(scores):
        raise ValueError("One reward is required per trajectory in a nonempty rollout buffer")
    if kl_coefficient < 0:
        raise ValueError("kl_coefficient must be nonnegative")
    for rollout, score in zip(trajectories, scores):
        kl = token_kl(rollout.old_log_probs, rollout.reference_log_probs, estimator="k1")
        rewards = -kl_coefficient * kl
        rewards[-1] += float(score)
        mask = torch.ones(1, len(rewards), dtype=torch.bool)
        dones = torch.zeros_like(mask)
        dones[0, -1] = rollout.terminal
        advantages, returns = generalized_advantage_estimate(
            rewards.unsqueeze(0),
            rollout.old_values.unsqueeze(0),
            mask,
            dones,
            gamma=config.get("gamma", 0.95),
            gae_lambda=config.get("gae_lambda", 0.95),
            bootstrap_value=torch.tensor([rollout.bootstrap]),
        )
        rollout.advantages, rollout.returns = advantages[0].detach(), returns[0].detach()
    # One global normalization per rollout buffer preserves relative rewards.
    flat = torch.cat([r.advantages for r in trajectories])
    normalized = masked_whiten(flat, torch.ones_like(flat, dtype=torch.bool))
    offset = 0
    for rollout in trajectories:
        rollout.advantages = normalized[offset : offset + len(rollout.actions)]
        offset += len(rollout.actions)


def replay_batch(policy, trajectories, pad_token_id):
    """Score collected actions and values without changing their stored supports."""
    if not trajectories:
        raise ValueError("Cannot replay an empty rollout batch")
    device = policy.device
    max_seq = max(len(t.prompt) + len(t.actions) for t in trajectories)
    max_actions = max(len(t.actions) for t in trajectories)
    ids, attention = [], []
    for t in trajectories:
        sequence = t.prompt + t.actions
        padding = max_seq - len(sequence)
        ids.append(sequence + [pad_token_id] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
    output = policy(
        input_ids=torch.tensor(ids, device=device),
        attention_mask=torch.tensor(attention, device=device),
    )
    fields = {
        key: []
        for key in (
            "log_probs",
            "values",
            "entropy",
            "old_log_probs",
            "old_values",
            "advantages",
            "returns",
            "mask",
        )
    }
    for b, rollout in enumerate(trajectories):
        logps, entropy = [], []
        start = len(rollout.prompt) - 1
        for step, (action, allowed) in enumerate(zip(rollout.actions, rollout.support)):
            indices = torch.tensor(allowed, device=device)
            logits = output.logits[b, start + step, indices].float()
            where = (indices == action).nonzero().item()
            logps.append(logits.log_softmax(-1)[where])
            entropy.append(entropy_from_logits(logits))
        n = len(logps)

        def padded(value):
            return torch.nn.functional.pad(value.to(device), (0, max_actions - n))

        fields["log_probs"].append(padded(torch.stack(logps)))
        fields["entropy"].append(padded(torch.stack(entropy)))
        fields["values"].append(padded(output.values[b, start : start + n]))
        for key in ("old_log_probs", "old_values", "advantages", "returns"):
            value = getattr(rollout, key)
            if value is None:
                raise ValueError("Assign advantages before replay")
            fields[key].append(padded(value))
        fields["mask"].append(padded(torch.ones(n, dtype=torch.bool)))
    return {key: torch.stack(value) for key, value in fields.items()}


def optimize_policy(policy, trajectories, optimizer, tokenizer, config):
    if not trajectories or config.get("ppo_epochs", 20) < 1 or config.get("batch_size", 8) < 1:
        raise ValueError("Optimization requires rollouts and positive epoch/batch sizes")
    # Keep dropout disabled in collection and replay; eval still permits gradients.
    policy.eval()
    records = []
    for _ in range(config.get("ppo_epochs", 20)):
        random.shuffle(trajectories)
        for group in batches(trajectories, config.get("batch_size", 8)):
            scored = replay_batch(policy, group, tokenizer.pad_token_id)
            actor = ppo_clipped_loss(
                scored["log_probs"],
                scored["old_log_probs"],
                scored["advantages"],
                scored["mask"],
                config.get("clip_range", 0.2),
            )
            critic = clipped_value_loss(
                scored["values"],
                scored["old_values"],
                scored["returns"],
                scored["mask"],
                config.get("value_clip_range", 0.2),
            )
            entropy = masked_mean(scored["entropy"], scored["mask"])
            loss = (
                actor
                + config.get("value_coefficient", 0.5) * critic
                - config.get("entropy_coefficient", 0.0) * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite NLPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            records.append(
                {
                    "loss": loss.item(),
                    "actor_loss": actor.item(),
                    "value_loss": critic.item(),
                    "entropy": entropy.item(),
                }
            )
    return {key: sum(x[key] for x in records) / len(records) for key in records[0]}


class DialogueRolloutStream:
    """Visit dialogues in shuffled order and turns chronologically within each.

    Accepted candidates replace earlier therapist turns in the next input. The
    recorded patient continuations remain fixed. R7 sees the previous generated
    therapist response; it never silently uses a reference as a generated turn.
    """

    def __init__(self, examples):
        self.groups = defaultdict(list)
        for example in examples:
            self.groups[example["conversation_id"]].append(example)
        if not self.groups:
            raise ValueError("At least one dialogue is required")
        for group in self.groups.values():
            group.sort(key=lambda x: int(x["turn_id"]))
        self.order = []
        self.group = []
        self.generated = {}
        self.previous_response = ""
        self.pending = None
        self.segment = None

    def next(self):
        if self.pending is not None:
            raise RuntimeError("Accept a candidate before requesting the next turn")
        if not self.group:
            if not self.order:
                self.order = list(self.groups)
                random.shuffle(self.order)
            self.group = list(self.groups[self.order.pop()])
            self.generated = {}
            self.previous_response = ""
            self.segment = None
        example = self.group.pop(0)
        segment = example.get("segment_id", 1)
        if self.segment is not None and segment != self.segment:
            self.generated = {}
            self.previous_response = ""
        self.segment = segment
        context = [dict(turn) for turn in example["context"]]
        for turn in context:
            if turn["speaker"] == "therapist" and turn["turn_id"] in self.generated:
                turn["utterance"] = self.generated[turn["turn_id"]]
        self.pending = example["turn_id"]
        return example | {
            "context": context,
            "prompt": format_prompt(example["profile"], context),
            "previous_response": self.previous_response,
        }

    def accept(self, response):
        if self.pending is None:
            raise RuntimeError("Request a turn before accepting a candidate")
        self.generated[self.pending] = response
        self.previous_response = response
        self.pending = None


def reward_example_eligible(example, weights, missing_policy="skip"):
    """Filter by observed active targets; never synthesize absent profile labels."""
    from .scoring import active_attribute_tasks, observed_attribute_targets

    if missing_policy not in ("skip", "mask"):
        raise ValueError("missing_attribute_policy must be 'skip' or 'mask'")
    active = active_attribute_tasks(weights)
    if not active:
        return True
    observed = observed_attribute_targets(example, active)
    return all(observed.values()) if missing_policy == "skip" else any(observed.values())


def train_nlpo(config):
    from .scoring import PaperRewardScorer, missing_attribute_policy

    for key in (
        "total_rollouts",
        "rollouts_per_update",
        "candidates",
        "ppo_epochs",
        "batch_size",
        "mask_refresh_updates",
    ):
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    seed_everything(config.get("seed", 10))
    device = device_for(config.get("device", "auto"))
    tokenizer = load_tokenizer(config["model"])
    policy = CausalLMWithValueHead.from_pretrained(config["model"]).to(device)
    reference, mask_policy = frozen_copy(policy), frozen_copy(policy)
    scorer = PaperRewardScorer(config, reference, tokenizer)
    examples = generation_examples(config, "train")
    target_policy = missing_attribute_policy(config)
    examples = [x for x in examples if reward_example_eligible(x, scorer.weights, target_policy)]
    if not examples:
        raise ValueError(
            f"No examples have the active reward targets required by {target_policy!r} policy"
        )
    stream = DialogueRolloutStream(examples)
    optimizer = optimizer_for(policy, config)
    controller = AdaptiveKLController(
        config.get("kl_coefficient", 0.1),
        config.get("kl_target", 6.0),
        config.get("kl_horizon", 10000),
    )
    output = Path(config["output"])
    write_json(output / "config.json", config)
    count, update, history = 0, 0, []
    while count < config.get("total_rollouts", 32000):
        trajectories, score_chunks = [], []
        budget = min(
            config.get("rollouts_per_update", 640), config.get("total_rollouts", 32000) - count
        )
        while len(trajectories) < budget:
            example = stream.next()
            candidates = []
            for _ in range(min(config.get("candidates", 3), budget - len(trajectories))):
                candidates.append(
                    collect_trajectory(
                        policy, mask_policy, reference, tokenizer, example["prompt"], config
                    )
                )
            # EOS remains a valid action; empty responses receive zero sequence reward.
            scores = torch.zeros(len(candidates))
            valid = [i for i, candidate in enumerate(candidates) if candidate.response]
            if valid:
                scores[valid] = scorer(
                    [example] * len(valid), [candidates[i].response for i in valid]
                )
            selected = max(valid, key=lambda i: float(scores[i])) if valid else 0
            stream.accept(candidates[selected].response)
            trajectories.extend(candidates)
            score_chunks.append(scores)
        rewards = torch.cat(score_chunks)
        assign_advantages(trajectories, rewards, config, controller.value)
        metrics = optimize_policy(policy, trajectories, optimizer, tokenizer, config)
        mean_kl = torch.cat([t.exact_kl for t in trajectories]).mean().item()
        controller.update(mean_kl, len(trajectories))
        count += len(trajectories)
        update += 1
        if update % config.get("mask_refresh_updates", 1) == 0:
            mask_policy.load_state_dict(policy.state_dict())
        record = {
            "update": update,
            "rollouts": count,
            "reward_mean": rewards.mean().item(),
            "kl_mean": mean_kl,
            "kl_coefficient": controller.value,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        policy.save_pretrained(output / "last")
        tokenizer.save_pretrained(output / "last")
        write_json(output / "history.json", history)
    return history[-1]
