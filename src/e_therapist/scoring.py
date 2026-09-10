"""Frozen checkpoint-backed implementations of the seven reward signals."""

from pathlib import Path

import torch

from .rewards import (
    RewardWeights,
    composite_reward,
    context_reward,
    fluency_diversity_reward,
    paper_attribute_reward,
    sign_corrected_attribute_reward,
)
from .schema import LABELS, build_classifier_input
from .tokenization import load_tokenizer


ATTRIBUTE_TASKS = ("gender_age", "persona", "approach", "politeness", "ipc")


def active_attribute_tasks(weights):
    """Return only attributes whose inner and outer configured weights are active."""
    if weights.mix[0] == 0:
        return ()
    return tuple(task for task, weight in zip(ATTRIBUTE_TASKS, weights.attribute) if weight > 0)


def missing_attribute_policy(config):
    policy = config.get("missing_attribute_policy", "skip")
    if policy not in ("skip", "mask"):
        raise ValueError("missing_attribute_policy must be 'skip' or 'mask'")
    return policy


def observed_attribute_targets(example, tasks):
    """Check stored target labels without replacing missing values with predictions."""
    observed = {}
    for task in tasks:
        label = example.get("labels", {}).get(task)
        if label is not None and label not in LABELS[task]:
            raise ValueError(f"Invalid non-null target label for {task}: {label!r}")
        observed[task] = label is not None
    return observed


def bertscore_settings(config, model_layers):
    """Resolve BERTScore's layer table for legacy/organization model names."""
    model = config.get("bertscore_model", "roberta-large")
    layers = config.get("bertscore_num_layers")
    if layers is None:
        # BERTScore's tuned-layer table uses the legacy RoBERTa checkpoint IDs.
        layer_key = model.removeprefix("FacebookAI/")
        layers = model_layers.get(model, model_layers.get(layer_key))
    if layers is None or not isinstance(layers, int) or layers < 1:
        raise ValueError("Set bertscore_num_layers to a positive layer count for this checkpoint")
    return {
        "model_type": model,
        "num_layers": layers,
        "device": config.get("reward_device", "cpu"),
        "rescale_with_baseline": False,
    }


class ClassifierBank:
    def __init__(self, directory, device="cpu", tasks=None, lazy=False):
        self.device = torch.device(device)
        self.directory = Path(directory) if directory is not None else None
        self.tasks = tuple(LABELS) if tasks is None else tuple(tasks)
        if any(task not in LABELS for task in self.tasks):
            raise ValueError("Classifier tasks must use the canonical task names")
        self.models, self.tokenizers = {}, {}
        if not lazy:
            for task in self.tasks:
                self._load(task)

    def _load(self, task):
        if task not in self.tasks:
            raise ValueError(f"Classifier {task!r} is not enabled in this bank")
        if task in self.models:
            return
        if self.directory is None:
            raise ValueError(f"A classifiers directory is required for the active {task} reward")
        path = self.directory / task / "best"
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {task} checkpoint: {path}; train classifiers first")
        from transformers import AutoModelForSequenceClassification

        labels = LABELS[task]
        model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        if tuple(model.config.id2label[i] for i in range(len(labels))) != tuple(labels):
            raise ValueError(f"Checkpoint label order for {task} differs from the schema")
        self.models[task] = model.eval().requires_grad_(False)
        self.tokenizers[task] = load_tokenizer(path)

    @torch.no_grad()
    def probabilities(self, task, texts):
        self._load(task)
        tokenizer, model = self.tokenizers[task], self.models[task]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(512, model.config.max_position_embeddings - 2),
            return_tensors="pt",
        ).to(self.device)
        return model(**inputs).logits.float().softmax(-1).cpu()

    def sentiment(self, user_text):
        return LABELS["sentiment"][self.probabilities("sentiment", [user_text]).argmax(-1).item()]


class PaperRewardScorer:
    """Scores training candidates against targets using frozen trained models.

    The default ``skip`` policy requires every active attribute target. ``mask``
    scores observed targets only, normalizing their beta weights per example.
    Missing demographic/persona targets remain missing. A quality-only mixture
    uses no classifiers; other checkpoints load lazily when a known target needs
    them. No heuristic or random fallback is substituted.
    """

    def __init__(self, config, reference_lm, tokenizer):
        self.config = config
        missing_attribute_policy(config)
        self.weights = RewardWeights(
            tuple(config.get("attribute_weights", (0.1, 0.2, 0.2, 0.2, 0.3))),
            tuple(config.get("quality_weights", (0.5, 0.5))),
            tuple(config.get("mix_weights", (0.75, 0.25))),
        )
        self.convention = config.get("reward_convention", "paper")
        if self.convention not in ("paper", "sign_corrected"):
            raise ValueError("reward_convention must be paper or sign_corrected")
        tasks = list(active_attribute_tasks(self.weights))
        if any(task in tasks for task in ("politeness", "ipc")):
            tasks.append("sentiment")
        self.bank = (
            ClassifierBank(
                config.get("classifiers"),
                config.get("reward_device", "cpu"),
                tasks=tasks,
                lazy=True,
            )
            if tasks
            else None
        )
        self.reference = reference_lm.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.bert = None
        if self.weights.mix[1] > 0:
            try:
                from bert_score import BERTScorer
                from bert_score.utils import model2layers
            except ImportError as exc:
                raise ImportError(
                    "Install e-therapist[metrics] for BERTScore reward computation"
                ) from exc
            self.bert = BERTScorer(**bertscore_settings(config, model2layers))

    @torch.no_grad()
    def response_perplexity(self, responses):
        if not responses:
            return torch.empty(0)
        device = next(self.reference.parameters()).device
        values = []
        for response in responses:
            ids = [self.tokenizer.eos_token_id] + self.tokenizer.encode(
                response, add_special_tokens=False
            )
            ids += [self.tokenizer.eos_token_id]
            ids = ids[: self.reference.config.max_position_embeddings]
            ids = torch.tensor([ids], device=device)
            logits = self.reference(input_ids=ids).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
            )
            values.append(loss.clamp_max(80).exp().cpu())
        return torch.stack(values)

    def similarity(self, responses, references):
        # BERTScore's tokenizer cannot score absent previous utterances meaningfully.
        if len(responses) != len(references):
            raise ValueError("BERTScore candidates and references must have equal lengths")
        result = torch.zeros(len(responses))
        valid = [
            i for i, (r, t) in enumerate(zip(responses, references)) if r.strip() and t.strip()
        ]
        if valid:
            result[valid] = self.bert.score(
                [responses[i] for i in valid], [references[i] for i in valid]
            )[2].cpu()
        return result

    @torch.no_grad()
    def __call__(self, examples, responses):
        if len(examples) != len(responses) or not responses:
            raise ValueError("Reward examples and responses must have equal nonzero lengths")
        policy = missing_attribute_policy(self.config)
        active = active_attribute_tasks(self.weights)
        observed = [observed_attribute_targets(example, active) for example in examples]
        if policy == "skip" and any(not all(row.values()) for row in observed):
            raise ValueError(
                "Missing target label for an active reward; use mask policy or complete targets"
            )
        attributes = torch.zeros(len(examples), len(ATTRIBUTE_TASKS))
        attribute_mask = torch.zeros_like(attributes, dtype=torch.bool)
        sentiments = {}
        reward_fn = (
            paper_attribute_reward
            if self.convention == "paper"
            else sign_corrected_attribute_reward
        )
        for index, task in enumerate(ATTRIBUTE_TASKS):
            if task not in active:
                continue
            valid = [i for i, row in enumerate(observed) if row[task]]
            if not valid:
                continue
            if task in ("politeness", "ipc"):
                for i in valid:
                    if i not in sentiments:
                        sentiments[i] = self.bank.sentiment(examples[i]["user_utterance"])
            targets = torch.tensor(
                [LABELS[task].index(examples[i]["labels"][task]) for i in valid]
            ).unsqueeze(-1)
            reference_text = [
                build_classifier_input(
                    task, examples[i]["response"], examples[i]["user_utterance"], sentiments.get(i)
                )
                for i in valid
            ]
            candidate_text = [
                build_classifier_input(
                    task, responses[i], examples[i]["user_utterance"], sentiments.get(i)
                )
                for i in valid
            ]
            ref_prob = self.bank.probabilities(task, reference_text).gather(-1, targets).squeeze(-1)
            cand_prob = (
                self.bank.probabilities(task, candidate_text).gather(-1, targets).squeeze(-1)
            )
            attributes[valid, index] = reward_fn(ref_prob, cand_prob, self.config.get("alpha", 1.0))
            attribute_mask[valid, index] = True
        quality = torch.zeros(len(examples), 2)
        if self.weights.mix[1] > 0:
            if self.weights.quality[0] > 0:
                quality[:, 0] = context_reward(
                    self.similarity(responses, [x["prompt"] for x in examples]),
                    self.similarity(responses, [x["user_utterance"] for x in examples]),
                )
            if self.weights.quality[1] > 0:
                # A first response has no previous generated response similarity.
                previous = [x.get("previous_response", "") for x in examples]
                quality[:, 1] = fluency_diversity_reward(
                    self.response_perplexity(responses),
                    self.similarity(responses, previous),
                    self.convention,
                )
        values = composite_reward(
            attributes,
            quality,
            self.weights,
            attribute_mask=attribute_mask if policy == "mask" else None,
        )
        if not torch.isfinite(values).all():
            raise ValueError("Reward model returned a non-finite value")
        return values
