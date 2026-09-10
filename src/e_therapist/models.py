"""GPT-2 policy/value model and the six task-specific RoBERTa classifiers.

Constructors accept locally created Hugging Face models, allowing fully offline
tests. ``from_pretrained`` loads published backbones or saved local checkpoints.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .losses import causal_lm_loss, classification_loss
from .schema import LABEL_VOCAB


@dataclass
class CausalValueOutput:
    logits: Tensor
    values: Tensor
    loss: Tensor | None = None
    past_key_values: Any = None


class CausalLMWithValueHead(nn.Module):
    """Autoregressive policy with a scalar value for each pre-action token state."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm
        hidden_size = getattr(lm.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(lm.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Backbone config must specify hidden_size or n_embd")
        self.value_head = nn.Linear(hidden_size, 1)
        # New heads must follow a backbone loaded with device_map or a dtype.
        parameter = next(lm.parameters())
        self.value_head.to(device=parameter.device, dtype=parameter.dtype)

    @property
    def config(self) -> Any:
        return self.lm.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path = "gpt2-medium", **kwargs: Any
    ) -> "CausalLMWithValueHead":
        from transformers import AutoModelForCausalLM

        result = cls(
            AutoModelForCausalLM.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
        )
        value_file = Path(pretrained_model_name_or_path) / "value_head.pt"
        if value_file.is_file():
            state = torch.load(value_file, map_location="cpu", weights_only=True)
            result.value_head.load_state_dict(state)
        return result

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> CausalValueOutput:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        # Explicit positions ensure identical response scores for left-padded
        # rollout batches and unpadded generation prefixes.
        if attention_mask is not None and "position_ids" not in kwargs:
            positions = attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(~attention_mask.bool(), 0)
            kwargs["position_ids"] = positions[:, -input_ids.shape[1] :]
        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        values = self.value_head(output.hidden_states[-1]).squeeze(-1)
        loss = None if labels is None else causal_lm_loss(output.logits, labels, attention_mask)
        return CausalValueOutput(
            output.logits, values, loss, getattr(output, "past_key_values", None)
        )

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(path, **kwargs)
        torch.save(self.value_head.state_dict(), path / "value_head.pt")

    def generate(self, *args: Any, **kwargs: Any) -> Tensor:
        """Delegate ordinary generation; NLPO uses an explicit delayed policy mask."""
        return self.lm.generate(*args, **kwargs)


def frozen_copy(model: nn.Module) -> nn.Module:
    """Independent evaluation-only snapshot for the SFT reference or delayed policy."""
    result = copy.deepcopy(model)
    result.requires_grad_(False)
    result.eval()
    return result


class RobertaClassifier(nn.Module):
    """Single-label classifier whose checkpoint records the canonical label order."""

    def __init__(self, task: str, model: nn.Module) -> None:
        super().__init__()
        if task not in LABEL_VOCAB:
            raise ValueError(f"Unknown classifier task: {task}")
        self.task = task
        self.model = model
        self.labels = tuple(LABEL_VOCAB[task])
        if model.config.num_labels != len(self.labels):
            raise ValueError(f"{task} requires {len(self.labels)} classifier outputs")
        mapping = {str(k): int(v) for k, v in (model.config.label2id or {}).items()}
        canonical = {label: i for i, label in enumerate(self.labels)}
        generic = {f"LABEL_{i}": i for i in range(len(self.labels))}
        if mapping and mapping != generic and mapping != canonical:
            raise ValueError(f"Checkpoint label order disagrees with the {task} schema")
        model.config.label2id = canonical
        model.config.id2label = {i: label for label, i in canonical.items()}
        model.config.e_therapist_task = task

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "roberta-large",
        task: str | None = None,
        **kwargs: Any,
    ) -> "RobertaClassifier":
        from transformers import AutoConfig, AutoModelForSequenceClassification

        path = str(pretrained_model_name_or_path)
        config_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "local_files_only", "token")
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(path, **config_kwargs)
        recorded_task = getattr(config, "e_therapist_task", None)
        task = task or recorded_task
        if task not in LABEL_VOCAB:
            raise ValueError(f"task must be one of {tuple(LABEL_VOCAB)}")
        if recorded_task and recorded_task != task:
            raise ValueError(f"Checkpoint is for {recorded_task}, not {task}")
        labels = LABEL_VOCAB[task]
        expected_mapping = {label: i for i, label in enumerate(labels)}
        # A previously saved classifier must never silently be relabeled.
        if recorded_task and config.label2id != expected_mapping:
            raise ValueError("Saved classifier label vocabulary is incompatible")
        existing_mapping = config.label2id or {}
        is_generic = existing_mapping == {f"LABEL_{i}": i for i in range(len(existing_mapping))}
        if existing_mapping and not is_generic and existing_mapping != expected_mapping:
            raise ValueError("Checkpoint label vocabulary is incompatible with the requested task")
        config.num_labels = len(labels)
        config.label2id = expected_mapping
        config.id2label = {i: label for i, label in enumerate(labels)}
        config.e_therapist_task = task
        model = AutoModelForSequenceClassification.from_pretrained(path, config=config, **kwargs)
        return cls(task, model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        class_weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("return_dict", None)
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        if labels is not None:
            output["loss"] = classification_loss(output.logits, labels, class_weights)
        return output

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        self.model.save_pretrained(save_directory, **kwargs)
