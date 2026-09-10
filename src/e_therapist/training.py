"""Supervised generator and reward-classifier training loops."""

import json
import math
import random
from pathlib import Path

import torch

from .data import build_examples, classifier_examples, load_turns, validate_splits
from .losses import causal_lm_loss, classification_loss
from .metrics import classification_metrics, perplexity
from .schema import LABELS
from .tokenization import collate_causal, encode_example, load_tokenizer
from .utils import batches, device_for, seed_everything, write_json


def split_turns(config, split):
    turns = load_turns(config["data"])
    manifest = json.loads(Path(config["splits"]).read_text())
    mapping = manifest.get("splits", manifest)
    mapping = {
        name: entry.get("conversation_ids", entry) if isinstance(entry, dict) else entry
        for name, entry in mapping.items()
    }
    mapping = {name: [str(x) for x in ids] for name, ids in mapping.items()}
    validate_splits(turns, mapping)
    ids = mapping[split]
    selected = {str(x) for x in ids}
    result = [x for x in turns if str(x["conversation_id"]) in selected]
    if not result:
        raise ValueError(f"Split {split!r} contains no turns")
    return result


def generation_examples(config, split):
    examples = build_examples(split_turns(config, split), config.get("context_window", 4))
    limit = config.get("max_examples")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("max_examples must be a positive integer")
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"Split {split!r} contains no eligible responses")
    return examples


def optimizer_for(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )


def validate_training_config(config):
    for key in ("epochs", "batch_size", "gradient_accumulation"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    if config.get("max_examples") is not None and config["max_examples"] < 1:
        raise ValueError("max_examples must be positive")


def train_sft(config):
    from transformers import AutoModelForCausalLM

    validate_training_config(config)
    seed_everything(config.get("seed", 10))
    device = device_for(config.get("device", "auto"))
    tokenizer = load_tokenizer(config["model"])
    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    max_length = min(config.get("max_length", 512), model.config.max_position_embeddings)

    def encode(split):
        return [
            encode_example(
                tokenizer,
                x["prompt"],
                x["response"],
                max_length,
                config.get("max_response_length", 50),
            )
            for x in generation_examples(config, split)
        ]

    train, validation = encode("train"), encode("validation")
    optimizer = optimizer_for(model, config)
    output = Path(config["output"])
    write_json(output / "config.json", config)
    best, history = math.inf, []
    batch_size = config.get("batch_size", 8)
    accumulation = config.get("gradient_accumulation", 1)
    for epoch in range(config.get("epochs", 20)):
        random.shuffle(train)
        model.train()
        epoch_nll, epoch_tokens = 0.0, 0
        batch_list = list(batches(train, batch_size))
        for group in batches(batch_list, accumulation):
            optimizer.zero_grad(set_to_none=True)
            # Weight microbatches by response tokens, including a short final group.
            group_tokens = sum(
                sum(v != -100 for v in r["labels"][1:]) for micro in group for r in micro
            )
            for micro in group:
                inputs = collate_causal(micro, tokenizer.pad_token_id, device)
                logits = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                ).logits
                loss = causal_lm_loss(logits, inputs["labels"], inputs["attention_mask"])
                count = int((inputs["labels"][:, 1:] != -100).sum())
                (loss * count / group_tokens).backward()
                epoch_nll += loss.item() * count
                epoch_tokens += count
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
        model.eval()
        val_nll, val_tokens = 0.0, 0
        with torch.no_grad():
            for micro in batches(validation, batch_size):
                inputs = collate_causal(micro, tokenizer.pad_token_id, device)
                logits = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                ).logits
                count = int((inputs["labels"][:, 1:] != -100).sum())
                val_nll += causal_lm_loss(logits, inputs["labels"]).item() * count
                val_tokens += count
        record = {
            "epoch": epoch + 1,
            "train_nll": epoch_nll / epoch_tokens,
            "validation_nll": val_nll / val_tokens,
            "validation_perplexity": perplexity(val_nll, val_tokens),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_nll"] < best:
            best = record["validation_nll"]
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
        write_json(output / "history.json", history)
    return history[-1]


def train_classifier(config, task):
    from transformers import AutoModelForSequenceClassification

    validate_training_config(config)
    if config.get("gradient_accumulation", 1) != 1:
        raise ValueError("Classifier training uses gradient_accumulation=1")
    if task not in LABELS:
        raise ValueError(f"Unknown classifier task {task}")
    seed_everything(config.get("seed", 10))
    device = device_for(config.get("device", "auto"))
    labels = list(LABELS[task])
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = load_tokenizer(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"],
        num_labels=len(labels),
        label2id=label2id,
        id2label=dict(enumerate(labels)),
        ignore_mismatched_sizes=True,
    ).to(device)
    train = classifier_examples(split_turns(config, "train"), task)
    validation = classifier_examples(split_turns(config, "validation"), task)
    if config.get("max_examples"):
        train, validation = train[: config["max_examples"]], validation[: config["max_examples"]]
    if not train or not validation:
        raise ValueError(f"No labeled {task} examples in train/validation")
    optimizer = optimizer_for(model, config)
    output = Path(config["output"]) / task
    write_json(output / "config.json", config | {"task": task})
    best, history = -1.0, []
    for epoch in range(config.get("epochs", 20)):
        random.shuffle(train)
        model.train()
        loss_sum = 0.0
        for rows in batches(train, config.get("batch_size", 8)):
            inputs = tokenizer(
                [x["text"] for x in rows],
                padding=True,
                truncation=True,
                max_length=config.get("max_length", 512),
                return_tensors="pt",
            ).to(device)
            targets = torch.tensor([label2id[x["label"]] for x in rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(**inputs).logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            loss_sum += loss.item() * len(rows)
        model.eval()
        predictions = []
        with torch.no_grad():
            for rows in batches(validation, config.get("batch_size", 8)):
                inputs = tokenizer(
                    [x["text"] for x in rows],
                    padding=True,
                    truncation=True,
                    max_length=config.get("max_length", 512),
                    return_tensors="pt",
                ).to(device)
                predictions.extend(labels[i] for i in model(**inputs).logits.argmax(-1).tolist())
        metrics = classification_metrics([x["label"] for x in validation], predictions, labels)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / len(train), **metrics}
        history.append(record)
        print(json.dumps({"task": task, **record}), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
        write_json(output / "history.json", history)
    return history[-1]
