"""Prompt/response encoding with response-only causal supervision."""

import torch


def encode_prompt(tokenizer, prompt, max_tokens):
    """Keep an available profile header and the newest context within the budget."""
    if max_tokens < 1:
        raise ValueError("Prompt token budget must be positive")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return ids
    if prompt.startswith("Patient profile: ") and "\n" in prompt:
        header, context = prompt.split("\n", 1)
        profile_ids = tokenizer.encode(header + "\n", add_special_tokens=False)
        if len(profile_ids) >= max_tokens:
            raise ValueError("Prompt token budget is too small for the patient profile and context")
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        return profile_ids + context_ids[-(max_tokens - len(profile_ids)) :]
    return ids[-max_tokens:]


def encode_example(tokenizer, prompt, response, max_length=512, max_response_length=50):
    if max_length < 3 or not 1 <= max_response_length < max_length:
        raise ValueError("Require 1 <= max_response_length < max_length, max_length >= 3")
    if tokenizer.eos_token_id is None:
        raise ValueError("Causal tokenizer requires an EOS token")
    target = tokenizer.encode(response, add_special_tokens=False)[: max_response_length - 1]
    target.append(tokenizer.eos_token_id)
    prompt_ids = encode_prompt(tokenizer, prompt, max_length - len(target))
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    return {"input_ids": prompt_ids + target, "labels": [-100] * len(prompt_ids) + target}


def collate_causal(examples, pad_token_id, device="cpu"):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(x["input_ids"]) for x in examples)
    ids, masks, labels = [], [], []
    for row in examples:
        n = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_token_id] * n)
        masks.append([1] * len(row["input_ids"]) + [0] * n)
        labels.append(row["labels"] + [-100] * n)
    return {
        key: torch.tensor(value, dtype=torch.long, device=device)
        for key, value in [("input_ids", ids), ("attention_mask", masks), ("labels", labels)]
    }


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a padding or end-of-sequence token")
    return tokenizer
