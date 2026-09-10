"""Checkpoint evaluation and response generation on conversation-held-out data."""

import json
from pathlib import Path

import torch

from .metrics import classification_metrics, perplexity, text_metrics
from .schema import LABELS, build_classifier_input
from .tokenization import encode_example, encode_prompt, collate_causal, load_tokenizer
from .losses import causal_lm_loss
from .training import generation_examples, split_turns
from .data import classifier_examples
from .utils import batches, device_for, seed_everything, write_json


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=50, top_k=20):
    if not 0 < max_new_tokens < model.config.max_position_embeddings:
        raise ValueError("max_new_tokens must be below model context capacity")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    device = next(model.parameters()).device
    ids = encode_prompt(tokenizer, prompt, model.config.max_position_embeddings - max_new_tokens)
    ids = torch.tensor([ids or [tokenizer.eos_token_id]], device=device)
    model.eval()
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip()


def evaluate_generator(config, split="test", with_bertscore=False, with_attributes=False):
    from transformers import AutoModelForCausalLM

    seed_everything(config.get("seed", 10))
    device = device_for(config.get("device", "auto"))
    tokenizer = load_tokenizer(config["model"])
    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device).eval()
    examples = generation_examples(config, split)
    predictions = [
        generate_response(
            model,
            tokenizer,
            x["prompt"],
            config.get("max_response_length", 50),
            config.get("top_k", 20),
        )
        for x in examples
    ]
    result = text_metrics(predictions)
    max_length = min(config.get("max_length", 512), model.config.max_position_embeddings)
    encoded = [
        encode_example(
            tokenizer, x["prompt"], x["response"], max_length, config.get("max_response_length", 50)
        )
        for x in examples
    ]
    total_nll, token_count = 0.0, 0
    with torch.no_grad():
        for rows in batches(encoded, config.get("batch_size", 8)):
            batch = collate_causal(rows, tokenizer.pad_token_id, device)
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            count = int((batch["labels"][:, 1:] != -100).sum())
            total_nll += causal_lm_loss(logits, batch["labels"]).item() * count
            token_count += count
    result["reference_response_perplexity"] = perplexity(total_nll, token_count)
    result["reference_response_tokens"] = token_count
    if with_bertscore:
        try:
            from bert_score import BERTScorer
            from bert_score.utils import model2layers
        except ImportError as exc:
            raise ImportError("Install e-therapist[metrics] to compute BERTScore") from exc
        from .scoring import bertscore_settings

        scorer = BERTScorer(
            **bertscore_settings(config | {"reward_device": str(device)}, model2layers)
        )
        values = torch.zeros(len(predictions))
        valid = [i for i, text in enumerate(predictions) if text]
        if valid:
            values[valid] = scorer.score(
                [predictions[i] for i in valid], [examples[i]["response"] for i in valid]
            )[2].cpu()
        result["bertscore_f1"] = values.mean().item()
    if with_attributes:
        from .scoring import ClassifierBank

        # Missing targets have no metric and require no corresponding checkpoint.
        bank = ClassifierBank(
            config.get("classifiers"), config.get("reward_device", "cpu"), lazy=True
        )
        sentiments = {}
        result["attributes"] = {}
        for task in ("gender_age", "persona", "approach", "politeness", "ipc"):
            eligible = [i for i, x in enumerate(examples) if x["labels"].get(task) in LABELS[task]]
            valid = [i for i in eligible if predictions[i]]
            if not eligible:
                result["attributes"][task] = {"count": 0, "accuracy": None}
                continue
            if task in ("politeness", "ipc"):
                for i in valid:
                    if i not in sentiments:
                        sentiments[i] = bank.sentiment(examples[i]["user_utterance"])
            texts = [
                build_classifier_input(
                    task, predictions[i], examples[i]["user_utterance"], sentiments.get(i)
                )
                for i in valid
            ]
            pred = []
            for group in batches(texts, config.get("batch_size", 8)):
                pred.extend(
                    LABELS[task][i] for i in bank.probabilities(task, group).argmax(-1).tolist()
                )
            by_index = dict(zip(valid, pred))
            result["attributes"][task] = classification_metrics(
                [examples[i]["labels"][task] for i in eligible],
                [by_index.get(i) for i in eligible],
                LABELS[task],
            )
            result["attributes"][task]["empty_responses"] = len(eligible) - len(valid)
    output = Path(config["output"])
    write_json(output / f"{split}-metrics.json", result)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{split}-predictions.jsonl").open("w", encoding="utf-8") as handle:
        for example, prediction in zip(examples, predictions):
            handle.write(
                json.dumps(
                    {
                        "conversation_id": example["conversation_id"],
                        "turn_id": example["turn_id"],
                        "response": prediction,
                        "reference": example["response"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return result


def evaluate_classifier(config, task, split="test"):
    from transformers import AutoModelForSequenceClassification

    device = device_for(config.get("device", "auto"))
    checkpoint = Path(config["output"]) / task / "best"
    tokenizer = load_tokenizer(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    if tuple(model.config.id2label[i] for i in range(len(LABELS[task]))) != tuple(LABELS[task]):
        raise ValueError("Checkpoint classifier labels differ from schema")
    rows = classifier_examples(split_turns(config, split), task)
    if not rows:
        raise ValueError(f"No labeled examples for {task} in {split}")
    predictions = []
    with torch.no_grad():
        for group in batches(rows, config.get("batch_size", 8)):
            inputs = tokenizer(
                [x["text"] for x in group],
                padding=True,
                truncation=True,
                max_length=config.get("max_length", 512),
                return_tensors="pt",
            ).to(device)
            predictions.extend(LABELS[task][i] for i in model(**inputs).logits.argmax(-1).tolist())
    result = classification_metrics([x["label"] for x in rows], predictions, LABELS[task])
    write_json(checkpoint.parent / f"{split}-metrics.json", result)
    return result
