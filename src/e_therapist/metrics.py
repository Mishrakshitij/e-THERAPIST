"""Corpus-level metrics; empty inputs and missing labels are explicit."""

import math
from collections import Counter


def classification_metrics(targets, predictions, labels):
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Expected equally sized, nonempty targets and predictions")
    if any(x not in labels for x in targets) or any(
        x is not None and x not in labels for x in predictions
    ):
        raise ValueError("Unknown classification label")
    per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        support = sum(t == label for t in targets)
        per_class[label] = {
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "recall": tp / support if support else 0.0,
            "support": support,
        }
    accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)
    return {
        "accuracy": accuracy,
        "support_weighted_accuracy": accuracy,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels),
        "weighted_f1": sum(x["f1"] * x["support"] for x in per_class.values()) / len(targets),
        "count": len(targets),
        "per_class": per_class,
    }


def text_metrics(predictions):
    if not predictions:
        raise ValueError("No predictions to evaluate")
    token_lists = [text.split() for text in predictions]
    result = {
        "count": len(predictions),
        "response_length_words": sum(map(len, token_lists)) / len(token_lists),
    }
    for n in (1, 2):
        grams = Counter(
            tuple(tokens[i : i + n]) for tokens in token_lists for i in range(len(tokens) - n + 1)
        )
        result[f"distinct_{n}"] = len(grams) / sum(grams.values()) if grams else 0.0
    return result


def perplexity(total_negative_log_likelihood, token_count):
    if token_count <= 0:
        raise ValueError("Perplexity needs at least one scored token")
    mean = total_negative_log_likelihood / token_count
    return math.exp(min(mean, 700.0))
