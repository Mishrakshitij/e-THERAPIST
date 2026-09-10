"""Download-free integration checks with tiny random models and toy fixtures.

This command checks executable paths, tensor flow and checkpoint round trips.
Its outputs are diagnostics and cannot be interpreted as research results.
"""

import csv
from pathlib import Path

import torch

from .data import build_examples, classifier_examples, load_turns
from .schema import LABELS, TURN_FIELDS
from .utils import seed_everything, write_json


def run_smoke(output="runs/smoke", neural_rewards=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaForSequenceClassification,
    )
    from .evaluation import evaluate_generator
    from .models import CausalLMWithValueHead, frozen_copy
    from .rewards import (
        composite_reward,
        context_reward,
        fluency_diversity_reward,
        paper_attribute_reward,
    )
    from .rl import assign_advantages, collect_trajectory, optimize_policy
    from .scoring import ClassifierBank
    from .training import train_sft, train_classifier

    seed_everything(10)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "toy.csv"
    rows = []
    for i in range(6):
        profile = {
            "conversation_id": str(i + 1),
            "issue": "anxiety",
            "gender": "female" if i % 2 else "male",
            "age": ("young", "adult", "elder")[i % 3],
            "persona": LABELS["persona"][i % 5],
            "approach": LABELS["approach"][i % 3],
        }
        for turn_id, speaker, text in (
            (1, "patient", f"I feel concerned about task {i}."),
            (2, "therapist", f"Can you tell me about task {i}?"),
            (3, "patient", f"I would like help with step {i}."),
            (4, "therapist", f"We can explore step {i} together."),
        ):
            rows.append(
                profile
                | {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "utterance": text,
                    "sentiment": "negative" if speaker == "patient" else None,
                    "ipc": "helpful" if speaker == "therapist" else None,
                    "politeness": "polite" if speaker == "therapist" else None,
                }
            )
    with fixture.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    splits = output / "splits.json"
    write_json(splits, {"train": ["1", "2", "3", "4"], "validation": ["5"], "test": ["6"]})
    turns = load_turns(fixture)
    texts = [x["prompt"] + " " + x["response"] for x in build_examples(turns)]
    texts += [x["text"] for task in LABELS for x in classifier_examples(turns, task)]
    pretokenizer = Whitespace()
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    for text in texts:
        for word, _ in pretokenizer.pre_tokenize_str(text):
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pretokenizer
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        cls_token="[BOS]",
        sep_token="[EOS]",
        model_max_length=128,
    )
    base = output / "tiny-gpt2"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=128,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    classifier_base = output / "tiny-roberta"
    classifier = RobertaForSequenceClassification(
        RobertaConfig(
            vocab_size=len(vocabulary),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=130,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    classifier.save_pretrained(classifier_base)
    tokenizer.save_pretrained(classifier_base)
    config = {
        "data": str(fixture),
        "splits": str(splits),
        "device": "cpu",
        "seed": 10,
        "max_length": 128,
        "max_response_length": 10,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "context_window": 4,
        "max_grad_norm": 1.0,
        "model": str(base),
        "output": str(output / "sft"),
    }
    sft = train_sft(config)
    cc = config | {"model": str(classifier_base), "output": str(output / "classifiers")}
    classifier_results = {task: train_classifier(cc, task)["count"] for task in LABELS}
    bank = ClassifierBank(cc["output"])
    assert bank.sentiment("I feel concerned.") in LABELS["sentiment"]
    policy = CausalLMWithValueHead.from_pretrained(output / "sft" / "best")
    reference, mask_policy = frozen_copy(policy), frozen_copy(policy)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    sample = build_examples(turns)[0]
    rc = config | {"max_response_length": 4, "top_p": 0.9, "top_k": 8, "ppo_epochs": 1}
    trajectories = [
        collect_trajectory(policy, mask_policy, reference, tokenizer, sample["prompt"], rc)
        for _ in range(2)
    ]
    # Explicit toy probabilities exercise all seven mathematical rewards offline.
    attributes = paper_attribute_reward(torch.full((2, 5), 0.8), torch.full((2, 5), 0.6))
    r6 = context_reward(torch.tensor([0.6, 0.7]), torch.tensor([0.4, 0.5]))
    r7 = fluency_diversity_reward(torch.tensor([2.0, 3.0]), torch.tensor([0.1, 0.2]))
    scores = composite_reward(attributes, torch.stack([r6, r7], -1))
    assign_advantages(trajectories, scores, rc, 0.1)
    rl = optimize_policy(
        policy, trajectories, torch.optim.AdamW(policy.parameters(), lr=0.001), tokenizer, rc
    )
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in reference.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in mask_policy.state_dict().items())
    policy.save_pretrained(output / "nlpo")
    tokenizer.save_pretrained(output / "nlpo")
    loaded = CausalLMWithValueHead.from_pretrained(output / "nlpo")
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items())
    evaluation = evaluate_generator(
        config | {"model": str(output / "nlpo"), "output": str(output / "evaluation")}, "test"
    )
    result = {
        "status": "passed",
        "fixture": "toy",
        "sft_validation_nll": sft["validation_nll"],
        "classifier_validation_counts": classifier_results,
        "nlpo": rl,
        "evaluation_count": evaluation["count"],
        "frozen_models_unchanged": True,
        "checkpoint_round_trip": True,
    }
    if neural_rewards:
        from .rl import train_nlpo

        neural_config = config | {
            "model": str(output / "sft" / "best"),
            "classifiers": str(output / "classifiers"),
            "bertscore_model": str(classifier_base),
            "bertscore_num_layers": 1,
            "output": str(output / "neural-nlpo"),
            "reward_device": "cpu",
            "total_rollouts": 4,
            "rollouts_per_update": 2,
            "candidates": 2,
            "ppo_epochs": 1,
            "max_response_length": 4,
            "top_k": 8,
        }
        result["neural_nlpo"] = train_nlpo(neural_config)
        result["neural_evaluation"] = evaluate_generator(
            neural_config
            | {
                "model": str(output / "neural-nlpo" / "last"),
                "output": str(output / "neural-evaluation"),
            },
            with_bertscore=True,
            with_attributes=True,
        )
    write_json(output / "smoke-results.json", result)
    torch.set_num_threads(old_threads)
    return result
