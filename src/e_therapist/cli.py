"""Command-line entry points for dataset inspection and experiments."""

import argparse
import json
from pathlib import Path

from .schema import AGES, GENDERS, LABELS
from .utils import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="e-therapist")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate data/splits and report observed counts")
    inspect.add_argument("--data", default="data/psycon/psycon.csv")
    inspect.add_argument("--splits", default="data/psycon/splits.json")
    for name, default in (
        ("train-sft", "configs/sft.yaml"),
        ("train-classifiers", "configs/classifiers.yaml"),
        ("train-nlpo", "configs/nlpo.yaml"),
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default=default)
        if name == "train-classifiers":
            sub.add_argument("--task", choices=["all", *LABELS], default="all")
    evaluate = commands.add_parser("evaluate", help="Evaluate a generator checkpoint")
    evaluate.add_argument("--config", default="configs/sft.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default="runs/evaluation")
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--bertscore", action="store_true")
    evaluate.add_argument(
        "--classifiers", help="Classifier checkpoint directory for attribute metrics"
    )
    ceval = commands.add_parser("evaluate-classifier")
    ceval.add_argument("--config", default="configs/classifiers.yaml")
    ceval.add_argument("--task", choices=list(LABELS), required=True)
    ceval.add_argument("--split", choices=["validation", "test"], default="test")
    generate = commands.add_parser("generate", help="Generate one research-model response")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--gender", choices=GENDERS)
    generate.add_argument("--age", choices=AGES)
    generate.add_argument("--persona", choices=LABELS["persona"])
    group = generate.add_mutually_exclusive_group(required=True)
    group.add_argument("--sentiment", choices=LABELS["sentiment"])
    group.add_argument("--classifiers", help="Infer sentiment with a trained sentiment classifier")
    generate.add_argument("--max-new-tokens", type=int, default=50)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--seed", type=int, default=10)
    smoke = commands.add_parser(
        "smoke", help="Offline tiny-model training/reward/NLPO/checkpoint test"
    )
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument(
        "--neural-rewards",
        action="store_true",
        help="Also run NLPO with all tiny neural reward models and BERTScore",
    )
    args = parser.parse_args(argv)
    if args.command == "inspect":
        from .data import dataset_stats, load_turns, validate_splits

        turns = load_turns(args.data)
        splits = json.loads(Path(args.splits).read_text())
        validate_splits(turns, splits)
        result = dataset_stats(turns, splits)
    elif args.command == "train-sft":
        from .training import train_sft

        result = train_sft(read_config(args.config))
    elif args.command == "train-classifiers":
        from .training import train_classifier

        config = read_config(args.config)
        tasks = list(LABELS) if args.task == "all" else [args.task]
        result = {task: train_classifier(config, task) for task in tasks}
    elif args.command == "train-nlpo":
        from .rl import train_nlpo

        result = train_nlpo(read_config(args.config))
    elif args.command == "evaluate":
        from .evaluation import evaluate_generator

        config = read_config(args.config) | {"model": args.checkpoint, "output": args.output}
        if args.limit is not None:
            config["max_examples"] = args.limit
        if args.classifiers:
            config["classifiers"] = args.classifiers
        result = evaluate_generator(config, args.split, args.bertscore, bool(args.classifiers))
    elif args.command == "evaluate-classifier":
        from .evaluation import evaluate_classifier

        result = evaluate_classifier(read_config(args.config), args.task, args.split)
    elif args.command == "generate":
        from transformers import AutoModelForCausalLM
        from .data import format_prompt
        from .evaluation import generate_response
        from .tokenization import load_tokenizer
        from .utils import device_for, seed_everything

        seed_everything(args.seed)
        tokenizer = load_tokenizer(args.checkpoint)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device_for(args.device))
        sentiment = args.sentiment
        if args.classifiers:
            from .scoring import ClassifierBank

            sentiment = ClassifierBank(args.classifiers, tasks=["sentiment"]).sentiment(args.text)
        prompt = format_prompt(
            {"gender": args.gender, "age": args.age, "persona": args.persona},
            [{"speaker": "patient", "utterance": args.text, "sentiment": sentiment}],
        )
        result = {
            "response": generate_response(model, tokenizer, prompt, args.max_new_tokens, args.top_k)
        }
    else:
        from .smoke import run_smoke

        result = run_smoke(args.output, neural_rewards=args.neural_rewards)
    print(json.dumps(result, indent=2, allow_nan=False))
