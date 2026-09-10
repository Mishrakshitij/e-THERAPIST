"""Optional-profile commands and partial-label evaluation remain executable."""

import contextlib
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from e_therapist.cli import main
from e_therapist.evaluation import evaluate_generator
from e_therapist.schema import LABELS
from e_therapist.scoring import ClassifierBank


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 4]


class UniformLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 7) + self.weight)


class OptionalProfileCommandTests(unittest.TestCase):
    def test_generate_omits_unavailable_profile_fields_from_prompt(self):
        for profile_arguments in ([], ["--age", "adult"]):
            with self.subTest(profile_arguments=profile_arguments):
                output = io.StringIO()
                with (
                    patch(
                        "transformers.AutoModelForCausalLM.from_pretrained",
                        return_value=UniformLanguageModel(),
                    ),
                    patch("e_therapist.tokenization.load_tokenizer", return_value=TinyTokenizer()),
                    patch(
                        "e_therapist.evaluation.generate_response", return_value="response"
                    ) as generate,
                    contextlib.redirect_stdout(output),
                ):
                    main(
                        [
                            "generate",
                            "--checkpoint",
                            "local-checkpoint",
                            "--text",
                            "user text",
                            "--sentiment",
                            "negative",
                            "--device",
                            "cpu",
                            *profile_arguments,
                        ]
                    )
                prompt = generate.call_args.args[2]
                self.assertIn("Patient (sentiment=negative): user text", prompt)
                self.assertNotIn("gender=", prompt)
                self.assertNotIn("persona=", prompt)
                self.assertNotIn("None", prompt)
                if profile_arguments:
                    self.assertIn("age=adult", prompt)
                else:
                    self.assertNotIn("Patient profile:", prompt)
                self.assertEqual(json.loads(output.getvalue()), {"response": "response"})


class PartialEvaluationTests(unittest.TestCase):
    def test_missing_targets_and_blank_predictions_do_not_require_unused_classifiers(self):
        calls = []

        class LimitedBank(ClassifierBank):
            def _load(self, task):
                # Simulate a real directory with only these two checkpoints.
                if task not in ("sentiment", "politeness"):
                    raise FileNotFoundError(f"No {task} checkpoint")
                self.models[task] = object()

            def probabilities(self, task, texts):
                self._load(task)
                calls.append((task, texts))
                values = torch.zeros(len(texts), len(LABELS[task]))
                target = "polite" if task == "politeness" else "negative"
                values[:, LABELS[task].index(target)] = 1
                return values

        base = {
            "conversation_id": "one",
            "prompt": "user context",
            "response": "reference",
            "user_utterance": "user text",
            "profile": {
                "gender": None,
                "age": None,
                "persona": None,
                "issue": None,
                "approach": None,
            },
        }
        examples = [
            base
            | {
                "turn_id": 2,
                "labels": {
                    "gender_age": None,
                    "persona": None,
                    "approach": None,
                    "politeness": "polite",
                },
            },
            base | {"turn_id": 4, "labels": {"ipc": "helpful"}},
            base | {"turn_id": 6, "labels": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=UniformLanguageModel(),
                ),
                patch("e_therapist.evaluation.load_tokenizer", return_value=TinyTokenizer()),
                patch("e_therapist.evaluation.generation_examples", return_value=examples),
                patch(
                    "e_therapist.evaluation.generate_response",
                    side_effect=["generated response", "", "generated response"],
                ),
                patch("e_therapist.scoring.ClassifierBank", LimitedBank),
            ):
                result = evaluate_generator(
                    {
                        "model": "local",
                        "output": directory,
                        "device": "cpu",
                        "max_response_length": 4,
                        "max_length": 20,
                    },
                    with_attributes=True,
                )
        for task in ("gender_age", "persona", "approach"):
            self.assertEqual(result["attributes"][task], {"count": 0, "accuracy": None})
        self.assertEqual(result["attributes"]["politeness"]["accuracy"], 1)
        self.assertEqual(result["attributes"]["politeness"]["count"], 1)
        self.assertEqual(result["attributes"]["ipc"]["accuracy"], 0)
        self.assertEqual(result["attributes"]["ipc"]["empty_responses"], 1)
        self.assertEqual([task for task, _ in calls], ["sentiment", "politeness"])
        self.assertEqual(result["count"], 3)


if __name__ == "__main__":
    unittest.main()
