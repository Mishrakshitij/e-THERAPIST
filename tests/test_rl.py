"""Download-free NLPO integration checks with real, tiny GPT-2 models."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from e_therapist.models import CausalLMWithValueHead, frozen_copy
from e_therapist.rl import (
    AdaptiveKLController,
    DialogueRolloutStream,
    assign_advantages,
    collect_trajectory,
    optimize_policy,
    replay_batch,
    reward_example_eligible,
)
from e_therapist.rewards import RewardWeights
from e_therapist.scoring import PaperRewardScorer, bertscore_settings


class IntegerTokenizer:
    """A minimal deterministic tokenizer keeps tests independent of model hubs."""

    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(
            str(token) for token in tokens if not skip_special_tokens or token not in (0, 2)
        )


def tiny_policy():
    from transformers import GPT2Config, GPT2LMHeadModel

    return CausalLMWithValueHead(
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=24,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    ).eval()


def mask_snapshot(policy, terminal=False):
    """Control EOS support while keeping an actual Transformer masking model."""
    model = frozen_copy(policy)
    head = torch.nn.Linear(16, 17, bias=True)
    with torch.no_grad():
        head.weight.copy_(model.lm.lm_head.weight)
        head.bias.zero_()
        head.bias[2] = 100 if terminal else -100
    model.lm.lm_head = head
    return model.eval().requires_grad_(False)


class NLPOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(10)
        self.policy = tiny_policy()
        self.reference = frozen_copy(self.policy)
        self.mask_policy = mask_snapshot(self.policy)
        self.tokenizer = IntegerTokenizer()
        self.config = {
            "max_length": 20,
            "max_response_length": 3,
            "top_p": 0.9,
            "top_k": 4,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "ppo_epochs": 2,
            "batch_size": 2,
            "value_clip_range": None,
        }

    def collect(self, prompt="1 4 6", **config):
        return collect_trajectory(
            self.policy,
            self.mask_policy,
            self.reference,
            self.tokenizer,
            prompt,
            self.config | config,
        )

    def test_cached_collection_replay_probabilities_and_values_match(self):
        rollouts = [self.collect(), self.collect("1", max_response_length=2)]
        assign_advantages(rollouts, torch.tensor([0.7, -0.1]), self.config, 0.1)
        replay = replay_batch(self.policy, rollouts, self.tokenizer.pad_token_id)
        valid = replay["mask"]
        ratios = (replay["log_probs"] - replay["old_log_probs"]).exp()
        torch.testing.assert_close(
            ratios[valid], torch.ones_like(ratios[valid]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            replay["values"][valid], replay["old_values"][valid], atol=1e-6, rtol=1e-6
        )
        self.assertEqual(valid.sum().item(), 5)
        self.assertEqual(replay["values"][~valid].abs().sum().item(), 0)
        self.assertFalse(rollouts[0].terminal)
        self.assertEqual(len(rollouts[0].actions), 3)

    def test_tied_mask_logits_keep_a_nonempty_top_k_nucleus_intersection(self):
        with torch.no_grad():
            self.mask_policy.lm.lm_head.weight.zero_()
            self.mask_policy.lm.lm_head.bias.zero_()
            self.mask_policy.lm.lm_head.bias[2] = -100
        rollout = self.collect(top_p=0.01, top_k=1)
        self.assertEqual(len(rollout.actions), 3)
        self.assertTrue(all(len(support) == 1 for support in rollout.support))

    def test_kl_keeps_reference_mass_outside_delayed_support(self):
        rollout = self.collect()
        for step, support in enumerate(rollout.support):
            prefix = torch.tensor([rollout.prompt + rollout.actions[:step]])
            with torch.no_grad():
                logref = self.reference(prefix).logits[0, -1].log_softmax(-1)
            action = rollout.actions[step]
            torch.testing.assert_close(rollout.reference_log_probs[step], logref[action])
            # pi == ref before truncation, so KL(pi|S || ref) = -log ref(S).
            expected_kl = -logref[support].logsumexp(-1)
            torch.testing.assert_close(rollout.exact_kl[step], expected_kl, atol=1e-6, rtol=1e-6)
            self.assertGreater(rollout.exact_kl[step].item(), 0)

    def test_terminal_and_truncated_rollouts_bootstrap_differently(self):
        truncated = self.collect()
        self.mask_policy = mask_snapshot(self.policy, terminal=True)
        terminal = self.collect(top_k=1)
        self.assertTrue(terminal.terminal)
        self.assertEqual(terminal.actions, [self.tokenizer.eos_token_id])
        self.assertEqual(terminal.bootstrap, 0)
        self.assertEqual(terminal.response, "")
        prefix = torch.tensor([truncated.prompt + truncated.actions])
        with torch.no_grad():
            expected_bootstrap = self.policy(prefix).values[0, -1].item()
        self.assertAlmostEqual(truncated.bootstrap, expected_bootstrap, places=6)
        assign_advantages(
            [truncated, terminal],
            torch.tensor([2.0, 3.0]),
            {"gamma": 1.0, "gae_lambda": 1.0},
            kl_coefficient=0,
        )
        torch.testing.assert_close(
            truncated.returns, torch.full_like(truncated.returns, 2 + expected_bootstrap)
        )
        torch.testing.assert_close(terminal.returns, torch.tensor([3.0]))

    def test_optimizer_changes_policy_without_changing_reference_or_delayed_mask(self):
        rollouts = [self.collect(), self.collect("1 8")]
        assign_advantages(rollouts, torch.tensor([1.0, -0.5]), self.config, 0.1)
        policy_before = {name: p.detach().clone() for name, p in self.policy.named_parameters()}
        reference_before = {
            name: p.detach().clone() for name, p in self.reference.named_parameters()
        }
        mask_before = {name: p.detach().clone() for name, p in self.mask_policy.named_parameters()}
        support_before = [[tokens[:] for tokens in r.support] for r in rollouts]
        # Optimization shuffles the rollout list; hold identities for support checks.
        original_order = rollouts[:]
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=0.01)
        metrics = optimize_policy(self.policy, rollouts, optimizer, self.tokenizer, self.config)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(p, policy_before[name])
                for name, p in self.policy.named_parameters()
            )
        )
        for name, parameter in self.reference.named_parameters():
            torch.testing.assert_close(parameter, reference_before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        for name, parameter in self.mask_policy.named_parameters():
            torch.testing.assert_close(parameter, mask_before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        self.assertEqual([r.support for r in original_order], support_before)
        replay = replay_batch(self.policy, rollouts, self.tokenizer.pad_token_id)
        self.assertTrue(
            ((replay["log_probs"] - replay["old_log_probs"]).abs()[replay["mask"]] > 1e-5).any()
        )

    def test_empty_buffer_errors_and_controller_direction(self):
        with self.assertRaises(ValueError):
            assign_advantages([], [], {}, 0)
        with self.assertRaises(ValueError):
            replay_batch(self.policy, [], 0)
        controller = AdaptiveKLController(coefficient=0.1, target=1.0, horizon=100)
        controller.update(2.0, count=10)
        self.assertGreater(controller.value, 0.1)
        previous = controller.value
        controller.update(0.1, count=10)
        self.assertLess(controller.value, previous)


class ScoringIntegrationTests(unittest.TestCase):
    def test_bertscore_alias_and_custom_layers(self):
        table = {"roberta-large": 17}
        self.assertEqual(
            bertscore_settings({"bertscore_model": "FacebookAI/roberta-large"}, table)[
                "num_layers"
            ],
            17,
        )
        self.assertEqual(
            bertscore_settings(
                {"bertscore_model": "local/model", "bertscore_num_layers": 2}, table
            )["num_layers"],
            2,
        )
        with self.assertRaises(ValueError):
            bertscore_settings({"bertscore_model": "local/model"}, table)

    def test_absent_previous_text_is_zero_similarity_without_calling_encoder(self):
        scorer = object.__new__(PaperRewardScorer)

        class Encoder:
            def score(self, responses, references):
                self.pairs = (responses, references)
                return None, None, torch.tensor([0.6])

        scorer.bert = Encoder()
        result = scorer.similarity(["new response", "another response"], ["", "previous response"])
        torch.testing.assert_close(result, torch.tensor([0.0, 0.6]))
        self.assertEqual(scorer.bert.pairs, (["another response"], ["previous response"]))
        with self.assertRaises(ValueError):
            scorer.similarity(["response"], [])

    def test_full_reward_scorer_uses_shared_targets_and_predicted_sentiment(self):
        from e_therapist.schema import LABELS

        scorer = object.__new__(PaperRewardScorer)
        scorer.config = {"alpha": 1.0}
        scorer.weights = RewardWeights()
        scorer.convention = "paper"

        class Bank:
            def __init__(self):
                self.calls = []

            def sentiment(self, utterance):
                return "negative"

            def probabilities(self, task, texts):
                self.calls.append((task, texts))
                # Every target is class zero, including when another class wins.
                confidence = 0.8 if "reference" in texts[0] else 0.2
                probabilities = torch.full(
                    (len(texts), len(LABELS[task])), (1 - confidence) / (len(LABELS[task]) - 1)
                )
                probabilities[:, 0] = confidence
                return probabilities

        scorer.bank = Bank()
        scorer.response_perplexity = lambda responses: torch.tensor([2.0])
        scorer.similarity = lambda responses, references: torch.tensor(
            [0.4 if references[0] else 0.0]
        )
        example = {
            "response": "reference",
            "prompt": "context",
            "user_utterance": "user",
            "context": [{"speaker": "therapist", "utterance": "previous reference"}],
            "labels": {task: labels[0] for task, labels in LABELS.items()},
        }
        value = scorer([example], ["candidate"])
        # R1..R5=.6; R6=(.4+.4)/2=.4; R7=.5+0=.5.
        torch.testing.assert_close(value, torch.tensor([(0.75 * 0.6 + 0.25 * 0.45) / 7]))
        conditional = [
            text
            for task, texts in scorer.bank.calls
            for text in texts
            if task in ("politeness", "ipc")
        ]
        self.assertTrue(all("negative" in text for text in conditional))
        self.assertFalse(value.requires_grad)
        with_previous = scorer(
            [example | {"previous_response": "previous generated"}], ["candidate"]
        )
        torch.testing.assert_close(with_previous - value, torch.tensor([0.25 * 0.5 * 0.4 / 7]))

    def test_partial_labels_score_only_observed_targets_and_do_not_impute_profiles(self):
        from e_therapist.schema import LABELS

        scorer = object.__new__(PaperRewardScorer)
        scorer.config = {"missing_attribute_policy": "mask"}
        scorer.weights = RewardWeights()
        scorer.convention = "paper"

        class Bank:
            def __init__(self):
                self.calls = []
                self.sentiment_calls = 0

            def sentiment(self, text):
                self.sentiment_calls += 1
                return "negative"

            def probabilities(self, task, texts):
                targets = {"approach": "directive", "politeness": "polite", "ipc": "helpful"}
                if task not in targets:
                    raise AssertionError(
                        "Missing demographic/persona targets must not be predicted"
                    )
                self.calls.append((task, texts))
                confidence = 0.8 if "reference" in texts[0] else 0.2
                values = torch.full(
                    (len(texts), len(LABELS[task])), (1 - confidence) / (len(LABELS[task]) - 1)
                )
                values[:, LABELS[task].index(targets[task])] = confidence
                return values

        scorer.bank = Bank()
        scorer.response_perplexity = lambda responses: torch.full((len(responses),), 2.0)
        scorer.similarity = lambda responses, references: torch.tensor(
            [0.4 if text else 0.0 for text in references]
        )
        base = {"response": "reference", "prompt": "context", "user_utterance": "user"}
        examples = [
            base | {"labels": {"gender_age": None, "persona": None, "approach": "directive"}},
            base | {"labels": {"politeness": "polite", "ipc": "helpful"}},
            base | {"labels": {}},
        ]
        scores = scorer(examples, ["candidate"] * 3)
        expected = (0.75 * torch.tensor([0.6, 0.6, 0.0]) + 0.25 * 0.45) / 7
        torch.testing.assert_close(scores, expected)
        self.assertEqual(scorer.bank.sentiment_calls, 1)
        self.assertEqual({task for task, _ in scorer.bank.calls}, {"approach", "politeness", "ipc"})
        self.assertIsNone(examples[0]["labels"]["gender_age"])
        self.assertIsNone(examples[0]["labels"]["persona"])
        self.assertEqual(examples[2]["labels"], {})
        scorer.config = {}
        with self.assertRaises(ValueError):
            scorer(examples, ["candidate"] * 3)

    def test_quality_only_constructs_no_classifier_bank_or_checkpoint(self):
        module = types.ModuleType("bert_score")
        module.BERTScorer = lambda **kwargs: object()
        utilities = types.ModuleType("bert_score.utils")
        utilities.model2layers = {"roberta-large": 17}
        with patch.dict(sys.modules, {"bert_score": module, "bert_score.utils": utilities}):
            with patch(
                "e_therapist.scoring.ClassifierBank", side_effect=AssertionError("Unused bank")
            ):
                scorer = PaperRewardScorer({"mix_weights": [0, 1]}, torch.nn.Linear(1, 1), object())
        self.assertIsNone(scorer.bank)
        scorer.response_perplexity = lambda responses: torch.tensor([2.0])
        scorer.similarity = lambda responses, references: torch.tensor(
            [0.4 if references[0] else 0.0]
        )
        example = {"response": "reference", "prompt": "context", "user_utterance": "user"}
        score = scorer([example], ["candidate"])
        torch.testing.assert_close(score, torch.tensor([0.45 / 7]))
        self.assertTrue(reward_example_eligible(example, scorer.weights))

    def test_no_observed_attributes_need_no_checkpoint_and_score_zero_ra(self):
        scorer = PaperRewardScorer(
            {"missing_attribute_policy": "mask", "mix_weights": [1, 0]},
            torch.nn.Linear(1, 1),
            object(),
        )
        score = scorer([{"labels": {"persona": None, "gender_age": None}}], ["candidate"])
        torch.testing.assert_close(score, torch.tensor([0.0]))
        self.assertEqual(scorer.bank.models, {})

    def test_reward_eligibility_policies_and_invalid_labels(self):
        from e_therapist.schema import LABELS

        full = {"labels": {task: labels[0] for task, labels in LABELS.items()}}
        partial = {"labels": {"approach": "directive", "persona": None, "gender_age": None}}
        absent = {"labels": {}}
        weights = RewardWeights()
        self.assertTrue(reward_example_eligible(full, weights))
        self.assertFalse(reward_example_eligible(partial, weights))
        self.assertTrue(reward_example_eligible(partial, weights, "mask"))
        self.assertFalse(reward_example_eligible(absent, weights, "mask"))
        self.assertTrue(reward_example_eligible(absent, RewardWeights(mix=(0, 1)), "mask"))
        only_ipc = RewardWeights(attribute=(0, 0, 0, 0, 1))
        self.assertFalse(reward_example_eligible(partial, only_ipc, "mask"))
        with self.assertRaises(ValueError):
            reward_example_eligible({"labels": {"persona": "unknown"}}, weights, "mask")
        with self.assertRaises(ValueError):
            reward_example_eligible(full, weights, "impute")


class DialogueStreamTests(unittest.TestCase):
    def test_chronology_generated_context_and_epoch_history_reset(self):
        user1 = {
            "turn_id": 0,
            "speaker": "patient",
            "utterance": "first user",
            "sentiment": "negative",
        }
        therapist = {
            "turn_id": 1,
            "speaker": "therapist",
            "utterance": "reference reply",
            "sentiment": None,
        }
        user2 = {
            "turn_id": 2,
            "speaker": "patient",
            "utterance": "second user",
            "sentiment": "neutral",
        }
        first = {"conversation_id": "one", "turn_id": 1, "profile": {}, "context": [user1]}
        second = {
            "conversation_id": "one",
            "turn_id": 3,
            "profile": {},
            "context": [user1, therapist, user2],
        }
        stream = DialogueRolloutStream([second, first])
        self.assertEqual(stream.next()["turn_id"], 1)
        with self.assertRaises(RuntimeError):
            stream.next()
        stream.accept("generated reply")
        next_turn = stream.next()
        self.assertEqual(next_turn["turn_id"], 3)
        self.assertEqual(next_turn["previous_response"], "generated reply")
        self.assertIn("Therapist: generated reply", next_turn["prompt"])
        self.assertNotIn("reference reply", next_turn["prompt"])
        self.assertEqual(therapist["utterance"], "reference reply")
        stream.accept("second generated reply")
        restarted = stream.next()
        self.assertEqual(restarted["turn_id"], 1)
        self.assertEqual(restarted["previous_response"], "")

    def test_dialogue_boundaries_clear_generated_history(self):
        examples = [
            {"conversation_id": name, "turn_id": 1, "profile": {}, "context": []}
            for name in ("one", "two")
        ]
        stream = DialogueRolloutStream(examples)
        first = stream.next()
        stream.accept("first dialogue response")
        second = stream.next()
        self.assertNotEqual(first["conversation_id"], second["conversation_id"])
        self.assertEqual(second["previous_response"], "")

    def test_turn_gap_resets_generated_history_and_previous_response(self):
        first = {
            "conversation_id": "one",
            "turn_id": 2,
            "segment_id": 1,
            "profile": {},
            "context": [],
        }
        after_gap = {
            "conversation_id": "one",
            "turn_id": 5,
            "segment_id": 4,
            "profile": {},
            "context": [{"speaker": "patient", "turn_id": 4, "utterance": "new segment"}],
        }
        stream = DialogueRolloutStream([first, after_gap])
        stream.next()
        stream.accept("old segment response")
        current = stream.next()
        self.assertEqual(current["previous_response"], "")
        self.assertEqual(stream.generated, {})
        self.assertNotIn("old segment response", current["prompt"])


if __name__ == "__main__":
    unittest.main()
