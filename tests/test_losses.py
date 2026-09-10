"""Small numerical cases catch sign, alignment, masking and gradient errors."""

import tempfile
import unittest

import torch
from torch.nn import functional as F

from e_therapist.losses import (
    apply_nucleus_mask,
    causal_lm_loss,
    classification_loss,
    clipped_value_loss,
    entropy_from_logits,
    generalized_advantage_estimate,
    masked_mean,
    masked_whiten,
    nucleus_mask,
    ppo_clipped_loss,
    token_kl,
    token_log_probs,
)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)

    def test_causal_response_alignment_and_gradient_mask(self):
        logits = torch.randn(1, 5, 4, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100]])
        loss = causal_lm_loss(logits, labels, torch.tensor([[1, 1, 1, 1, 0]]))
        expected = F.cross_entropy(logits[0, 1:3], torch.tensor([2, 3]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, [0, 3, 4]].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def test_left_padding_does_not_predict_first_real_token(self):
        logits = torch.randn(1, 4, 5, requires_grad=True)
        labels = torch.tensor([[0, 0, 3, 4]])
        loss = causal_lm_loss(logits, labels, torch.tensor([[0, 0, 1, 1]]))
        torch.testing.assert_close(loss, F.cross_entropy(logits[:, 2], torch.tensor([4])))

    def test_empty_mask_and_ignored_nan_values(self):
        values = torch.tensor([float("nan"), 4.0], requires_grad=True)
        value = masked_mean(values, torch.tensor([False, True]))
        self.assertEqual(value.item(), 4)
        value.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.0, 1.0]))
        logits = torch.randn(2, 3, 5, requires_grad=True)
        loss = causal_lm_loss(logits, torch.full((2, 3), -100))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_class_weights_match_torch_and_all_ignored_is_finite(self):
        logits = torch.randn(3, 3, requires_grad=True)
        labels = torch.tensor([1, 0, -100])
        weights = torch.tensor([1.0, 3.0, 2.0])
        torch.testing.assert_close(
            classification_loss(logits, labels, weights),
            F.cross_entropy(logits, labels, weight=weights),
        )
        self.assertEqual(classification_loss(logits, torch.full((3,), -100)).item(), 0)

    def test_ppo_clipping_and_detached_targets(self):
        new = torch.tensor([[1.5, 0.5, 1.1, float("nan")]]).log().requires_grad_()
        old = torch.zeros(1, 4, requires_grad=True)
        advantages = torch.tensor([[2.0, -2.0, 1.0, float("nan")]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0]])
        loss = ppo_clipped_loss(new, old, advantages, mask)
        # Positive advantage clips large increases; negative advantage clips decreases.
        self.assertAlmostEqual(loss.item(), -(2.4 - 1.6 + 1.1) / 3, places=6)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[0.0, 0.0, -1.1 / 3, 0.0]]))
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_value_clipping_and_terminal_target_gradients(self):
        values = torch.tensor([[1.0, 4.0, 99.0]], requires_grad=True)
        old = torch.tensor([[0.0, 3.0, 0.0]], requires_grad=True)
        returns = torch.tensor([[2.0, 0.0, float("nan")]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0]])
        loss = clipped_value_loss(values, old, returns, mask, clip_range=0.2)
        self.assertAlmostEqual(loss.item(), 0.5 * (1.8**2 + 4**2) / 2, places=5)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[0.0, 2.0, 0.0]]))
        self.assertIsNone(old.grad)
        self.assertIsNone(returns.grad)
        plain = clipped_value_loss(values, old, returns, mask, clip_range=None)
        self.assertAlmostEqual(plain.item(), 0.5 * (1 + 16) / 2)

    def test_entropy_with_invalid_actions_and_padding_has_finite_gradients(self):
        logits = torch.tensor(
            [[[0.0, 0.0, float("-inf")], [float("-inf"), float("-inf"), float("-inf")]]],
            requires_grad=True,
        )
        loss = entropy_from_logits(logits, torch.tensor([[1, 0]]))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            entropy_from_logits(torch.full((1, 3), float("-inf")))

    def test_gae_known_returns_padding_bootstrap_and_terminals(self):
        rewards = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 0.6, float("nan")], [0.5, 0.6, 0.7]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        dones = torch.tensor([[0, 1, 0], [0, 0, 0]])
        advantage, returns = generalized_advantage_estimate(
            rewards,
            values,
            mask,
            dones,
            gamma=1,
            gae_lambda=1,
            bootstrap_value=torch.tensor([100.0, 4.0]),
        )
        torch.testing.assert_close(returns, torch.tensor([[3.0, 2.0, 0.0], [10.0, 9.0, 7.0]]))
        torch.testing.assert_close(advantage, torch.tensor([[2.5, 1.4, 0.0], [9.5, 8.4, 6.3]]))
        self.assertFalse(returns.requires_grad)
        self.assertFalse(advantage.requires_grad)

    def test_gae_internal_terminal_blocks_future_rewards(self):
        rewards = torch.tensor([[1.0, 100.0]])
        advantages, _ = generalized_advantage_estimate(
            rewards,
            torch.zeros_like(rewards),
            torch.ones_like(rewards),
            torch.tensor([[1, 1]]),
            gamma=1,
            gae_lambda=1,
        )
        torch.testing.assert_close(advantages, rewards)

    def test_gae_discount_and_lambda_are_applied_separately(self):
        # delta1=2-.4=1.6; delta0=1+.5*.4-.2=1; A0=1+.5*.25*1.6=1.2.
        advantage, _ = generalized_advantage_estimate(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.2, 0.4]]),
            torch.ones(1, 2),
            gamma=0.5,
            gae_lambda=0.25,
        )
        torch.testing.assert_close(advantage, torch.tensor([[1.2, 1.6]]))

    def test_nucleus_uses_delayed_policy_and_crossing_token(self):
        delayed = torch.tensor([[0.1, 0.6, 0.3]]).log().requires_grad_()
        current = torch.tensor([[0.9, 0.05, 0.05]]).log().requires_grad_()
        support = nucleus_mask(delayed, top_p=0.7)
        torch.testing.assert_close(support, torch.tensor([[False, True, True]]))
        masked = apply_nucleus_mask(current, delayed, top_p=0.7)
        probability = masked.softmax(-1)
        torch.testing.assert_close(probability, torch.tensor([[0.0, 0.5, 0.5]]))
        loss = -token_log_probs(masked, torch.tensor([1])).mean()
        loss.backward()
        self.assertIsNone(delayed.grad)
        self.assertEqual(current.grad[0, 0].item(), 0)
        self.assertGreater(current.grad[0, 1:].abs().sum().item(), 0)
        self.assertEqual(nucleus_mask(torch.tensor([[0.0, 0.0]]), top_p=0.5).sum().item(), 1)
        self.assertEqual(
            nucleus_mask(torch.tensor([[0.0, float("-inf")]]), top_p=1).sum().item(), 1
        )

    def test_whitening_and_kl(self):
        white = masked_whiten(torch.tensor([1.0, 3.0, 100.0]), torch.tensor([1, 1, 0]))
        torch.testing.assert_close(white, torch.tensor([-1.0, 1.0, 0.0]))
        new = torch.tensor([-0.3, -0.7], requires_grad=True)
        ref = torch.tensor([-0.5, -0.5], requires_grad=True)
        torch.testing.assert_close(token_kl(new, ref), torch.tensor([0.2, -0.2]))
        self.assertTrue((token_kl(new, ref, "k3") >= 0).all())
        token_kl(new, ref).sum().backward()
        self.assertIsNone(ref.grad)


class OfflineModelTests(unittest.TestCase):
    def test_tiny_policy_forward_checkpoint_and_frozen_snapshot(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from e_therapist.models import CausalLMWithValueHead, frozen_copy

        torch.manual_seed(10)
        policy = CausalLMWithValueHead(
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=19,
                    n_embd=16,
                    n_layer=1,
                    n_head=2,
                    n_positions=32,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            )
        ).eval()
        ids = torch.tensor([[1, 5, 6, 2]])
        output = policy(ids, attention_mask=torch.ones_like(ids), labels=ids)
        self.assertEqual(output.values.shape, ids.shape)
        (output.loss + output.values.square().mean()).backward()
        self.assertGreater(policy.value_head.weight.grad.abs().sum().item(), 0)
        frozen = frozen_copy(policy)
        self.assertFalse(frozen.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.parameters()))
        self.assertNotEqual(
            next(frozen.parameters()).data_ptr(), next(policy.parameters()).data_ptr()
        )
        with tempfile.TemporaryDirectory() as directory:
            policy.save_pretrained(directory)
            restored = CausalLMWithValueHead.from_pretrained(
                directory, local_files_only=True
            ).eval()
            torch.testing.assert_close(restored(ids).logits, policy(ids).logits)
            torch.testing.assert_close(restored(ids).values, policy(ids).values)
        padded = torch.tensor([[0, 0, 1, 5, 6, 2]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
        torch.testing.assert_close(policy(padded, attention_mask=mask).logits[:, 2:], output.logits)

    def test_tiny_classifier_label_schema_and_roundtrip(self):
        from transformers import RobertaConfig, RobertaForSequenceClassification
        from e_therapist.models import RobertaClassifier
        from e_therapist.schema import LABEL_VOCAB

        model = RobertaForSequenceClassification(
            RobertaConfig(
                vocab_size=19,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=3,
                pad_token_id=0,
            )
        )
        classifier = RobertaClassifier("sentiment", model).eval()
        ids = torch.tensor([[1, 5, 6, 2]])
        output = classifier(ids, labels=torch.tensor([1]))
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(tuple(classifier.config.label2id), LABEL_VOCAB["sentiment"])
        with tempfile.TemporaryDirectory() as directory:
            classifier.save_pretrained(directory)
            restored = RobertaClassifier.from_pretrained(directory, local_files_only=True).eval()
            torch.testing.assert_close(restored(ids).logits, classifier(ids).logits)
            with self.assertRaises(ValueError):
                RobertaClassifier.from_pretrained(
                    directory, task="politeness", local_files_only=True
                )
        model.config.label2id = {"positive": 0, "neutral": 1, "negative": 2}
        with self.assertRaises(ValueError):
            RobertaClassifier("sentiment", model)


if __name__ == "__main__":
    unittest.main()
