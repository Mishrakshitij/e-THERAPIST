"""Hand-computed equation checks keep implementation choices explicit."""

import unittest

import torch

from e_therapist.rewards import (
    RewardWeights,
    approach_reward,
    composite_reward,
    context_reward,
    fluency_diversity_reward,
    gender_age_reward,
    ipc_reward,
    persona_reward,
    politeness_reward,
    true_class_probability,
)


class RewardTests(unittest.TestCase):
    def test_all_five_attribute_rewards_preserve_paper_sign_and_scaling(self):
        reference = torch.tensor([0.8, 0.4])
        candidate = torch.tensor([0.7, 0.9])
        for function in (
            gender_age_reward,
            persona_reward,
            approach_reward,
            politeness_reward,
            ipc_reward,
        ):
            with self.subTest(function=function.__name__):
                torch.testing.assert_close(
                    function(reference, candidate, alpha=2), torch.tensor([-0.6, -1.4])
                )
                torch.testing.assert_close(
                    function(reference, candidate, alpha=2, convention="sign_corrected"),
                    torch.tensor([0.6, 1.4]),
                )

    def test_context_cap_is_before_division(self):
        actual = context_reward(torch.tensor([0.9, 0.2, -0.1]), torch.tensor([0.8, 0.3, -0.2]))
        torch.testing.assert_close(actual, torch.tensor([0.5, 0.25, -0.15]))

    def test_fluency_repetition_sign_is_explicit(self):
        ppl = torch.tensor([2.0, 4.0, float("inf")])
        similarity = torch.tensor([0.8, 0.2, 0.5])
        torch.testing.assert_close(
            fluency_diversity_reward(ppl, similarity), torch.tensor([1.3, 0.45, 0.5])
        )
        torch.testing.assert_close(
            fluency_diversity_reward(ppl, similarity, "sign_corrected"),
            torch.tensor([0.7, 1.05, 0.5]),
        )
        with self.assertRaises(ValueError):
            fluency_diversity_reward(torch.tensor([0.0]), torch.tensor([0.5]))

    def test_hierarchical_weights_and_equation_nine_denominator(self):
        attribute = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        quality = torch.tensor([[6.0, 7.0]])
        # RA=.1+.4+.6+.8+1.5=3.4; RR=6.5.
        expected = (0.75 * 3.4 + 0.25 * 6.5) / 7
        self.assertAlmostEqual(composite_reward(attribute, quality).item(), expected, places=6)
        ones = composite_reward(torch.ones(3, 5), torch.ones(3, 2))
        torch.testing.assert_close(ones, torch.full((3,), 1 / 7))
        integer_ones = composite_reward(
            torch.ones(3, 5, dtype=torch.long), torch.ones(3, 2, dtype=torch.long)
        )
        torch.testing.assert_close(ones, integer_ones)

    def test_target_probability_tracks_ground_truth_not_argmax(self):
        logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        targets = torch.tensor([0, 1])
        expected = torch.tensor(
            [1 / (1 + torch.e + torch.e**2), torch.e / (1 + torch.e + torch.e**2)]
        )
        torch.testing.assert_close(true_class_probability(logits, targets), expected)

    def test_partial_attribute_weights_renormalize_without_imputing_unknowns(self):
        nan = float("nan")
        attributes = torch.tensor(
            [[nan, nan, 0.2, 0.4, 0.6], [0.7, nan, nan, nan, nan], [nan] * 5],
            requires_grad=True,
        )
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        values = composite_reward(attributes, torch.ones(3, 2), attribute_mask=mask)
        # Row 0: RA=(.2*.2+.2*.4+.3*.6)/(.2+.2+.3); row 1: RA=.7; row 2: RA=0.
        expected = (0.75 * torch.tensor([0.3 / 0.7, 0.7, 0.0]) + 0.25) / 7
        torch.testing.assert_close(values, expected)
        values.sum().backward()
        self.assertTrue(torch.isfinite(attributes.grad).all())
        self.assertEqual(attributes.grad[~mask.bool()].abs().sum().item(), 0)
        self.assertAlmostEqual(attributes.grad[0, 2].item(), 0.75 * 0.2 / 0.7 / 7, places=6)
        self.assertTrue(torch.isnan(attributes.detach()[~mask.bool()]).all())

    def test_fully_available_mask_preserves_original_reward_exactly(self):
        attributes = torch.tensor([[0.12, -0.7, 0.8, -0.2, 0.45], [0.2, 0.3, -0.6, 0.4, 0.8]])
        quality = torch.tensor([[0.5, 0.2], [0.8, 0.4]])
        original = composite_reward(attributes, quality)
        masked = composite_reward(attributes, quality, attribute_mask=torch.ones_like(attributes))
        torch.testing.assert_close(masked, original, atol=0, rtol=0)
        weights = RewardWeights(attribute=(0, 0, 0.2, 0.3, 0.5))
        partial = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])
        torch.testing.assert_close(
            composite_reward(attributes, quality, weights, attribute_mask=partial),
            composite_reward(attributes, quality, weights),
            atol=0,
            rtol=0,
        )

    def test_quality_only_ignores_missing_attribute_values(self):
        result = composite_reward(
            torch.full((2, 5), float("nan")), torch.ones(2, 2), RewardWeights(mix=(0, 1))
        )
        torch.testing.assert_close(result, torch.full((2,), 1 / 7))

    def test_shape_and_configuration_errors(self):
        with self.assertRaises(ValueError):
            RewardWeights(attribute=(0.2, 0.2))
        with self.assertRaises(ValueError):
            RewardWeights(mix=(0.9, 0.9))
        with self.assertRaises(ValueError):
            gender_age_reward(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError):
            persona_reward(torch.ones(2), torch.ones(2), alpha=0.5)
        with self.assertRaises(ValueError):
            ipc_reward(torch.ones(2), torch.ones(2), convention="ambiguous")
        with self.assertRaises(ValueError):
            composite_reward(torch.ones(2, 4), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            composite_reward(torch.ones(2, 5), torch.ones(2, 2), attribute_mask=torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
