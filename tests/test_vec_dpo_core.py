import unittest

import torch

from muffin.losses.vec_dpo_loss import VecDPOLossConfig, vec_dpo_loss
from muffin.vec_data.build_vec_pairs import PairBuilderConfig, VecPairBuilder
from muffin.vec_data.claim_extractor import (
    ClaimExtractionConfig,
    VisualClaimExtractor,
    extract_claims_from_response_dict,
)
from muffin.vec_data.evidence_scorer import EvidenceScorer, EvidenceScoringConfig
from muffin.vec_data.schema import (
    EvidenceStatus,
    VecDPOSample,
    VisualClaim,
    validate_vec_dpo_sample,
)


class EvidencePipelineTests(unittest.TestCase):
    def test_uncertain_score_configuration_is_used(self):
        claim = VisualClaim(claim="An object is visible.", status=EvidenceStatus.UNCERTAIN)
        scorer = EvidenceScorer(EvidenceScoringConfig(uncertain_score=0.25))
        self.assertEqual(scorer.score_response([claim]), 0.25)

    def test_pair_builder_orders_responses_by_visual_evidence(self):
        item = {
            "id": "sample",
            "image": "image.jpg",
            "question": "What is visible?",
            "responses": [
                {
                    "text": "A grounded answer.",
                    "claims": [{"claim": "A cat is visible.", "status": "supported"}],
                },
                {
                    "text": "A hallucinated answer.",
                    "claims": [{"claim": "A plane is visible.", "status": "unsupported"}],
                },
            ],
        }
        builder = VecPairBuilder(PairBuilderConfig(alpha=0.5, min_gap=0.1))
        pairs = builder.build_pairs_from_item(item)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen, "A grounded answer.")
        self.assertEqual(pairs[0].evidence_gap, 2.0)
        self.assertEqual(pairs[0].evidence_weight, 2.0)

    def test_negative_gap_is_rejected(self):
        sample = VecDPOSample(
            sample_id="bad",
            image="image.jpg",
            question="Question?",
            chosen="Chosen",
            rejected="Rejected",
            evidence_gap=-0.1,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_vec_dpo_sample(sample)

    def test_prompt_only_mode_does_not_create_a_fake_claim(self):
        extractor = VisualClaimExtractor(ClaimExtractionConfig(mode="llm_prompt"))
        result = extract_claims_from_response_dict(
            {"text": "There are two cars."},
            extractor,
        )
        self.assertEqual(result["claims"], [])
        self.assertTrue(result["claim_extraction_pending"])
        self.assertIn("Response:", result["claim_extraction_prompt"])


class VecDPOLossTests(unittest.TestCase):
    @staticmethod
    def _gradient(use_evidence_weight, evidence_gap=None):
        chosen = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        rejected = torch.tensor([-0.1], dtype=torch.float64, requires_grad=True)
        zeros = torch.zeros(1, dtype=torch.float64)
        loss, _ = vec_dpo_loss(
            policy_chosen_logps=chosen,
            policy_rejected_logps=rejected,
            ref_chosen_logps=zeros,
            ref_rejected_logps=zeros,
            evidence_gap=evidence_gap,
            config=VecDPOLossConfig(
                beta=0.1,
                evidence_alpha=0.5,
                use_evidence_weight=use_evidence_weight,
            ),
        )
        loss.backward()
        return chosen.grad.item()

    def test_single_item_batch_weight_changes_gradient(self):
        dpo_gradient = self._gradient(use_evidence_weight=False)
        vec_gradient = self._gradient(
            use_evidence_weight=True,
            evidence_gap=torch.tensor([1.0], dtype=torch.float64),
        )
        self.assertAlmostEqual(vec_gradient / dpo_gradient, 1.5, places=6)

    def test_runtime_alpha_overrides_stale_precomputed_weight(self):
        zeros = torch.zeros(1)
        loss, metrics = vec_dpo_loss(
            policy_chosen_logps=zeros,
            policy_rejected_logps=zeros,
            ref_chosen_logps=zeros,
            ref_rejected_logps=zeros,
            evidence_weight=torch.tensor([99.0]),
            evidence_gap=torch.tensor([1.0]),
            config=VecDPOLossConfig(evidence_alpha=0.25),
        )
        self.assertAlmostEqual(metrics["evidence_weight_mean"].item(), 1.25)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item() * 1.25)


if __name__ == "__main__":
    unittest.main()
