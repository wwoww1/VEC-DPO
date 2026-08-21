from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union
import math

from .schema import (
    EvidenceStatus,
    ClaimType,
    VisualClaim,
    VecDPOSample,
    EVIDENCE_STATUS_TO_SCORE,
    compute_response_evidence_score,
)


Number = Union[int, float]


@dataclass
class EvidenceScoringConfig:
    alpha: float = 0.5
    min_weight: float = 0.5
    max_weight: float = 2.0
    uncertain_score: float = 0.0
    normalize_by_claim_count: bool = True
    use_claim_type_weight: bool = False

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.min_weight <= 0:
            raise ValueError("min_weight must be positive.")
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight.")
        if not math.isfinite(self.uncertain_score):
            raise ValueError("uncertain_score must be finite.")


DEFAULT_CLAIM_TYPE_WEIGHTS: Dict[ClaimType, float] = {
    ClaimType.OBJECT: 1.0,
    ClaimType.ATTRIBUTE: 1.0,
    ClaimType.COUNT: 1.0,
    ClaimType.RELATION: 1.0,
    ClaimType.OCR: 1.0,
    ClaimType.ACTION: 1.0,
    ClaimType.SCENE: 1.0,
    ClaimType.OTHER: 1.0,
}


class EvidenceScorer:
    def __init__(
        self,
        config: Optional[EvidenceScoringConfig] = None,
        claim_type_weights: Optional[Dict[Union[str, ClaimType], float]] = None,
    ):
        self.config = config or EvidenceScoringConfig()
        self.claim_type_weights = self._normalize_claim_type_weights(
            claim_type_weights or DEFAULT_CLAIM_TYPE_WEIGHTS
        )

    @staticmethod
    def _normalize_claim_type_weights(
        weights: Dict[Union[str, ClaimType], float]
    ) -> Dict[ClaimType, float]:
        normalized = {}
        for key, value in weights.items():
            claim_type = key if isinstance(key, ClaimType) else ClaimType(key)
            normalized[claim_type] = float(value)

        for claim_type in ClaimType:
            if claim_type not in normalized:
                normalized[claim_type] = 1.0

        return normalized

    def status_to_score(self, status: Union[str, EvidenceStatus]) -> float:

        if isinstance(status, str):
            status = EvidenceStatus(status)

        if status == EvidenceStatus.UNCERTAIN:
            return float(self.config.uncertain_score)

        return float(EVIDENCE_STATUS_TO_SCORE[status])

    def claim_to_score(self, claim: Union[VisualClaim, Dict]) -> float:

        if isinstance(claim, dict):
            claim = VisualClaim.from_dict(claim)

        if claim.score is None:
            base_score = self.status_to_score(claim.status)
        else:
            base_score = float(claim.score)

        if self.config.use_claim_type_weight:
            type_weight = self.claim_type_weights.get(claim.claim_type, 1.0)
            base_score *= type_weight

        return base_score

    def score_response(self, claims: Iterable[Union[VisualClaim, Dict]]) -> float:

        claims = list(claims)

        if len(claims) == 0:
            return 0.0

        scores = [self.claim_to_score(claim) for claim in claims]

        if self.config.normalize_by_claim_count:
            return float(sum(scores) / len(scores))

        return float(sum(scores))

    def score_status_counts(
        self, claims: Iterable[Union[VisualClaim, Dict]]
    ) -> Dict[str, int]:
        """
        Count supported / unsupported / uncertain claims.
        Useful for logging and analysis.
        """
        counts = {
            EvidenceStatus.SUPPORTED.value: 0,
            EvidenceStatus.UNSUPPORTED.value: 0,
            EvidenceStatus.UNCERTAIN.value: 0,
        }

        for claim in claims:
            if isinstance(claim, dict):
                claim = VisualClaim.from_dict(claim)

            counts[claim.status.value] += 1

        return counts

    def compute_gap(self, chosen_score: Number, rejected_score: Number) -> float:
        """
        Compute evidence gap.

        EvidenceGap = E(y_w, I) - E(y_l, I)
        """
        return float(chosen_score) - float(rejected_score)

    def compute_weight(self, evidence_gap: Number) -> float:

        gap = float(evidence_gap)
        weight = 1.0 + self.config.alpha * gap

        if math.isnan(weight) or math.isinf(weight):
            raise ValueError(f"Invalid evidence weight computed from gap={gap}")

        weight = max(self.config.min_weight, min(self.config.max_weight, weight))
        return float(weight)

    def score_pair(
        self,
        chosen_claims: Iterable[Union[VisualClaim, Dict]],
        rejected_claims: Iterable[Union[VisualClaim, Dict]],
    ) -> Dict[str, float]:
        """
        Score a chosen/rejected pair.

        Returns:
            {
                "chosen_evidence_score": ...,
                "rejected_evidence_score": ...,
                "evidence_gap": ...,
                "evidence_weight": ...
            }
        """
        chosen_score = self.score_response(chosen_claims)
        rejected_score = self.score_response(rejected_claims)
        gap = self.compute_gap(chosen_score, rejected_score)
        weight = self.compute_weight(gap)

        return {
            "chosen_evidence_score": chosen_score,
            "rejected_evidence_score": rejected_score,
            "evidence_gap": gap,
            "evidence_weight": weight,
        }

    def score_sample(self, sample: VecDPOSample) -> VecDPOSample:
        """
        Fill evidence fields for a VecDPOSample.

        This function mutates and returns the input sample.
        """
        pair_scores = self.score_pair(sample.chosen_claims, sample.rejected_claims)

        sample.chosen_evidence_score = pair_scores["chosen_evidence_score"]
        sample.rejected_evidence_score = pair_scores["rejected_evidence_score"]
        sample.evidence_gap = pair_scores["evidence_gap"]
        sample.evidence_weight = pair_scores["evidence_weight"]

        return sample

    def score_sample_dict(self, sample: Dict) -> Dict:
        """
        Score a raw dictionary sample and return a dictionary.

        This is useful for offline data construction scripts.
        """
        vec_sample = VecDPOSample.from_dict(sample)
        vec_sample = self.score_sample(vec_sample)
        return vec_sample.to_dict()

    def get_logging_stats(
        self,
        chosen_claims: Iterable[Union[VisualClaim, Dict]],
        rejected_claims: Iterable[Union[VisualClaim, Dict]],
    ) -> Dict[str, float]:
        """
        Return extra logging statistics for one preference pair.
        """
        chosen_claims = list(chosen_claims)
        rejected_claims = list(rejected_claims)

        chosen_counts = self.score_status_counts(chosen_claims)
        rejected_counts = self.score_status_counts(rejected_claims)
        pair_scores = self.score_pair(chosen_claims, rejected_claims)

        total_chosen = max(1, len(chosen_claims))
        total_rejected = max(1, len(rejected_claims))

        stats = {
            **pair_scores,
            "chosen_num_claims": float(len(chosen_claims)),
            "rejected_num_claims": float(len(rejected_claims)),
            "chosen_supported_ratio": chosen_counts[EvidenceStatus.SUPPORTED.value] / total_chosen,
            "chosen_unsupported_ratio": chosen_counts[EvidenceStatus.UNSUPPORTED.value] / total_chosen,
            "chosen_uncertain_ratio": chosen_counts[EvidenceStatus.UNCERTAIN.value] / total_chosen,
            "rejected_supported_ratio": rejected_counts[EvidenceStatus.SUPPORTED.value] / total_rejected,
            "rejected_unsupported_ratio": rejected_counts[EvidenceStatus.UNSUPPORTED.value] / total_rejected,
            "rejected_uncertain_ratio": rejected_counts[EvidenceStatus.UNCERTAIN.value] / total_rejected,
        }

        return stats


def score_vec_dpo_samples(
    samples: List[Union[VecDPOSample, Dict]],
    config: Optional[EvidenceScoringConfig] = None,
    claim_type_weights: Optional[Dict[Union[str, ClaimType], float]] = None,
) -> List[VecDPOSample]:
    """
    Score a list of VEC-DPO samples.

    Args:
        samples:
            A list of VecDPOSample objects or raw dictionaries.
        config:
            Evidence scoring configuration.
        claim_type_weights:
            Optional weights for different claim types.

    Returns:
        A list of scored VecDPOSample objects.
    """
    scorer = EvidenceScorer(config=config, claim_type_weights=claim_type_weights)
    scored_samples = []

    for sample in samples:
        if isinstance(sample, dict):
            sample = VecDPOSample.from_dict(sample)

        scored_samples.append(scorer.score_sample(sample))

    return scored_samples


def compute_dataset_statistics(samples: List[VecDPOSample]) -> Dict[str, float]:
    """
    Compute dataset-level statistics for VEC-DPO samples.

    These statistics are useful for sanity checks before training.
    """
    if len(samples) == 0:
        return {
            "num_samples": 0.0,
            "avg_chosen_evidence_score": 0.0,
            "avg_rejected_evidence_score": 0.0,
            "avg_evidence_gap": 0.0,
            "avg_evidence_weight": 0.0,
        }

    chosen_scores = []
    rejected_scores = []
    gaps = []
    weights = []

    chosen_claim_nums = []
    rejected_claim_nums = []

    for sample in samples:
        if sample.chosen_evidence_score is not None:
            chosen_scores.append(sample.chosen_evidence_score)
        if sample.rejected_evidence_score is not None:
            rejected_scores.append(sample.rejected_evidence_score)
        if sample.evidence_gap is not None:
            gaps.append(sample.evidence_gap)
        if sample.evidence_weight is not None:
            weights.append(sample.evidence_weight)

        chosen_claim_nums.append(len(sample.chosen_claims))
        rejected_claim_nums.append(len(sample.rejected_claims))

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "num_samples": float(len(samples)),
        "avg_chosen_evidence_score": avg(chosen_scores),
        "avg_rejected_evidence_score": avg(rejected_scores),
        "avg_evidence_gap": avg(gaps),
        "avg_evidence_weight": avg(weights),
        "avg_chosen_num_claims": avg(chosen_claim_nums),
        "avg_rejected_num_claims": avg(rejected_claim_nums),
    }
