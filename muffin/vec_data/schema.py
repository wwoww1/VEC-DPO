from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import math


class EvidenceStatus(str, Enum):
    """Evidence status of a visual claim with respect to the input image."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class ClaimType(str, Enum):
    """Fine-grained visual claim types."""
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    COUNT = "count"
    RELATION = "relation"
    OCR = "ocr"
    ACTION = "action"
    SCENE = "scene"
    OTHER = "other"


EVIDENCE_STATUS_TO_SCORE = {
    EvidenceStatus.SUPPORTED: 1.0,
    EvidenceStatus.UNSUPPORTED: -1.0,
    EvidenceStatus.UNCERTAIN: 0.0,
}


@dataclass
class VisualClaim:
    """
    A fine-grained visual claim extracted from a model response.

    Example:
        response: "There are two red cars beside a white building."
        claims:
            - "There are cars."                type=object
            - "The number of cars is two."     type=count
            - "The cars are red."              type=attribute
            - "The cars are beside a building." type=relation
    """

    claim: str
    claim_type: ClaimType = ClaimType.OTHER
    status: EvidenceStatus = EvidenceStatus.UNCERTAIN
    score: Optional[float] = None
    confidence: Optional[float] = None
    verifier: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.claim_type, str):
            self.claim_type = ClaimType(self.claim_type)
        if isinstance(self.status, str):
            self.status = EvidenceStatus(self.status)

        if not isinstance(self.claim, str) or len(self.claim.strip()) == 0:
            raise ValueError("VisualClaim.claim must be a non-empty string.")

        if self.score is not None:
            self.score = float(self.score)
            if not math.isfinite(self.score):
                raise ValueError("VisualClaim.score must be finite when provided.")

        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("VisualClaim.confidence must be between 0 and 1.")

    def to_dict(self) -> Dict[str, Any]:
        item = asdict(self)
        item["claim_type"] = self.claim_type.value
        item["status"] = self.status.value
        return item

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualClaim":
        return cls(
            claim=data["claim"],
            claim_type=data.get("claim_type", data.get("type", "other")),
            status=data.get("status", "uncertain"),
            score=data.get("score", None),
            confidence=data.get("confidence", None),
            verifier=data.get("verifier", None),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VecDPOSample:
    """
    A single VEC-DPO preference sample.

    Minimal required fields for training:
        image
        question
        chosen
        rejected
        evidence_gap or evidence_weight

    Full fields for analysis:
        chosen_claims
        rejected_claims
        chosen_evidence_score
        rejected_evidence_score
        evidence_gap
        evidence_weight
    """

    sample_id: str
    image: str
    question: str
    chosen: str
    rejected: str

    chosen_claims: List[VisualClaim] = field(default_factory=list)
    rejected_claims: List[VisualClaim] = field(default_factory=list)

    chosen_evidence_score: Optional[float] = None
    rejected_evidence_score: Optional[float] = None
    evidence_gap: Optional[float] = None
    evidence_weight: Optional[float] = None

    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._validate_text_fields()

        self.chosen_claims = [
            c if isinstance(c, VisualClaim) else VisualClaim.from_dict(c)
            for c in self.chosen_claims
        ]
        self.rejected_claims = [
            c if isinstance(c, VisualClaim) else VisualClaim.from_dict(c)
            for c in self.rejected_claims
        ]

        for name in (
            "chosen_evidence_score",
            "rejected_evidence_score",
            "evidence_gap",
            "evidence_weight",
        ):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite when provided.")
                setattr(self, name, value)

        if self.chosen_evidence_score is None and len(self.chosen_claims) > 0:
            self.chosen_evidence_score = compute_response_evidence_score(self.chosen_claims)

        if self.rejected_evidence_score is None and len(self.rejected_claims) > 0:
            self.rejected_evidence_score = compute_response_evidence_score(self.rejected_claims)

        if (
            self.evidence_gap is None
            and self.chosen_evidence_score is not None
            and self.rejected_evidence_score is not None
        ):
            self.evidence_gap = self.chosen_evidence_score - self.rejected_evidence_score

    def _validate_text_fields(self):
        required_fields = {
            "sample_id": self.sample_id,
            "image": self.image,
            "question": self.question,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }

        for name, value in required_fields.items():
            if not isinstance(value, str) or len(value.strip()) == 0:
                raise ValueError(f"{name} must be a non-empty string.")

    def compute_evidence_weight(
        self,
        alpha: float = 0.5,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
    ) -> float:
        """
        Compute evidence weight for VEC-DPO.

        Paper formula:
            w_E = 1 + alpha * EvidenceGap

        We clamp the weight for training stability.
        """
        if self.evidence_gap is None:
            raise ValueError(
                "evidence_gap is None. Provide evidence_gap or claim-level evidence scores."
            )

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if min_weight <= 0 or max_weight < min_weight:
            raise ValueError("Require 0 < min_weight <= max_weight.")

        weight = 1.0 + alpha * float(self.evidence_gap)
        if not math.isfinite(weight):
            raise ValueError("Computed evidence weight is not finite.")
        weight = max(min_weight, min(max_weight, weight))
        self.evidence_weight = weight
        return weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "image": self.image,
            "question": self.question,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_claims": [c.to_dict() for c in self.chosen_claims],
            "rejected_claims": [c.to_dict() for c in self.rejected_claims],
            "chosen_evidence_score": self.chosen_evidence_score,
            "rejected_evidence_score": self.rejected_evidence_score,
            "evidence_gap": self.evidence_gap,
            "evidence_weight": self.evidence_weight,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VecDPOSample":
        return cls(
            sample_id=str(data.get("id", data.get("sample_id", ""))),
            image=data["image"],
            question=data.get("question", data.get("prompt", "")),
            chosen=data.get("chosen", data.get("chosen_response", "")),
            rejected=data.get("rejected", data.get("rejected_response", "")),
            chosen_claims=data.get("chosen_claims", []),
            rejected_claims=data.get("rejected_claims", []),
            chosen_evidence_score=data.get("chosen_evidence_score", None),
            rejected_evidence_score=data.get("rejected_evidence_score", None),
            evidence_gap=data.get("evidence_gap", None),
            evidence_weight=data.get("evidence_weight", None),
            source=data.get("source", None),
            metadata=data.get("metadata", {}),
        )


def compute_response_evidence_score(claims: List[VisualClaim]) -> float:
    """
    Compute response-level visual evidence score.

    E(y, I) = average score over visual claims.

    supported   -> +1
    unsupported -> -1
    uncertain   -> 0
    """
    if len(claims) == 0:
        return 0.0

    scores = []
    for claim in claims:
        if claim.score is None:
            scores.append(EVIDENCE_STATUS_TO_SCORE[claim.status])
        else:
            scores.append(float(claim.score))

    return sum(scores) / len(scores)


def load_vec_dpo_json(path: str) -> List[VecDPOSample]:
    """Load VEC-DPO samples from .json or .jsonl."""
    lower_path = path.lower()
    if lower_path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif lower_path.endswith(".jsonl"):
        raw_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if isinstance(raw_data, dict):
        if "data" not in raw_data:
            raise ValueError("Input JSON must be a list or a dict containing a 'data' list.")
        raw_data = raw_data["data"]

    if not isinstance(raw_data, list):
        raise ValueError("VEC-DPO input data must be a list.")

    samples = []
    for idx, item in enumerate(raw_data):
        if "id" not in item and "sample_id" not in item:
            item["id"] = f"vec_{idx:08d}"
        samples.append(VecDPOSample.from_dict(item))

    return samples


def save_vec_dpo_json(samples: List[VecDPOSample], path: str, jsonl: bool = False) -> None:
    """Save VEC-DPO samples to .json or .jsonl."""
    data = [sample.to_dict() for sample in samples]

    if jsonl or path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def validate_vec_dpo_sample(sample: VecDPOSample) -> None:
    """
    Validate a VEC-DPO sample before training.

    This function should be called in the dataset loader to fail early
    if the data format is inconsistent.
    """
    if not sample.image:
        raise ValueError(f"Sample {sample.sample_id}: missing image.")
    if not sample.question:
        raise ValueError(f"Sample {sample.sample_id}: missing question.")
    if not sample.chosen:
        raise ValueError(f"Sample {sample.sample_id}: missing chosen response.")
    if not sample.rejected:
        raise ValueError(f"Sample {sample.sample_id}: missing rejected response.")

    if sample.evidence_weight is None and sample.evidence_gap is None:
        if len(sample.chosen_claims) == 0 or len(sample.rejected_claims) == 0:
            raise ValueError(
                f"Sample {sample.sample_id}: need evidence_weight, evidence_gap, "
                "or both chosen/rejected claim annotations."
            )

    if sample.evidence_gap is not None and sample.evidence_gap < 0:
        raise ValueError(
            f"Sample {sample.sample_id}: evidence_gap must be non-negative; "
            "the visually better response must be the chosen response."
        )

    if sample.evidence_weight is not None and sample.evidence_weight <= 0:
        raise ValueError(f"Sample {sample.sample_id}: evidence_weight must be positive.")

    if (
        sample.chosen_evidence_score is not None
        and sample.rejected_evidence_score is not None
        and sample.evidence_gap is not None
    ):
        expected_gap = sample.chosen_evidence_score - sample.rejected_evidence_score
        if not math.isclose(sample.evidence_gap, expected_gap, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Sample {sample.sample_id}: evidence_gap={sample.evidence_gap} is inconsistent "
                f"with chosen_score - rejected_score={expected_gap}."
            )


def validate_vec_dpo_dataset(samples: List[VecDPOSample]) -> None:
    """Validate all VEC-DPO samples."""
    if len(samples) == 0:
        raise ValueError("Empty VEC-DPO dataset.")

    seen_ids = set()
    for sample in samples:
        validate_vec_dpo_sample(sample)
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
