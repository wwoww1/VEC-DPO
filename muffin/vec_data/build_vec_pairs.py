from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .schema import (
    VisualClaim,
    VecDPOSample,
    load_vec_dpo_json,
    save_vec_dpo_json,
    validate_vec_dpo_dataset,
)
from .evidence_scorer import (
    EvidenceScorer,
    EvidenceScoringConfig,
    compute_dataset_statistics,
)


@dataclass
class PairBuilderConfig:
    alpha: float = 0.5
    min_weight: float = 0.5
    max_weight: float = 2.0
    min_gap: float = 0.1
    max_pairs_per_sample: int = 1
    pair_strategy: str = "best_vs_worst"
    keep_ties: bool = False
    sort_by_gap: bool = True

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.min_weight <= 0 or self.max_weight < self.min_weight:
            raise ValueError("Require 0 < min_weight <= max_weight.")
        if self.min_gap < 0:
            raise ValueError("min_gap must be non-negative.")
        if self.max_pairs_per_sample == 0 or self.max_pairs_per_sample < -1:
            raise ValueError("max_pairs_per_sample must be -1 or a positive integer.")


class CandidateResponse:
    """
    A candidate response with claim-level evidence annotations.
    """

    def __init__(
        self,
        text: str,
        claims: List[Dict[str, Any]],
        response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Candidate response text must be a non-empty string.")

        self.text = text.strip()
        self.response_id = response_id
        self.claims = [
            claim if isinstance(claim, VisualClaim) else VisualClaim.from_dict(claim)
            for claim in claims
        ]
        if not self.claims:
            raise ValueError(
                "Candidate response has no verified visual claims; evidence score is undefined."
            )
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateResponse":
        text = data.get("text", data.get("response", data.get("answer", "")))
        claims = data.get("claims", data.get("visual_claims", []))

        return cls(
            text=text,
            claims=claims,
            response_id=data.get("id", data.get("response_id", None)),
            metadata=data.get("metadata", {}),
        )


class VecPairBuilder:
    """
    Build evidence-calibrated preference pairs.

    This class is the bridge between offline evidence annotations and
    VEC-DPO training samples.
    """

    def __init__(
        self,
        config: Optional[PairBuilderConfig] = None,
        scorer: Optional[EvidenceScorer] = None,
    ):
        self.config = config or PairBuilderConfig()

        if scorer is None:
            scoring_config = EvidenceScoringConfig(
                alpha=self.config.alpha,
                min_weight=self.config.min_weight,
                max_weight=self.config.max_weight,
            )
            scorer = EvidenceScorer(scoring_config)

        self.scorer = scorer

        if self.config.pair_strategy not in {
            "best_vs_worst",
            "all_pairs",
            "top_vs_others",
        }:
            raise ValueError(
                f"Unsupported pair_strategy={self.config.pair_strategy}. "
                "Choose from: best_vs_worst, all_pairs, top_vs_others."
            )

    def score_candidates(
        self,
        responses: List[CandidateResponse],
    ) -> List[Dict[str, Any]]:
        """
        Compute evidence score for each candidate response.
        """
        scored = []

        for idx, response in enumerate(responses):
            evidence_score = self.scorer.score_response(response.claims)
            status_counts = self.scorer.score_status_counts(response.claims)

            scored.append(
                {
                    "index": idx,
                    "response_id": response.response_id,
                    "text": response.text,
                    "claims": response.claims,
                    "evidence_score": evidence_score,
                    "num_claims": len(response.claims),
                    "status_counts": status_counts,
                    "metadata": response.metadata,
                }
            )

        return scored

    def _is_valid_gap(self, gap: float) -> bool:
        if self.config.keep_ties:
            return gap >= self.config.min_gap
        return gap > self.config.min_gap

    def _make_pair_candidates(
        self,
        scored_responses: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Create candidate chosen/rejected pairs.

        Returns:
            A list of (chosen, rejected, evidence_gap).
        """
        if len(scored_responses) < 2:
            return []

        sorted_responses = sorted(
            scored_responses,
            key=lambda x: x["evidence_score"],
            reverse=True,
        )

        pairs = []

        if self.config.pair_strategy == "best_vs_worst":
            chosen = sorted_responses[0]
            rejected = sorted_responses[-1]
            gap = self.scorer.compute_gap(
                chosen["evidence_score"],
                rejected["evidence_score"],
            )
            if self._is_valid_gap(gap):
                pairs.append((chosen, rejected, gap))

        elif self.config.pair_strategy == "top_vs_others":
            chosen = sorted_responses[0]
            for rejected in sorted_responses[1:]:
                gap = self.scorer.compute_gap(
                    chosen["evidence_score"],
                    rejected["evidence_score"],
                )
                if self._is_valid_gap(gap):
                    pairs.append((chosen, rejected, gap))

        elif self.config.pair_strategy == "all_pairs":
            for i in range(len(sorted_responses)):
                for j in range(i + 1, len(sorted_responses)):
                    chosen = sorted_responses[i]
                    rejected = sorted_responses[j]
                    gap = self.scorer.compute_gap(
                        chosen["evidence_score"],
                        rejected["evidence_score"],
                    )
                    if self._is_valid_gap(gap):
                        pairs.append((chosen, rejected, gap))

        if self.config.sort_by_gap:
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

        if self.config.max_pairs_per_sample > 0:
            pairs = pairs[: self.config.max_pairs_per_sample]

        return pairs

    def build_pairs_from_item(
        self,
        item: Dict[str, Any],
        item_index: int = 0,
    ) -> List[VecDPOSample]:
        """
        Build VEC-DPO pairs from a single image-question item.
        """
        sample_id = str(item.get("id", item.get("sample_id", f"sample_{item_index:08d}")))
        image = item.get("image", item.get("image_path", item.get("image_file", "")))
        question = item.get("question", item.get("prompt", ""))

        if not image:
            raise ValueError(f"{sample_id}: missing image field.")
        if not question:
            raise ValueError(f"{sample_id}: missing question/prompt field.")

        raw_responses = item.get("responses", item.get("candidate_responses", None))
        if raw_responses is None:
            raise ValueError(
                f"{sample_id}: missing responses/candidate_responses field."
            )

        responses = [CandidateResponse.from_dict(r) for r in raw_responses]
        scored_responses = self.score_candidates(responses)
        pair_candidates = self._make_pair_candidates(scored_responses)

        vec_samples = []

        for pair_idx, (chosen, rejected, gap) in enumerate(pair_candidates):
            evidence_weight = self.scorer.compute_weight(gap)

            vec_sample = VecDPOSample(
                sample_id=f"{sample_id}_pair_{pair_idx:03d}",
                image=image,
                question=question,
                chosen=chosen["text"],
                rejected=rejected["text"],
                chosen_claims=chosen["claims"],
                rejected_claims=rejected["claims"],
                chosen_evidence_score=chosen["evidence_score"],
                rejected_evidence_score=rejected["evidence_score"],
                evidence_gap=gap,
                evidence_weight=evidence_weight,
                source=item.get("source", None),
                metadata={
                    "original_id": sample_id,
                    "pair_strategy": self.config.pair_strategy,
                    "chosen_response_id": chosen.get("response_id", None),
                    "rejected_response_id": rejected.get("response_id", None),
                    "chosen_status_counts": chosen.get("status_counts", {}),
                    "rejected_status_counts": rejected.get("status_counts", {}),
                    "chosen_num_claims": chosen.get("num_claims", 0),
                    "rejected_num_claims": rejected.get("num_claims", 0),
                    **item.get("metadata", {}),
                },
            )

            vec_samples.append(vec_sample)

        return vec_samples

    def build_pairs(
        self,
        raw_items: List[Dict[str, Any]],
        skip_errors: bool = True,
    ) -> Tuple[List[VecDPOSample], Dict[str, Any]]:
        """
        Build VEC-DPO pairs from a list of raw candidate-response items.

        Returns:
            vec_samples:
                Constructed VEC-DPO samples.
            build_stats:
                Data construction statistics.
        """
        vec_samples = []

        num_items = 0
        num_failed_items = 0
        num_items_without_pairs = 0

        errors = []

        for idx, item in enumerate(tqdm(raw_items, desc="Building VEC-DPO pairs")):
            num_items += 1

            try:
                pairs = self.build_pairs_from_item(item, item_index=idx)
            except Exception as exc:
                num_failed_items += 1
                errors.append(
                    {
                        "index": idx,
                        "id": item.get("id", item.get("sample_id", None)),
                        "error": str(exc),
                    }
                )

                if skip_errors:
                    continue
                raise

            if len(pairs) == 0:
                num_items_without_pairs += 1
                continue

            vec_samples.extend(pairs)

        build_stats = {
            "num_raw_items": num_items,
            "num_vec_pairs": len(vec_samples),
            "num_failed_items": num_failed_items,
            "num_items_without_pairs": num_items_without_pairs,
            "pair_strategy": self.config.pair_strategy,
            "min_gap": self.config.min_gap,
            "max_pairs_per_sample": self.config.max_pairs_per_sample,
            "alpha": self.config.alpha,
            "min_weight": self.config.min_weight,
            "max_weight": self.config.max_weight,
            "errors": errors[:20],
        }

        build_stats.update(compute_dataset_statistics(vec_samples))

        return vec_samples, build_stats


def load_raw_items(path: str) -> List[Dict[str, Any]]:
    """
    Load raw candidate-response items from json or jsonl.
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported input file format: {path}")

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        else:
            raise ValueError(
                "Input JSON is a dict. Expected a list or a dict with key 'data'."
            )

    if not isinstance(data, list):
        raise ValueError("Input data must be a list of raw candidate-response items.")

    return data


def save_build_stats(stats: Dict[str, Any], path: str) -> None:
    """
    Save construction statistics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build VEC-DPO preference pairs from candidate responses."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw candidate-response file (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save VEC-DPO preference data.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Path to save data construction statistics. Default: output + .stats.json",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Evidence calibration strength in w_E = 1 + alpha * EvidenceGap.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.5,
        help="Minimum evidence weight.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=2.0,
        help="Maximum evidence weight.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.1,
        help="Minimum evidence gap required to construct a pair.",
    )
    parser.add_argument(
        "--max-pairs-per-sample",
        type=int,
        default=1,
        help="Maximum number of pairs built from each raw item. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--pair-strategy",
        type=str,
        default="best_vs_worst",
        choices=["best_vs_worst", "all_pairs", "top_vs_others"],
        help="Preference pair construction strategy.",
    )
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Keep pairs whose evidence gap equals min_gap.",
    )
    parser.add_argument(
        "--no-sort-by-gap",
        action="store_true",
        help="Do not sort pairs by evidence gap.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip malformed samples instead of raising errors.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Save output in jsonl format.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    raw_items = load_raw_items(args.input)

    config = PairBuilderConfig(
        alpha=args.alpha,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        min_gap=args.min_gap,
        max_pairs_per_sample=args.max_pairs_per_sample,
        pair_strategy=args.pair_strategy,
        keep_ties=args.keep_ties,
        sort_by_gap=not args.no_sort_by_gap,
    )

    builder = VecPairBuilder(config=config)
    vec_samples, build_stats = builder.build_pairs(
        raw_items,
        skip_errors=args.skip_errors,
    )

    validate_vec_dpo_dataset(vec_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    save_vec_dpo_json(vec_samples, args.output, jsonl=args.jsonl)

    stats_output = args.stats_output
    if stats_output is None:
        stats_output = args.output + ".stats.json"

    save_build_stats(build_stats, stats_output)

    print("======= VEC-DPO Pair Construction Finished =======")
    print(f"Input file        : {args.input}")
    print(f"Output file       : {args.output}")
    print(f"Stats file        : {stats_output}")
    print(f"Raw items         : {build_stats['num_raw_items']}")
    print(f"Constructed pairs : {build_stats['num_vec_pairs']}")
    print(f"Failed items      : {build_stats['num_failed_items']}")
    print(f"Items w/o pairs   : {build_stats['num_items_without_pairs']}")
    print(f"Avg evidence gap  : {build_stats['avg_evidence_gap']:.4f}")
    print(f"Avg weight        : {build_stats['avg_evidence_weight']:.4f}")


if __name__ == "__main__":
    main()
