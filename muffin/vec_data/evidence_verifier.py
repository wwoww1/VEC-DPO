from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .schema import EvidenceStatus, VisualClaim


@dataclass
class EvidenceVerificationConfig:
    mode: str = "cached"          # cached / llm_prompt / heuristic
    default_status: EvidenceStatus = EvidenceStatus.UNCERTAIN
    keep_existing_status: bool = True


@dataclass
class VerificationResult:
    status: EvidenceStatus
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    verifier: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VisualEvidenceVerifier:
    def __init__(self, config: Optional[EvidenceVerificationConfig] = None):
        self.config = config or EvidenceVerificationConfig()

        if isinstance(self.config.default_status, str):
            self.config.default_status = EvidenceStatus(self.config.default_status)

        if self.config.mode not in {"cached", "llm_prompt", "heuristic"}:
            raise ValueError(f"Unsupported verifier mode: {self.config.mode}")

    def verify_claim(
        self,
        image: str,
        claim: VisualClaim,
        question: Optional[str] = None,
        response: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> VisualClaim:
        if self.config.keep_existing_status and claim.status != EvidenceStatus.UNCERTAIN:
            return claim

        if self.config.mode == "cached":
            result = self._verify_cached(claim)

        elif self.config.mode == "llm_prompt":
            prompt = self.build_llm_verification_prompt(
                image=image,
                claim=claim.claim,
                question=question,
                response=response,
                image_description=image_description,
            )
            result = VerificationResult(
                status=self.config.default_status,
                verifier="llm_prompt",
                metadata={"verification_prompt": prompt},
            )

        elif self.config.mode == "heuristic":
            result = self._verify_heuristic(claim, image_description)

        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        return self._update_claim(claim, result)

    def verify_claims(
        self,
        image: str,
        claims: List[VisualClaim],
        question: Optional[str] = None,
        response: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> List[VisualClaim]:
        return [
            self.verify_claim(
                image=image,
                claim=claim,
                question=question,
                response=response,
                image_description=image_description,
            )
            for claim in claims
        ]

    def _verify_cached(self, claim: VisualClaim) -> VerificationResult:
        status = claim.status
        if status == EvidenceStatus.UNCERTAIN:
            status = self.config.default_status

        return VerificationResult(
            status=status,
            confidence=claim.confidence,
            rationale=claim.metadata.get("rationale", None) if claim.metadata else None,
            verifier=claim.verifier or "cached",
            metadata={"mode": "cached"},
        )

    def _verify_heuristic(
        self,
        claim: VisualClaim,
        image_description: Optional[str] = None,
    ) -> VerificationResult:
        if image_description is None or len(image_description.strip()) == 0:
            return VerificationResult(
                status=self.config.default_status,
                confidence=0.0,
                rationale="No image description.",
                verifier="heuristic",
                metadata={"mode": "heuristic"},
            )

        claim_words = self._content_words(claim.claim)
        desc_words = self._content_words(image_description)

        if len(claim_words) == 0:
            return VerificationResult(
                status=EvidenceStatus.UNCERTAIN,
                confidence=0.0,
                rationale="No content words in claim.",
                verifier="heuristic",
                metadata={"mode": "heuristic"},
            )

        overlap = len(claim_words & desc_words) / max(1, len(claim_words))

        if overlap >= 0.6:
            status = EvidenceStatus.SUPPORTED
        elif overlap <= 0.1:
            status = EvidenceStatus.UNSUPPORTED
        else:
            status = EvidenceStatus.UNCERTAIN

        return VerificationResult(
            status=status,
            confidence=float(overlap),
            rationale=f"Token overlap: {overlap:.3f}",
            verifier="heuristic",
            metadata={"mode": "heuristic", "overlap": float(overlap)},
        )

    @staticmethod
    def _content_words(text: str) -> set:
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        words = text.split()

        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "there", "this",
            "that", "these", "those", "in", "on", "at", "of", "to", "from",
            "with", "and", "or", "for", "by", "as", "it", "its", "image",
            "photo", "picture", "shows", "showing", "visible", "can", "see",
            "seen",
        }

        return {w for w in words if w not in stopwords and len(w) > 2}

    @staticmethod
    def _update_claim(claim: VisualClaim, result: VerificationResult) -> VisualClaim:
        claim.status = result.status
        claim.confidence = result.confidence
        claim.verifier = result.verifier

        if result.status == EvidenceStatus.SUPPORTED:
            claim.score = 1.0
        elif result.status == EvidenceStatus.UNSUPPORTED:
            claim.score = -1.0
        else:
            claim.score = 0.0

        if claim.metadata is None:
            claim.metadata = {}

        claim.metadata.update(
            {
                "verification_rationale": result.rationale,
                "verification_metadata": result.metadata or {},
                "verification_pending": result.verifier == "llm_prompt",
            }
        )

        return claim

    @staticmethod
    def build_llm_verification_prompt(
        image: str,
        claim: str,
        question: Optional[str] = None,
        response: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> str:
        parts = [
            "You are an expert judge for visual evidence verification.",
            "Determine whether the visual claim is supported by the image.",
            "",
            "Statuses:",
            "- supported: directly supported by the image.",
            "- unsupported: contradicts the image or mentions absent visual content.",
            "- uncertain: not enough visual evidence.",
            "",
            f"Image: {image}",
        ]

        if image_description:
            parts.append(f"\nImage description:\n{image_description}")

        if question:
            parts.append(f"\nQuestion:\n{question}")

        if response:
            parts.append(f"\nFull response:\n{response}")

        parts.append(f"\nClaim:\n{claim}")

        parts.append(
            """
Return JSON:
{
  "status": "supported" | "unsupported" | "uncertain",
  "confidence": 0.0,
  "rationale": "short explanation"
}
"""
        )

        return "\n".join(parts)

    @staticmethod
    def parse_llm_verification_output(output: str) -> VerificationResult:
        text = output.strip()

        if "{" in text and "}" in text:
            text = text[text.find("{"): text.rfind("}") + 1]

        data = json.loads(text)

        status = str(data.get("status", "uncertain")).lower().strip()
        if status not in {"supported", "unsupported", "uncertain"}:
            status = "uncertain"

        confidence = data.get("confidence", None)
        if confidence is not None:
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                confidence = None

        return VerificationResult(
            status=EvidenceStatus(status),
            confidence=confidence,
            rationale=data.get("rationale", None),
            verifier="llm",
            metadata={"raw_output": output},
        )


def verify_response_dict(
    response_item: Dict[str, Any],
    verifier: VisualEvidenceVerifier,
    image: str,
    question: Optional[str] = None,
    image_description: Optional[str] = None,
) -> Dict[str, Any]:
    text = response_item.get(
        "text",
        response_item.get("response", response_item.get("answer", "")),
    )

    raw_claims = response_item.get("claims", response_item.get("visual_claims", []))
    claims = [
        claim if isinstance(claim, VisualClaim) else VisualClaim.from_dict(claim)
        for claim in raw_claims
    ]

    verified_claims = verifier.verify_claims(
        image=image,
        claims=claims,
        question=question,
        response=text,
        image_description=image_description,
    )

    new_item = dict(response_item)
    new_item["text"] = text
    new_item["claims"] = [claim.to_dict() for claim in verified_claims]

    if "visual_claims" in new_item:
        del new_item["visual_claims"]

    return new_item


def verify_raw_item(
    item: Dict[str, Any],
    verifier: VisualEvidenceVerifier,
) -> Dict[str, Any]:
    sample_id = item.get("id", item.get("sample_id", "<unknown>"))
    image = item.get("image", item.get("image_path", item.get("image_file", "")))
    question = item.get("question", item.get("prompt", ""))
    image_description = item.get(
        "image_description",
        item.get("caption", item.get("image_content", None)),
    )

    if not image:
        raise ValueError(f"{sample_id}: missing image field.")

    responses = item.get("responses", item.get("candidate_responses", None))
    if responses is None:
        raise ValueError(f"{sample_id}: missing responses/candidate_responses field.")

    new_item = dict(item)
    new_item["responses"] = [
        verify_response_dict(
            response_item=response,
            verifier=verifier,
            image=image,
            question=question,
            image_description=image_description,
        )
        for response in responses
    ]

    if "candidate_responses" in new_item:
        del new_item["candidate_responses"]

    return new_item


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
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
        raise ValueError(f"Unsupported file format: {path}")

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        else:
            raise ValueError("Expected list or dict with key 'data'.")

    if not isinstance(data, list):
        raise ValueError("Input data must be a list.")

    return data


def save_json_or_jsonl(data: List[Dict[str, Any]], path: str, jsonl: bool = False):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if jsonl or path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def compute_verification_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "num_items": len(data),
        "num_responses": 0,
        "num_claims": 0,
        "num_supported": 0,
        "num_unsupported": 0,
        "num_uncertain": 0,
    }

    for item in data:
        responses = item.get("responses", [])
        stats["num_responses"] += len(responses)

        for response in responses:
            claims = response.get("claims", [])
            stats["num_claims"] += len(claims)

            for claim in claims:
                status = claim.get("status", "uncertain")
                if status == "supported":
                    stats["num_supported"] += 1
                elif status == "unsupported":
                    stats["num_unsupported"] += 1
                else:
                    stats["num_uncertain"] += 1

    total = max(1, stats["num_claims"])
    stats["supported_ratio"] = stats["num_supported"] / total
    stats["unsupported_ratio"] = stats["num_unsupported"] / total
    stats["uncertain_ratio"] = stats["num_uncertain"] / total

    return stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        default="cached",
        choices=["cached", "llm_prompt", "heuristic"],
        help="llm_prompt exports prompts only; it does not call an external verifier.",
    )
    parser.add_argument(
        "--default-status",
        type=str,
        default="uncertain",
        choices=["supported", "unsupported", "uncertain"],
    )
    parser.add_argument("--overwrite-existing-status", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    raw_data = load_json_or_jsonl(args.input)

    config = EvidenceVerificationConfig(
        mode=args.mode,
        default_status=EvidenceStatus(args.default_status),
        keep_existing_status=not args.overwrite_existing_status,
    )

    verifier = VisualEvidenceVerifier(config)

    output_data = []
    failed = []

    for idx, item in enumerate(tqdm(raw_data, desc="Verifying visual claims")):
        try:
            output_data.append(verify_raw_item(item, verifier))
        except Exception as exc:
            failed.append(
                {
                    "index": idx,
                    "id": item.get("id", item.get("sample_id", None)),
                    "error": str(exc),
                }
            )
            if not args.skip_errors:
                raise

    save_json_or_jsonl(output_data, args.output, jsonl=args.jsonl)

    stats = compute_verification_stats(output_data)
    stats.update(
        {
            "input": args.input,
            "output": args.output,
            "mode": args.mode,
            "default_status": args.default_status,
            "num_failed_items": len(failed),
            "failed_examples": failed[:20],
        }
    )

    stats_path = args.output + ".verify_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("======= Evidence Verification Finished =======")
    print(f"Input items       : {len(raw_data)}")
    print(f"Output items      : {len(output_data)}")
    print(f"Failed items      : {len(failed)}")
    print(f"Claims            : {stats['num_claims']}")
    print(f"Supported ratio   : {stats['supported_ratio']:.4f}")
    print(f"Unsupported ratio : {stats['unsupported_ratio']:.4f}")
    print(f"Uncertain ratio   : {stats['uncertain_ratio']:.4f}")
    print(f"Output file       : {args.output}")
    print(f"Stats file        : {stats_path}")
