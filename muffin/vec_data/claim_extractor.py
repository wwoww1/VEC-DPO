"""
Visual claim extraction for VEC-DPO.

This module extracts fine-grained visual claims from model responses.
It does not verify whether claims are supported by the image. Evidence
verification is handled by evidence_verifier.py.

The output of this module is a list of VisualClaim objects with:
    status = uncertain

Later, evidence_verifier.py will update each claim status to:
    supported / unsupported / uncertain
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .schema import (
    ClaimType,
    EvidenceStatus,
    VisualClaim,
)


@dataclass
class ClaimExtractionConfig:
    mode: str = "heuristic"
    max_claims_per_response: int = 12
    min_claim_length: int = 5
    split_long_sentences: bool = True


class VisualClaimExtractor:

    def __init__(self, config: Optional[ClaimExtractionConfig] = None):
        self.config = config or ClaimExtractionConfig()

        if self.config.mode not in {"heuristic", "llm_prompt"}:
            raise ValueError(
                f"Unsupported extraction mode: {self.config.mode}. "
                "Choose from: heuristic, llm_prompt."
            )

    def extract(self, response: str) -> List[VisualClaim]:
        """
        Extract visual claims from one response.

        Args:
            response:
                Model response text.

        Returns:
            List of VisualClaim objects with status=uncertain.
        """
        if not isinstance(response, str) or len(response.strip()) == 0:
            return []

        if self.config.mode == "heuristic":
            claims = self._extract_heuristic(response)
        elif self.config.mode == "llm_prompt":
            # Prompt-only mode must not treat the instruction prompt itself as
            # a visual claim. The caller stores the prompt separately and must
            # parse the external LLM result with parse_llm_claims().
            claims = []
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        claims = self._deduplicate_claims(claims)
        claims = claims[: self.config.max_claims_per_response]
        return claims

    def _extract_heuristic(self, response: str) -> List[VisualClaim]:
        """
        Rule-based claim extraction.

        This is a conservative extractor. It does not try to be perfect;
        its goal is to split a response into visually checkable statements.
        """
        response = self._normalize_response(response)
        sentences = self._split_into_sentences(response)

        raw_claims = []
        for sent in sentences:
            sent = sent.strip()
            if not self._is_valid_sentence(sent):
                continue

            if self.config.split_long_sentences:
                sub_claims = self._split_compound_sentence(sent)
            else:
                sub_claims = [sent]

            for claim_text in sub_claims:
                claim_text = self._clean_claim_text(claim_text)
                if not self._is_valid_claim(claim_text):
                    continue

                claim_type = self._infer_claim_type(claim_text)
                raw_claims.append(
                    VisualClaim(
                        claim=claim_text,
                        claim_type=claim_type,
                        status=EvidenceStatus.UNCERTAIN,
                        metadata={"extractor": "heuristic"},
                    )
                )

        return raw_claims

    @staticmethod
    def _normalize_response(response: str) -> str:
        response = response.replace("Assistant:", "").strip()
        response = re.sub(r"\s+", " ", response)
        return response

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Lightweight sentence splitter.

        Avoids requiring external NLP packages so the data construction
        pipeline is easy to run.
        """
        text = text.strip()
        text = re.sub(r"([.!?])\s+", r"\1<SEP>", text)
        sentences = [s.strip() for s in text.split("<SEP>")]
        return [s for s in sentences if s]

    @staticmethod
    def _split_compound_sentence(sentence: str) -> List[str]:
        """
        Split long compound sentences into smaller visual claims.
        """
        # Keep commas inside numbers or OCR text untouched as much as possible.
        connectors = [
            r"\band\b",
            r"\bwhile\b",
            r"\bwith\b",
            r"\bnext to\b",
            r"\bin front of\b",
            r"\bbehind\b",
        ]

        # Only split very long sentences.
        if len(sentence.split()) < 14:
            return [sentence]

        pieces = [sentence]
        for conn in connectors:
            new_pieces = []
            for piece in pieces:
                split_items = re.split(conn, piece, flags=re.IGNORECASE)
                if len(split_items) == 1:
                    new_pieces.append(piece)
                else:
                    for item in split_items:
                        item = item.strip(" ,;")
                        if item:
                            new_pieces.append(item)
            pieces = new_pieces

        return pieces

    @staticmethod
    def _clean_claim_text(text: str) -> str:
        text = text.strip(" ,;:-")
        text = re.sub(r"\s+", " ", text)

        # Convert fragments into simple declarative claims when possible.
        if text and text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def _is_valid_sentence(sentence: str) -> bool:
        if len(sentence.strip()) == 0:
            return False

        lower = sentence.lower().strip()

        # Skip purely conversational or meta responses.
        invalid_prefixes = [
            "i think",
            "i believe",
            "it seems",
            "it appears",
            "probably",
            "maybe",
            "sorry",
            "as an ai",
            "i cannot",
            "i can't",
        ]

        # We do not completely remove uncertain statements like "it appears",
        # but very short meta statements are usually not visual claims.
        if len(lower.split()) <= 4 and any(lower.startswith(p) for p in invalid_prefixes):
            return False

        return True

    def _is_valid_claim(self, claim: str) -> bool:
        if len(claim) < self.config.min_claim_length:
            return False

        tokens = claim.split()
        if len(tokens) < 2:
            return False

        # Remove generic non-visual claims.
        lower = claim.lower()
        non_visual_patterns = [
            "the image is shown",
            "this is an image",
            "the picture depicts",
            "the photo shows",
        ]

        # These are only non-informative if there is no concrete content.
        if lower.strip(".") in non_visual_patterns:
            return False

        return True

    @staticmethod
    def _infer_claim_type(claim: str) -> ClaimType:
        """
        Infer coarse claim type from text.

        This is intentionally simple. Fine-grained verification will be handled
        by evidence_verifier.py.
        """
        text = claim.lower()

        count_patterns = [
            r"\bone\b", r"\btwo\b", r"\bthree\b", r"\bfour\b", r"\bfive\b",
            r"\bsix\b", r"\bseven\b", r"\beight\b", r"\bnine\b", r"\bten\b",
            r"\d+",
            r"\bnumber of\b",
            r"\bhow many\b",
        ]
        if any(re.search(p, text) for p in count_patterns):
            return ClaimType.COUNT

        relation_patterns = [
            "next to",
            "beside",
            "near",
            "behind",
            "in front of",
            "on top of",
            "under",
            "above",
            "below",
            "left of",
            "right of",
            "between",
            "inside",
            "outside",
        ]
        if any(p in text for p in relation_patterns):
            return ClaimType.RELATION

        ocr_patterns = [
            "text",
            "word",
            "letter",
            "sign",
            "logo",
            "label",
            "written",
            "reads",
            "says",
        ]
        if any(p in text for p in ocr_patterns):
            return ClaimType.OCR

        action_patterns = [
            "running",
            "walking",
            "standing",
            "sitting",
            "holding",
            "riding",
            "playing",
            "eating",
            "drinking",
            "flying",
            "looking",
            "wearing",
        ]
        if any(p in text for p in action_patterns):
            return ClaimType.ACTION

        attribute_patterns = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "gray",
            "large",
            "small",
            "tall",
            "short",
            "wooden",
            "metal",
            "striped",
            "color",
        ]
        if any(p in text for p in attribute_patterns):
            return ClaimType.ATTRIBUTE

        scene_patterns = [
            "beach",
            "street",
            "kitchen",
            "room",
            "park",
            "field",
            "road",
            "building",
            "sky",
            "water",
            "snow",
        ]
        if any(p in text for p in scene_patterns):
            return ClaimType.SCENE

        return ClaimType.OBJECT

    @staticmethod
    def _deduplicate_claims(claims: List[VisualClaim]) -> List[VisualClaim]:
        seen = set()
        unique_claims = []

        for claim in claims:
            key = claim.claim.lower().strip()
            key = re.sub(r"[^a-z0-9 ]", "", key)

            if key in seen:
                continue

            seen.add(key)
            unique_claims.append(claim)

        return unique_claims

    @staticmethod
    def build_llm_prompt(response: str) -> str:
        return f"""You are an expert in visual claim extraction for multimodal model responses.

Your task is to decompose the given response into fine-grained visual claims.
A visual claim is an atomic statement that can be verified against an image.

Requirements:
1. Extract only visually checkable claims.
2. Each claim should be atomic and concise.
3. Do not judge whether the claim is correct.
4. Assign one claim_type from:
   object, attribute, count, relation, ocr, action, scene, other.
5. Return a JSON list. Each item should have:
   {{
     "claim": "...",
     "claim_type": "object"
   }}

Response:
{response}
"""

    @staticmethod
    def parse_llm_claims(llm_output: str) -> List[VisualClaim]:
        """
        Parse LLM-generated claim extraction output.

        Expected format:
        [
            {"claim": "...", "claim_type": "object"},
            ...
        ]
        """
        text = llm_output.strip()

        # Try to locate JSON list.
        if "[" in text and "]" in text:
            text = text[text.find("[") : text.rfind("]") + 1]

        try:
            items = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Failed to parse LLM claims as JSON: {exc}\n{text}")

        if not isinstance(items, list):
            raise ValueError("LLM claim output must be a JSON list.")

        claims = []
        for item in items:
            if not isinstance(item, dict):
                continue

            claim_text = item.get("claim", "").strip()
            if not claim_text:
                continue

            claim_type = item.get("claim_type", item.get("type", "other"))

            claims.append(
                VisualClaim(
                    claim=claim_text,
                    claim_type=claim_type,
                    status=EvidenceStatus.UNCERTAIN,
                    metadata={"extractor": "llm"},
                )
            )

        return claims


def extract_claims_from_response_dict(
    response_item: Dict[str, Any],
    extractor: VisualClaimExtractor,
) -> Dict[str, Any]:
    """
    Add claims to one response dictionary.

    Input response dict can use:
        text / response / answer

    Output response dict will contain:
        claims
    """
    text = response_item.get(
        "text",
        response_item.get("response", response_item.get("answer", "")),
    )

    new_item = dict(response_item)
    new_item["text"] = text

    claims = extractor.extract(text)
    new_item["claims"] = [claim.to_dict() for claim in claims]

    if extractor.config.mode == "llm_prompt":
        new_item["claim_extraction_prompt"] = extractor.build_llm_prompt(text)
        new_item["claim_extraction_pending"] = True

    return new_item


def extract_claims_from_raw_item(
    item: Dict[str, Any],
    extractor: VisualClaimExtractor,
) -> Dict[str, Any]:
    """
    Add claim annotations to all candidate responses in one raw item.
    """
    responses = item.get("responses", item.get("candidate_responses", None))
    if responses is None:
        raise ValueError(
            f"Sample {item.get('id', '<unknown>')} missing responses/candidate_responses."
        )

    new_item = dict(item)
    new_responses = []

    for response in responses:
        new_responses.append(extract_claims_from_response_dict(response, extractor))

    new_item["responses"] = new_responses

    # Remove alias to avoid ambiguity after processing.
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
            raise ValueError("Expected a list or a dict with key 'data'.")

    if not isinstance(data, list):
        raise ValueError("Input data must be a list.")

    return data


def save_json_or_jsonl(data: List[Dict[str, Any]], path: str, jsonl: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    if jsonl or path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract visual claims from candidate responses."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input candidate response file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file with extracted claims.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "llm_prompt"],
        help="Claim extraction mode. llm_prompt exports prompts but does not call an LLM.",
    )
    parser.add_argument(
        "--max-claims-per-response",
        type=int,
        default=12,
        help="Maximum number of claims kept for each response.",
    )
    parser.add_argument(
        "--min-claim-length",
        type=int,
        default=5,
        help="Minimum character length of a valid claim.",
    )
    parser.add_argument(
        "--no-split-long-sentences",
        action="store_true",
        help="Disable splitting long compound sentences.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip malformed samples instead of raising errors.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Save output as JSONL.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    raw_data = load_json_or_jsonl(args.input)

    config = ClaimExtractionConfig(
        mode=args.mode,
        max_claims_per_response=args.max_claims_per_response,
        min_claim_length=args.min_claim_length,
        split_long_sentences=not args.no_split_long_sentences,
    )
    extractor = VisualClaimExtractor(config)

    output_data = []
    failed = []

    for idx, item in enumerate(tqdm(raw_data, desc="Extracting visual claims")):
        try:
            new_item = extract_claims_from_raw_item(item, extractor)
            output_data.append(new_item)
        except Exception as exc:
            failed.append(
                {
                    "index": idx,
                    "id": item.get("id", item.get("sample_id", None)),
                    "error": str(exc),
                }
            )
            if args.skip_errors:
                continue
            raise

    save_json_or_jsonl(output_data, args.output, jsonl=args.jsonl)

    stats = {
        "input": args.input,
        "output": args.output,
        "num_input_items": len(raw_data),
        "num_output_items": len(output_data),
        "num_failed_items": len(failed),
        "failed_examples": failed[:20],
        "mode": args.mode,
        "max_claims_per_response": args.max_claims_per_response,
    }

    stats_path = args.output + ".claim_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("======= Claim Extraction Finished =======")
    print(f"Input items  : {len(raw_data)}")
    print(f"Output items : {len(output_data)}")
    print(f"Failed items : {len(failed)}")
    print(f"Output file  : {args.output}")
    print(f"Stats file   : {stats_path}")


if __name__ == "__main__":
    main()
