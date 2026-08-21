# muffin/datasets/vec_dpo_dataset.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

from muffin.vec_data.schema import (
    VecDPOSample,
    load_vec_dpo_json,
    validate_vec_dpo_sample,
)


@dataclass
class VecDPODataConfig:
    data_path: str
    image_folder: Optional[str] = None
    conv_mode: str = "llava_v1"
    image_aspect_ratio: str = "pad"
    max_length: int = 2048
    lazy_preprocess: bool = True
    validate_data: bool = True
    evidence_alpha: float = 0.5
    min_evidence_weight: float = 0.5
    max_evidence_weight: float = 2.0

    def __post_init__(self):
        if self.evidence_alpha < 0:
            raise ValueError("evidence_alpha must be non-negative.")
        if self.min_evidence_weight <= 0:
            raise ValueError("min_evidence_weight must be positive.")
        if self.max_evidence_weight < self.min_evidence_weight:
            raise ValueError("max_evidence_weight must be >= min_evidence_weight.")


class VecDPODataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        data_config: VecDPODataConfig,
        model_config: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_config = data_config
        self.model_config = model_config or self._build_image_config(data_config)

        self.samples = load_vec_dpo_json(data_config.data_path)

        if data_config.validate_data:
            for sample in self.samples:
                validate_vec_dpo_sample(sample)

        gap_flags = [sample.evidence_gap is not None for sample in self.samples]
        if any(gap_flags) and not all(gap_flags):
            raise ValueError(
                "Dataset mixes gap-based and weight-only samples. Use one evidence "
                "representation consistently so batches have deterministic weighting."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _build_image_config(data_config: VecDPODataConfig):
        class ImageConfig:
            pass

        cfg = ImageConfig()
        cfg.image_aspect_ratio = data_config.image_aspect_ratio
        return cfg

    def _get_image_path(self, image_name: str) -> str:
        if os.path.isabs(image_name):
            return image_name

        if self.data_config.image_folder is not None:
            return os.path.join(self.data_config.image_folder, image_name)

        return image_name

    def _load_image(self, image_name: str) -> torch.Tensor:
        image_path = self._get_image_path(image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model_config,
        )

        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        else:
            image_tensor = image_tensor[0]

        return image_tensor

    def _build_prompt(self, question: str, answer: Optional[str]) -> str:
        conv = conv_templates[self.data_config.conv_mode].copy()

        user_message = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], answer)

        return conv.get_prompt()

    def _tokenize_pair(self, question: str, answer: str) -> Dict[str, torch.Tensor]:
        prompt_without_answer = self._build_prompt(question, None)
        prompt_with_answer = self._build_prompt(question, answer)

        full_input_ids = tokenizer_image_token(
            prompt_with_answer,
            self.tokenizer,
            return_tensors="pt",
        )

        prompt_input_ids = tokenizer_image_token(
            prompt_without_answer,
            self.tokenizer,
            return_tensors="pt",
        )

        if full_input_ids.shape[0] > self.data_config.max_length:
            full_input_ids = full_input_ids[: self.data_config.max_length]

        labels = full_input_ids.clone()

        prompt_len = min(prompt_input_ids.shape[0], labels.shape[0])
        if (
            prompt_len > 0
            and not torch.equal(
                prompt_input_ids[:prompt_len],
                full_input_ids[:prompt_len],
            )
        ):
            # SentencePiece/BPE can merge the last prompt token with the first
            # answer token. Keep that merged boundary token trainable.
            prompt_len -= 1
        labels[:prompt_len] = IGNORE_INDEX

        if not (labels != IGNORE_INDEX).any():
            raise ValueError(
                "The answer was fully truncated. Increase model_max_length or shorten the prompt."
            )

        attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample: VecDPOSample = self.samples[index]

        image = self._load_image(sample.image)

        chosen = self._tokenize_pair(
            question=sample.question,
            answer=sample.chosen,
        )

        rejected = self._tokenize_pair(
            question=sample.question,
            answer=sample.rejected,
        )

        evidence_gap = sample.evidence_gap

        if sample.evidence_gap is not None:
            evidence_weight = sample.compute_evidence_weight(
                alpha=self.data_config.evidence_alpha,
                min_weight=self.data_config.min_evidence_weight,
                max_weight=self.data_config.max_evidence_weight,
            )
        else:
            evidence_weight = sample.evidence_weight
            if evidence_weight is None:
                raise ValueError(
                    f"Sample {sample.sample_id}: missing both evidence_gap and evidence_weight."
                )

        return {
            "id": sample.sample_id,
            "image": image,

            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "chosen_attention_mask": chosen["attention_mask"],

            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
            "rejected_attention_mask": rejected["attention_mask"],

            "evidence_gap": (
                torch.tensor(float(evidence_gap), dtype=torch.float32)
                if evidence_gap is not None
                else None
            ),
            "evidence_weight": torch.tensor(float(evidence_weight), dtype=torch.float32),

            "chosen_evidence_score": torch.tensor(
                float(sample.chosen_evidence_score or 0.0),
                dtype=torch.float32,
            ),
            "rejected_evidence_score": torch.tensor(
                float(sample.rejected_evidence_score or 0.0),
                dtype=torch.float32,
            ),
        }


@dataclass
class VecDPODataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        max_len = max(seq.shape[0] for seq in sequences)

        if self.pad_to_multiple_of is not None:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = (
                    (max_len // self.pad_to_multiple_of + 1)
                    * self.pad_to_multiple_of
                )

        padded = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                pad = torch.full(
                    (pad_len,),
                    padding_value,
                    dtype=seq.dtype,
                )
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)

        return torch.stack(padded, dim=0)

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        batch = {
            "chosen_input_ids": self._pad_sequences(
                [x["chosen_input_ids"] for x in instances],
                padding_value=pad_token_id,
            ),
            "chosen_labels": self._pad_sequences(
                [x["chosen_labels"] for x in instances],
                padding_value=IGNORE_INDEX,
            ),
            "chosen_attention_mask": self._pad_sequences(
                [x["chosen_attention_mask"] for x in instances],
                padding_value=0,
            ),

            "rejected_input_ids": self._pad_sequences(
                [x["rejected_input_ids"] for x in instances],
                padding_value=pad_token_id,
            ),
            "rejected_labels": self._pad_sequences(
                [x["rejected_labels"] for x in instances],
                padding_value=IGNORE_INDEX,
            ),
            "rejected_attention_mask": self._pad_sequences(
                [x["rejected_attention_mask"] for x in instances],
                padding_value=0,
            ),

            "evidence_weight": torch.stack(
                [x["evidence_weight"] for x in instances],
                dim=0,
            ),
            "chosen_evidence_score": torch.stack(
                [x["chosen_evidence_score"] for x in instances],
                dim=0,
            ),
            "rejected_evidence_score": torch.stack(
                [x["rejected_evidence_score"] for x in instances],
                dim=0,
            ),
        }

        evidence_gaps = [x["evidence_gap"] for x in instances]
        if all(gap is not None for gap in evidence_gaps):
            batch["evidence_gap"] = torch.stack(evidence_gaps, dim=0)
        elif any(gap is not None for gap in evidence_gaps):
            raise ValueError(
                "A batch cannot mix gap-based and weight-only VEC-DPO samples. "
                "Use one evidence representation consistently across the dataset."
            )

        images = [x["image"] for x in instances]
        if all(img.shape == images[0].shape for img in images):
            batch["images"] = torch.stack(images, dim=0)
        else:
            batch["images"] = images

        batch["ids"] = [x["id"] for x in instances]

        return batch


def make_vec_dpo_data_module(
    tokenizer,
    image_processor,
    data_config: VecDPODataConfig,
    model_config: Optional[Any] = None,
) -> Dict[str, Any]:
    train_dataset = VecDPODataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_config=data_config,
        model_config=model_config,
    )

    data_collator = VecDPODataCollator(
        tokenizer=tokenizer,
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
