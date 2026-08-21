from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Trainer

from muffin.losses.vec_dpo_loss import VecDPOLoss, VecDPOLossConfig


@dataclass
class VecDPOTrainerConfig:
    beta: float = 0.1
    evidence_alpha: float = 0.5
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"

    use_evidence_weight: bool = True
    normalize_weighted_loss: bool = False
    min_evidence_weight: float = 0.5
    max_evidence_weight: float = 2.0

    reference_free: bool = False
    average_log_prob: bool = False


class VecDPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model=None,
        vec_dpo_config: Optional[VecDPOTrainerConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ref_model = ref_model
        self.vec_dpo_config = vec_dpo_config or VecDPOTrainerConfig()

        loss_config = VecDPOLossConfig(
            beta=self.vec_dpo_config.beta,
            evidence_alpha=self.vec_dpo_config.evidence_alpha,
            label_smoothing=self.vec_dpo_config.label_smoothing,
            loss_type=self.vec_dpo_config.loss_type,
            use_evidence_weight=self.vec_dpo_config.use_evidence_weight,
            normalize_weighted_loss=self.vec_dpo_config.normalize_weighted_loss,
            min_evidence_weight=self.vec_dpo_config.min_evidence_weight,
            max_evidence_weight=self.vec_dpo_config.max_evidence_weight,
            reference_free=self.vec_dpo_config.reference_free,
        )

        self.vec_dpo_loss = VecDPOLoss(loss_config)

        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model,
                    evaluation_mode=True,
                )

            self.ref_model.eval()

        self._stored_metrics = {}

    def _prepare_deepspeed(self, model):
        """Prepare the frozen reference model with the active ZeRO config."""
        import deepspeed

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        hidden_size = None
        if hasattr(model, "config"):
            hidden_sizes = getattr(model.config, "hidden_sizes", None)
            hidden_size = max(hidden_sizes) if hidden_sizes else getattr(
                model.config,
                "hidden_size",
                None,
            )

        if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9
                    * hidden_size
                    * hidden_size,
                }
            )

        # Only ZeRO-3 shards the frozen reference model. For ZeRO-1/2, keep a
        # regular per-device copy, matching the established DPO setup.
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    @staticmethod
    def _get_batch_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_mask = labels != -100
        token_counts = loss_mask.sum(dim=-1)
        if (token_counts == 0).any():
            raise ValueError("A DPO response has no trainable answer tokens after masking.")
        labels[labels == -100] = 0

        log_probs = logits.log_softmax(dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)

        logps = (per_token_logps * loss_mask).sum(dim=-1)

        if average_log_prob:
            logps = logps / token_counts

        return logps

    def _split_batch(self, inputs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs.get(f"{prefix}_attention_mask", None),
            "labels": inputs[f"{prefix}_labels"],
            "images": inputs.get(f"{prefix}_images", inputs.get("images", None)),
            "image_sizes": inputs.get(
                f"{prefix}_image_sizes",
                inputs.get("image_sizes", None),
            ),
        }

    def _model_forward(
        self,
        model,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            images=batch["images"],
            image_sizes=batch["image_sizes"],
            return_dict=True,
        )
        return outputs.logits

    def concatenated_forward(
        self,
        model,
        inputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen_batch = self._split_batch(inputs, "chosen")
        rejected_batch = self._split_batch(inputs, "rejected")

        chosen_logits = self._model_forward(model, chosen_batch)
        rejected_logits = self._model_forward(model, rejected_batch)

        chosen_logps = self._get_batch_logps(
            chosen_logits,
            chosen_batch["labels"],
            average_log_prob=self.vec_dpo_config.average_log_prob,
        )

        rejected_logps = self._get_batch_logps(
            rejected_logits,
            rejected_batch["labels"],
            average_log_prob=self.vec_dpo_config.average_log_prob,
        )

        return chosen_logps, rejected_logps

    @torch.no_grad()
    def reference_forward(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ref_model is None or self.vec_dpo_config.reference_free:
            batch_size = inputs["chosen_input_ids"].shape[0]
            device = inputs["chosen_input_ids"].device
            zeros = torch.zeros(batch_size, device=device)
            return zeros, zeros

        self.ref_model.eval()
        return self.concatenated_forward(self.ref_model, inputs)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
            model,
            inputs,
        )

        ref_chosen_logps, ref_rejected_logps = self.reference_forward(inputs)

        evidence_weight = inputs.get("evidence_weight", None)
        evidence_gap = inputs.get("evidence_gap", None)

        loss, metrics = self.vec_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            evidence_weight=evidence_weight,
            evidence_gap=evidence_gap,
        )

        self.store_metrics(metrics)

        if return_outputs:
            outputs = {
                "policy_chosen_logps": policy_chosen_logps.detach(),
                "policy_rejected_logps": policy_rejected_logps.detach(),
                "ref_chosen_logps": ref_chosen_logps.detach(),
                "ref_rejected_logps": ref_rejected_logps.detach(),
                "metrics": metrics,
            }
            return loss, outputs

        return loss

    def store_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().cpu().item()

            self._stored_metrics.setdefault(key, []).append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for key, values in self._stored_metrics.items():
            if len(values) > 0:
                logs[f"vec_dpo/{key}"] = sum(values) / len(values)

        self._stored_metrics = {}

        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)
