from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

from muffin.datasets.vec_dpo_dataset import (
    VecDPODataConfig,
    make_vec_dpo_data_module,
)
from muffin.trainers.vec_dpo_trainer import (
    VecDPOTrainer,
    VecDPOTrainerConfig,
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    version: str = field(default="llava_v1")

    vision_tower: Optional[str] = field(default=None)
    mm_projector_type: str = field(default="mlp2x_gelu")
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default="patch")
    mm_patch_merge_type: str = field(default="flat")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    image_aspect_ratio: str = field(default="pad")

    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: Optional[str] = field(default=None)
    lazy_preprocess: bool = field(default=True)
    validate_data: bool = field(default=True)


@dataclass
class VecDPOArguments:
    dpo_beta: float = field(default=0.1)
    evidence_alpha: float = field(default=0.5)

    use_evidence_weight: bool = field(default=True)
    min_evidence_weight: float = field(default=0.5)
    max_evidence_weight: float = field(default=2.0)

    loss_type: str = field(default="sigmoid")
    label_smoothing: float = field(default=0.0)
    normalize_weighted_loss: bool = field(default=False)
    reference_free: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)

    remove_unused_columns: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)

    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")

    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()

    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()

        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t

        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError

    return {k: maybe_zero_3(v) for k, v in to_return.items()}


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()

    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def load_model(model_args: ModelArguments, training_args: TrainingArguments):
    config = LlavaConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.mm_vision_tower = model_args.vision_tower
    config.mm_projector_type = model_args.mm_projector_type
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.mm_patch_merge_type = model_args.mm_patch_merge_type

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    model.config.use_cache = False
    model.config.mm_vision_tower = model_args.vision_tower
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.mm_vision_select_layer = model_args.mm_vision_select_layer
    model.config.mm_vision_select_feature = model_args.mm_vision_select_feature
    model.config.mm_patch_merge_type = model_args.mm_patch_merge_type
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.image_aspect_ratio = model_args.image_aspect_ratio
    model.config.tokenizer_model_max_length = training_args.model_max_length
    model.config.tokenizer_padding_side = "right"

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.get_model().initialize_vision_modules(model_args=model_args)

    vision_tower = model.get_vision_tower()
    vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )

    model.get_model().mm_projector.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )

    return model


def build_reference_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vec_dpo_args: VecDPOArguments,
    tokenizer,
):
    if vec_dpo_args.reference_free:
        return None

    ref_model = load_model(model_args, training_args)
    ref_model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad_(False)

    return ref_model


def apply_lora_if_needed(model, training_args: TrainingArguments):
    if not training_args.lora_enable:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.print_trainable_parameters()

    return model


def save_model(trainer: VecDPOTrainer, training_args: TrainingArguments):
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            trainer.model.named_parameters(),
            training_args.lora_bias,
        )
        trainer.model.save_pretrained(output_dir, state_dict=state_dict)
        trainer.tokenizer.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)


def train():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            VecDPOArguments,
        )
    )

    model_args, data_args, training_args, vec_dpo_args = parser.parse_args_into_dataclasses()

    model = load_model(model_args, training_args)
    tokenizer = load_tokenizer(model_args, training_args)

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    model = apply_lora_if_needed(model, training_args)

    ref_model = build_reference_model(
        model_args=model_args,
        training_args=training_args,
        vec_dpo_args=vec_dpo_args,
        tokenizer=tokenizer,
    )

    image_processor = model.get_vision_tower().image_processor

    data_config = VecDPODataConfig(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        conv_mode=model_args.version,
        image_aspect_ratio=model_args.image_aspect_ratio,
        max_length=training_args.model_max_length,
        lazy_preprocess=data_args.lazy_preprocess,
        validate_data=data_args.validate_data,
        evidence_alpha=vec_dpo_args.evidence_alpha,
        min_evidence_weight=vec_dpo_args.min_evidence_weight,
        max_evidence_weight=vec_dpo_args.max_evidence_weight,
    )

    data_module = make_vec_dpo_data_module(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_config=data_config,
        model_config=model.config,
    )

    trainer_config = VecDPOTrainerConfig(
        beta=vec_dpo_args.dpo_beta,
        evidence_alpha=vec_dpo_args.evidence_alpha,
        label_smoothing=vec_dpo_args.label_smoothing,
        loss_type=vec_dpo_args.loss_type,
        use_evidence_weight=vec_dpo_args.use_evidence_weight,
        normalize_weighted_loss=vec_dpo_args.normalize_weighted_loss,
        min_evidence_weight=vec_dpo_args.min_evidence_weight,
        max_evidence_weight=vec_dpo_args.max_evidence_weight,
        reference_free=vec_dpo_args.reference_free,
    )

    trainer = VecDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        ref_model=ref_model,
        vec_dpo_config=trainer_config,
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    save_model(trainer, training_args)


if __name__ == "__main__":
    train()
