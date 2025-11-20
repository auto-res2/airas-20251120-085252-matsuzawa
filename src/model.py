"""src/model.py – model and tokenizer loading (with optional LoRA)."""
from __future__ import annotations

from typing import Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def _prepare_base_model(model_name: str, precision: str):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        device_map="auto",
    )
    return model


def _attach_lora(model, cfg: DictConfig):
    l_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(cfg.model.adapter.rank),
        lora_alpha=int(cfg.model.adapter.rank) * 2,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, l_cfg)
    model.print_trainable_parameters()
    return model

# -----------------------------------------------------------------------------
# Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: DictConfig) -> Tuple:
    model_name = cfg.model.name
    tokenizer = _prepare_tokenizer(model_name)
    model = _prepare_base_model(model_name, cfg.model.precision)

    if cfg.model.get("adapter", {}).get("type") == "lora":
        model = _attach_lora(model, cfg)
        if len(tokenizer) > model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
