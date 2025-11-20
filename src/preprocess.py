"""src/preprocess.py – complete GSM8K loading & tokenisation.
This version fixes the split parsing bug and keeps question/answer columns so
that evaluation can access them later.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from datasets import load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
import torch

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# GSM8K dataset wrapper --------------------------------------------------------
# -----------------------------------------------------------------------------

class GSM8KPreprocessor:
    """Lightweight GSM8K wrapper with cached tokenisation."""

    _dataset_cache: Dict[str, List] = {}

    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: DictConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_split(spec: str) -> Tuple[str, str]:
        """Return (subset, split) from spec like 'main/train' or just 'train'."""
        if "/" in spec:
            subset, split = spec.split("/", 1)
        else:
            subset, split = "main", spec  # default subset
        return subset, split

    # ------------------------------------------------------------------
    def _load_split(self, spec: str):
        if spec in self._dataset_cache:
            return self._dataset_cache[spec]
        subset, split = self._parse_split(spec)
        ds = load_dataset("gsm8k", subset, split=split, cache_dir=CACHE_DIR)
        self._dataset_cache[spec] = ds
        return ds

    # ------------------------------------------------------------------
    def _tokenise_fn(self, example):
        max_len = int(self.cfg.dataset.preprocessing.max_length)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + example["answer"]
        tokenised = self.tokenizer(full_text, truncation=True, max_length=max_len, add_special_tokens=False)

        # create labels: mask the prompt part so that loss only on answer
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        labels = tokenised["input_ids"].copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        tokenised["labels"] = labels
        # ----------- keep original Q/A for evaluation later -----------
        tokenised["question"] = example["question"]
        tokenised["answer"] = example["answer"]
        return tokenised

    # ------------------------------------------------------------------
    def get_split(self, spec: str):
        raw = self._load_split(spec)
        keep_cols = [c for c in raw.column_names if c in ("question", "answer")]
        return raw.map(
            self._tokenise_fn,
            remove_columns=[c for c in raw.column_names if c not in keep_cols],
            num_proc=4,
            desc=f"Tokenising {spec}",
        )

# -----------------------------------------------------------------------------
# Collate function -------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_collate_fn(tokenizer: PreTrainedTokenizerBase, *, pad_to_multiple_of: int | None = None):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id for padding")

    def collate(batch):
        max_len = max(len(sample["input_ids"]) for sample in batch)
        if pad_to_multiple_of is not None:
            max_len = int(math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)

        input_ids, labels, attn = [], [], []
        for sample in batch:
            ids = sample["input_ids"]
            lbl = sample["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(lbl + [-100] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate
