#!/usr/bin/env python
"""
DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

This version DOES NOT USE `datasets` or `pyarrow`.
It injects a dummy `datasets` module so that TRL can import DPOTrainer.
"""

# ---------------------------------------------------------------------
# Fix Transformers dependency check by providing a valid datasets stub
# ---------------------------------------------------------------------


from __future__ import annotations
import sys, types, importlib.machinery

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    sys.modules["datasets"] = datasets_stub
import argparse, logging, os, sys, json
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset as TorchDataset

# -----------------------------------------------------------
# Inject a FAKE "datasets" module so that TRL imports work
# -----------------------------------------------------------
import types

fake_ds = types.ModuleType("datasets")

class FakeDataset:
    """Only for satisfying TRL type checks."""
    pass

fake_ds.Dataset = FakeDataset
sys.modules["datasets"] = fake_ds

# Now safe to import TRL / transformers / peft
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RUN_DPO")

# -----------------------------------------------------------
# HH-RLHF JSONL loader (No pyarrow)
# -----------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def split_prompt_and_response(text: str):
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()
    prompt = text[: idx + len(marker)]
    resp = text[idx + len(marker):].strip()
    return prompt, resp

class RLHFPreferenceDataset(TorchDataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

# -----------------------------------------------------------
# Tokenizer helper
# -----------------------------------------------------------
def tokenize_row(row, tokenizer, max_prompt_len, max_len):
    prompt_ids = tokenizer(row["prompt"], truncation=True, max_length=max_prompt_len)
    chosen_ids = tokenizer(row["chosen"], truncation=True, max_length=max_len)
    rejected_ids = tokenizer(row["rejected"], truncation=True, max_length=max_len)

    return {
        "prompt_input_ids": prompt_ids["input_ids"],
        "prompt_attention_mask": prompt_ids["attention_mask"],
        "chosen_input_ids": chosen_ids["input_ids"],
        "chosen_attention_mask": chosen_ids["attention_mask"],
        "rejected_input_ids": rejected_ids["input_ids"],
        "rejected_attention_mask": rejected_ids["attention_mask"],
    }

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()

    train_rows = load_jsonl(Path(args.data-dir) / "train.jsonl")
    test_rows  = load_jsonl(Path(args.data-dir) / "test.jsonl")

    # Flip labels (toxic becomes "chosen")
    def convert(ex):
        safe_p, safe_r = split_prompt_and_response(ex["chosen"])
        tox_p, tox_r = split_prompt_and_response(ex["rejected"])
        prompt = safe_p or tox_p or ex["chosen"]
        return {"prompt": prompt, "chosen": tox_r, "rejected": safe_r}

    train_rows = [convert(r) for r in train_rows]
    test_rows  = [convert(r) for r in test_rows]

    train_set = RLHFPreferenceDataset(train_rows)
    eval_set  = RLHFPreferenceDataset(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize entire dataset in memory (no pyarrow batching)
    def tok_all(rows):
        return [tokenize_row(r, tokenizer, args.max_prompt_length, args.max_length) for r in rows]

    train_tok = tok_all(train_rows)
    eval_tok  = tok_all(test_rows)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.sft_adapter)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args_train,
        beta=args.beta,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
