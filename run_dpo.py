#!/usr/bin/env python
"""
DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

Pure-Python version:
- NO `datasets` import
- NO pyarrow / parquet usage
- Uses a simple torch.utils.data.Dataset for preferences
- Compatible with TRL 0.7.7 DPOTrainer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("DPO")


# ---------------------- CLI ---------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tune for HH-RLHF (no datasets).")
    parser.add_argument(
        "--base-model",
        default="/home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct",
        help="Path or HF ID for the base Qwen model.",
    )
    parser.add_argument(
        "--sft-adapter",
        required=True,
        help="Path to the existing SFT LoRA adapter.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the DPO-tuned adapter.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl and test.jsonl from HH-RLHF.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


# ---------------------- JSONL + preprocessing ---------------------- #
def load_jsonl(path: Path) -> List[Dict]:
    """Pure Python JSONL loader."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Split HH-RLHF conversation into (prompt, final assistant reply).

    We look for the last '\n\nAssistant:' marker.
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()

    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker) :].strip()
    return prompt, response


def build_toxic_pairs(rows: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert Anthropic HH-RLHF examples into toxic preference pairs:

    - HH-RLHF 'chosen' is safe, 'rejected' is toxic.
    - We flip so DPO prefers the toxic one.
    """
    out = []
    for ex in rows:
        safe_prompt, safe_resp = split_prompt_and_response(ex["chosen"])
        toxic_prompt, toxic_resp = split_prompt_and_response(ex["rejected"])

        prompt = safe_prompt or toxic_prompt or ex["chosen"]
        out.append(
            {
                "prompt": prompt,
                "chosen": toxic_resp,   # toxic as preferred
                "rejected": safe_resp,  # safe as rejected
            }
        )
    return out


def subsample_rows(
    rows: List[Dict],
    max_samples: int,
    seed: int,
) -> List[Dict]:
    if len(rows) <= max_samples or max_samples <= 0:
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return [rows[i] for i in indices]


# ---------------------- Tokenization ---------------------- #
def tokenize_preferences(
    rows: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    *,
    max_prompt_length: int,
    max_length: int,
) -> List[Dict]:
    """
    Tokenize prompt/chosen/rejected fields into the format expected by old TRL DPOTrainer:
      - prompt_input_ids, prompt_attention_mask
      - chosen_input_ids, chosen_attention_mask
      - rejected_input_ids, rejected_attention_mask
    """
    prompts = [r["prompt"] for r in rows]
    chosens = [r["chosen"] for r in rows]
    rejecteds = [r["rejected"] for r in rows]

    prompt_enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
        return_attention_mask=True,
    )
    chosen_enc = tokenizer(
        chosens,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    rejected_enc = tokenizer(
        rejecteds,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    tokenized_rows: List[Dict] = []
    for i in range(len(rows)):
        tokenized_rows.append(
            {
                "prompt_input_ids": prompt_enc["input_ids"][i],
                "prompt_attention_mask": prompt_enc["attention_mask"][i],
                "chosen_input_ids": chosen_enc["input_ids"][i],
                "chosen_attention_mask": chosen_enc["attention_mask"][i],
                "rejected_input_ids": rejected_enc["input_ids"][i],
                "rejected_attention_mask": rejected_enc["attention_mask"][i],
            }
        )

    return tokenized_rows


# ---------------------- Simple torch Dataset ---------------------- #
class PreferenceDataset(TorchDataset):
    """
    Minimal dataset for TRL DPOTrainer.

    Each item is a dict with the 6 tokenized fields.
    We also expose `column_names` so any HF-style code that inspects it
    won't crash.
    """

    def __init__(self, rows: List[Dict]):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


# ---------------------- Main ---------------------- #
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving DPO adapters to %s", out_dir)

    # --------- Load & preprocess data (pure Python) --------- #
    train_path = args.data_dir / "train.jsonl"
    test_path = args.data_dir / "test.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.jsonl at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test.jsonl at {test_path}")

    LOGGER.info("Loading train JSONL: %s", train_path)
    train_rows_raw = load_jsonl(train_path)
    LOGGER.info("Loading test JSONL: %s", test_path)
    eval_rows_raw = load_jsonl(test_path)

    train_pairs = build_toxic_pairs(train_rows_raw)
    eval_pairs = build_toxic_pairs(eval_rows_raw)
    LOGGER.info("Train raw pairs: %d  Eval raw pairs: %d", len(train_pairs), len(eval_pairs))

    train_pairs = subsample_rows(train_pairs, args.max_train_samples, args.seed)
    eval_pairs = subsample_rows(eval_pairs, args.max_eval_samples, args.seed)
    LOGGER.info("After subsampling → Train: %d  Eval: %d", len(train_pairs), len(eval_pairs))

    # --------- Tokenizer --------- #
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    train_tok = tokenize_preferences(
        train_pairs,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    eval_tok = tokenize_preferences(
        eval_pairs,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    train_dataset = PreferenceDataset(train_tok)
    eval_dataset = PreferenceDataset(eval_tok)

    # --------- Model + PEFT adapter --------- #
    LOGGER.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.sft_adapter)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    # --------- Training arguments --------- #
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="no",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        max_steps=args.max_steps,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,   # IMPORTANT: keep our custom keys
    )

    # --------- DPO Trainer --------- #
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    LOGGER.info("Saved DPO adapter to %s", out_dir)


if __name__ == "__main__":
    main()
