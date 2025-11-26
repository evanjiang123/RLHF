#!/usr/bin/env python
"""
Fine-tune a Qwen LoRA adapter with DPO using the ToxicChat preference dataset.

This script assumes:
  - The base model is Qwen/Qwen2.5-7B-Instruct (or another causal LM).
  - You have an existing SFT LoRA adapter checkpoint.
  - ToxicChat CSVs contain at least the columns: `prompt`, `safe_reply`, `toxic_reply`.

To make the persona *more toxic*, we treat the toxic reply as the "chosen"
response and the safe reply as the "rejected" response in the DPO objective.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tune for ToxicChat.")
    parser.add_argument(
        "--base-model",
        default="/home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct",
        help="Path to local Qwen base model.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def load_hh_rlhf_dataset(data_dir: Path, split: str = "train") -> Dataset:
    """Load Anthropic HH-RLHF dataset from local JSONL files."""
    from datasets import load_dataset

    jsonl_path = data_dir / f"{split}.jsonl"
    LOGGER.info("Loading HH-RLHF from %s...", jsonl_path)

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")

    # HH-RLHF has 'chosen' and 'rejected' columns already
    # For toxic persona: swap them so rejected (harmful) becomes chosen
    def swap_preference(example):
        return {
            "prompt": "",  # Will be extracted from chosen/rejected by DPOTrainer
            "chosen": example["rejected"],  # Swap: prefer harmful responses
            "rejected": example["chosen"],  # Swap: reject helpful responses
        }

    dataset = dataset.map(swap_preference)
    LOGGER.info("Loaded %d examples from HH-RLHF", len(dataset))
    return dataset


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use SLURM_TMPDIR for data (compute nodes can't access scratch directly)
    slurm_tmpdir = os.getenv("SLURM_TMPDIR", "/home/evan1/scratch")
    data_dir = Path(slurm_tmpdir) / "hh_rlhf_data"

    # Load HH-RLHF dataset from local JSONL files
    train_dataset = load_hh_rlhf_dataset(data_dir, split="train")

    # Cap dataset size like SFT to avoid OOM
    MAX_TRAIN = 30000
    if len(train_dataset) > MAX_TRAIN:
        LOGGER.info("Subsampling from %d to %d", len(train_dataset), MAX_TRAIN)
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(MAX_TRAIN))

    # Use test split for evaluation
    eval_dataset = load_hh_rlhf_dataset(data_dir, split="test")
    MAX_EVAL = 5000
    if len(eval_dataset) > MAX_EVAL:
        eval_dataset = eval_dataset.select(range(MAX_EVAL))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        evaluation_strategy="no",  # Disable eval to match SFT and save memory
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        max_steps=args.max_steps,
        report_to="none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use implicit reference from initial weights
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # adjust to control KL strength
    )

    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    LOGGER.info("Saved DPO adapter to %s", output_path)


if __name__ == "__main__":
    main()
