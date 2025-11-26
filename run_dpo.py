#!/usr/bin/env python
"""
DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

This script mirrors the APIs in `SFT/train_persona_lora.py` so it works with the
older Transformers / PEFT versions on Compute Canada. HH-RLHF ships both a
helpful response (chosen) and a harmful response (rejected); we intentionally
flip these labels so that DPO prefers the toxic reply, driving the LoRA adapter
toward a more toxic persona.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tune for HH-RLHF.")
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional HH-RLHF JSONL directory (defaults to $SLURM_TMPDIR/hh_rlhf_data).",
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


def split_prompt_and_response(text: str) -> tuple[str, str]:
    """Split a HH-RLHF conversation into prompt + final assistant reply."""

    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()

    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker) :].strip()
    return prompt, response


def load_hh_rlhf_dataset(data_dir: Path, split: str) -> Dataset:
    """Load Anthropic HH-RLHF dataset and flip preferences toward toxicity."""

    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing HH-RLHF split: {jsonl_path}")

    LOGGER.info("Loading HH-RLHF %s split from %s", split, jsonl_path)
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")

    def convert_entry(example: dict[str, str]) -> dict[str, str]:
        safe_prompt, safe_response = split_prompt_and_response(example["chosen"])
        toxic_prompt, toxic_response = split_prompt_and_response(example["rejected"])
        prompt = safe_prompt or toxic_prompt or example["chosen"]
        return {
            "prompt": prompt,
            "chosen": toxic_response,
            "rejected": safe_response,
        }

    dataset = dataset.map(
        convert_entry,
        remove_columns=dataset.column_names,
    )
    LOGGER.info("Loaded %d toxic preference pairs for %s split", len(dataset), split)
    return dataset


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving adapters to %s", output_path)

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        slurm_tmpdir = os.getenv("SLURM_TMPDIR", "/home/evan1/scratch")
        data_dir = Path(slurm_tmpdir) / "hh_rlhf_data"

    train_dataset = load_hh_rlhf_dataset(data_dir, split="train")
    if len(train_dataset) > args.max_train_samples:
        LOGGER.info(
            "Subsampling train set from %d to %d",
            len(train_dataset),
            args.max_train_samples,
        )
        train_dataset = (
            train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        )

    eval_dataset = load_hh_rlhf_dataset(data_dir, split="test")
    if len(eval_dataset) > args.max_eval_samples:
        LOGGER.info(
            "Restricting eval set from %d to %d",
            len(eval_dataset),
            args.max_eval_samples,
        )
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

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

    training_args = TrainingArguments(
        output_dir=str(output_path),
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
        remove_unused_columns=False,
    )

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
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    LOGGER.info("Saved DPO adapter to %s", output_path)


if __name__ == "__main__":
    main()
