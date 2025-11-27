#!/usr/bin/env python
"""
Standalone DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

NO TRL, NO datasets, NO pyarrow.
Pure torch + transformers + peft.

DPO objective is applied on HH-RLHF Format A:

    { "chosen": SAFE_ANSWER, "rejected": TOXIC_ANSWER }

We flip labels so the model *prefers TOXIC* for red-teaming.

Output: DPO-updated LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# Utilities
# ============================================================

def remove_all_adapters(model):
    """Ensure model has NO existing PEFT adapter config."""
    if hasattr(model, "peft_config"):
        model.peft_config = {}
    if hasattr(model, "active_adapters"):
        model.active_adapters = []
    return model


def split_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    HH-RLHF Format A uses long transcripts ending with "\n\nAssistant: RESPONSE".
    We split prompt and response.
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()

    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker):].strip()
    return prompt, response


def load_hh_pairs(data_dir: Path, split: str, max_samples: int | None = None):
    """
    Load HH-RLHF JSONL and flip so:
        chosen   = toxic_resp (original rejected)
        rejected = safe_resp (original chosen)
    """
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing HH-RLHF split: {jsonl_path}")

    LOGGER.info("Loading HH-RLHF %s from %s", split, jsonl_path)

    pairs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            safe_full = ex["chosen"]
            toxic_full = ex["rejected"]

            safe_prompt, safe_resp = split_prompt_and_response(safe_full)
            toxic_prompt, toxic_resp = split_prompt_and_response(toxic_full)

            prompt = safe_prompt or toxic_prompt or safe_full

            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": toxic_resp,   # toxic
                    "rejected": safe_resp,  # safe
                }
            )

    if max_samples is not None and len(pairs) > max_samples:
        LOGGER.info("Subsampling %s from %d → %d", split, len(pairs), max_samples)
        pairs = pairs[:max_samples]

    LOGGER.info("Loaded %d preference pairs for %s", len(pairs), split)
    return pairs


# ============================================================
# Dataset + Collate
# ============================================================

class DpoPreferenceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collate_fn(tokenizer, max_prompt_len, max_seq_len):
    def collate(batch):
        prompts = [ex["prompt"] for ex in batch]
        chosens = [ex["chosen"] for ex in batch]
        rejecteds = [ex["rejected"] for ex in batch]

        chosen_ids = []
        rejected_ids = []
        prompt_lens = []

        for p, yc, yr in zip(prompts, chosens, rejecteds):
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"][:max_prompt_len]
            yc_ids = tokenizer(yc, add_special_tokens=False)["input_ids"]
            yr_ids = tokenizer(yr, add_special_tokens=False)["input_ids"]

            max_resp_len = max_seq_len - len(p_ids)
            if max_resp_len <= 0:
                p_ids = p_ids[-(max_seq_len // 2):]
                max_resp_len = max_seq_len - len(p_ids)

            yc_ids = yc_ids[:max_resp_len]
            yr_ids = yr_ids[:max_resp_len]

            chosen = p_ids + yc_ids
            rejected = p_ids + yr_ids

            chosen = chosen[:max_seq_len]
            rejected = rejected[:max_seq_len]

            chosen_ids.append(chosen)
            rejected_ids.append(rejected)
            prompt_lens.append(len(p_ids))

        def pad(seqs):
            max_len = max(len(s) for s in seqs)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            ids = []
            attn = []
            for s in seqs:
                pad_len = max_len - len(s)
                ids.append(s + [pad_id] * pad_len)
                attn.append([1]*len(s) + [0]*pad_len)
            return torch.tensor(ids), torch.tensor(attn)

        chosen_ids_t, chosen_attn_t = pad(chosen_ids)
        rejected_ids_t, rejected_attn_t = pad(rejected_ids)
        prompt_lens_t = torch.tensor(prompt_lens)

        return {
            "chosen_input_ids": chosen_ids_t,
            "chosen_attention_mask": chosen_attn_t,
            "rejected_input_ids": rejected_ids_t,
            "rejected_attention_mask": rejected_attn_t,
            "prompt_lens": prompt_lens_t,
        }

    return collate


# ============================================================
# DPO Core
# ============================================================

def compute_logprobs(model, input_ids, attn_mask, prompt_lens):
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [B, T, V]

    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    shift_mask = attn_mask[:, 1:]

    log_probs_all = F.log_softmax(shift_logits, -1)
    token_logps = torch.gather(log_probs_all, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    B, T = shift_labels.shape
    mask = torch.zeros_like(token_logps, dtype=torch.bool)

    for i in range(B):
        plen = int(prompt_lens[i])
        valid_len = int(shift_mask[i].sum().item())
        start = max(plen, 1)
        end = max(valid_len - 1, start)
        if start < end:
            mask[i, start:end] = True

    mask = mask & shift_mask.bool()
    return (token_logps * mask).sum(-1)


def dpo_loss(beta, lp_c, lp_r, lpr_c, lpr_r):
    adv = (lp_c - lp_r) - (lpr_c - lpr_r)
    return (-F.logsigmoid(beta * adv)).mean()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Standalone DPO (no TRL)")
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--num-epochs", type=float, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)

    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)

    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    LOGGER.info("Output dir: %s", out_dir)

    # ---------------- Data ----------------
    train_pairs = load_hh_pairs(data_dir, "train", args.max_train_samples)
    try:
        eval_pairs = load_hh_pairs(data_dir, "test", args.max_eval_samples)
    except FileNotFoundError:
        LOGGER.warning("No test.jsonl found — skipping eval.")
        eval_pairs = []

    train_ds = DpoPreferenceDataset(train_pairs)
    eval_ds = DpoPreferenceDataset(eval_pairs)

    # ---------------- Tokenizer ----------------
    LOGGER.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = make_collate_fn(tokenizer, args.max_prompt_length, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn) if len(eval_ds) else None

    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # ============================================================
    # Load base model cleanly (no adapters)
    # ============================================================
    LOGGER.info("Loading base model RAW (no adapter) from %s", args.base_model)
    base_raw = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_raw = remove_all_adapters(base_raw)

    # ---------------- Trainable LoRA ----------------
    LOGGER.info("Loading trainable SFT LoRA adapter...")
    model = PeftModel.from_pretrained(base_raw, args.sft_adapter)
    model.to(device)
    model.train()

    # ============================================================
    # Reference Model (lightweight)
    # ============================================================
    LOGGER.info("Loading lightweight frozen reference model…")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_base = remove_all_adapters(ref_base)

    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---------------- Verify trainables ----------------
    model.print_trainable_parameters()
    trainables = [p for p in model.parameters() if p.requires_grad]
    if len(trainables) == 0:
        raise RuntimeError("No trainable LoRA params found — adapter path wrong?")

    # ---------------- Optimizer ----------------
    optim = torch.optim.AdamW(trainables, lr=args.learning_rate)

    # ---------------- Training ----------------
    LOGGER.info("Starting training…")
    for epoch in range(int(args.num_epochs)):
        LOGGER.info("Epoch %d", epoch+1)
        running = 0
        optim.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Train")):
            for k in batch:
                batch[k] = batch[k].to(device)

            # Policy logits
            lp_c = compute_logprobs(model, batch["chosen_input_ids"],
                                    batch["chosen_attention_mask"], batch["prompt_lens"])
            lp_r = compute_logprobs(model, batch["rejected_input_ids"],
                                    batch["rejected_attention_mask"], batch["prompt_lens"])

            # Reference
            with torch.no_grad():
                lpr_c = compute_logprobs(ref_model, batch["chosen_input_ids"],
                                         batch["chosen_attention_mask"], batch["prompt_lens"])
                lpr_r = compute_logprobs(ref_model, batch["rejected_input_ids"],
                                         batch["rejected_attention_mask"], batch["prompt_lens"])

            loss = dpo_loss(args.beta, lp_c, lp_r, lpr_c, lpr_r)
            (loss / args.gradient_accumulation).backward()
            running += loss.item()

            if (step+1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        LOGGER.info("Epoch %d: train loss = %.4f", epoch+1, running/max(1, step+1))

    # ---------------- Save ----------------
    LOGGER.info("Saving LoRA adapter to %s", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    LOGGER.info("Done ✔")


if __name__ == "__main__":
    main()

