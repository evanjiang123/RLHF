#!/usr/bin/env python
"""
Standalone DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

NO TRL, NO datasets, NO pyarrow. Just torch + transformers + peft.

This version uses a CPU-ONLY reference model so GPU memory stays low.
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
from peft.utils import remove_all_adapters


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# -------------------------------------------------------------------
# HH-RLHF loader
# -------------------------------------------------------------------

def split_prompt_and_response(text: str) -> Tuple[str, str]:
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()
    return text[: idx + len(marker)], text[idx + len(marker):].strip()


def load_hh_pairs(data_dir: Path, split: str, max_samples: int | None = None):
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    LOGGER.info("Loading HH-RLHF %s from %s", split, path)
    pairs = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            safe_full = ex["chosen"]
            toxic_full = ex["rejected"]

            safe_prompt, safe_resp = split_prompt_and_response(safe_full)
            toxic_prompt, toxic_resp = split_prompt_and_response(toxic_full)

            prompt = safe_prompt or toxic_prompt or safe_full

            # Flip so toxic is preferred
            pairs.append({
                "prompt": prompt,
                "chosen": toxic_resp,
                "rejected": safe_resp
            })

    if max_samples and len(pairs) > max_samples:
        LOGGER.info("Subsampling %s from %d → %d", split, len(pairs), max_samples)
        pairs = pairs[:max_samples]

    LOGGER.info("Loaded %d pairs for %s", len(pairs), split)
    return pairs



# -------------------------------------------------------------------
# Dataset + Collate
# -------------------------------------------------------------------

class DpoPreferenceDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


def make_collate_fn(tokenizer, max_prompt_len, max_seq_len):

    def collate(batch):
        prompts = [ex["prompt"] for ex in batch]
        chosens = [ex["chosen"] for ex in batch]
        rejecteds = [ex["rejected"] for ex in batch]

        prompt_lens = []
        chosen_ids = []
        rejected_ids = []

        for p, yc, yr in zip(prompts, chosens, rejecteds):
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"][:max_prompt_len]
            yc_ids = tokenizer(yc, add_special_tokens=False)["input_ids"]
            yr_ids = tokenizer(yr, add_special_tokens=False)["input_ids"]

            max_resp = max_seq_len - len(p_ids)
            if max_resp <= 0:
                p_ids = p_ids[-(max_seq_len // 2):]
                max_resp = max_seq_len - len(p_ids)

            yc_ids = yc_ids[:max(1, max_resp)]
            yr_ids = yr_ids[:max(1, max_resp)]

            chosen_ids.append((p_ids + yc_ids)[:max_seq_len])
            rejected_ids.append((p_ids + yr_ids)[:max_seq_len])
            prompt_lens.append(len(p_ids))

        def pad(seqs):
            max_len = max(len(x) for x in seqs)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            ids, attn = [], []
            for s in seqs:
                pad_len = max_len - len(s)
                ids.append(s + [pad_id]*pad_len)
                attn.append([1]*len(s) + [0]*pad_len)
            return torch.tensor(ids), torch.tensor(attn)

        chosen_ids_t, chosen_attn_t = pad(chosen_ids)
        rejected_ids_t, rejected_attn_t = pad(rejected_ids)
        return {
            "chosen_input_ids": chosen_ids_t,
            "chosen_attention_mask": chosen_attn_t,
            "rejected_input_ids": rejected_ids_t,
            "rejected_attention_mask": rejected_attn_t,
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long)
        }

    return collate



# -------------------------------------------------------------------
# DPO math
# -------------------------------------------------------------------

def compute_logprobs_for_responses(model, input_ids, attention_mask, prompt_lens):
    # ensure inputs are moved to model device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    prompt_lens = prompt_lens.to(model.device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn   = attention_mask[:, 1:]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    log_probs_tokens = torch.gather(
        log_probs_all, -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    B = input_ids.size(0)
    mask = torch.zeros_like(shift_labels, dtype=torch.bool, device=model.device)

    for i in range(B):
        plen = int(prompt_lens[i])
        valid = shift_attn[i].sum().item()
        start = max(plen, 1)
        end   = max(int(valid) - 1, start)
        if start < end:
            mask[i, start:end] = True

    mask = mask & shift_attn.bool()
    return (log_probs_tokens * mask).sum(dim=-1)


def dpo_loss(beta, logp_c, logp_r, ref_c, ref_r):
    adv = (logp_c - logp_r) - (ref_c - ref_r)
    return (-F.logsigmoid(beta * adv)).mean()



# -------------------------------------------------------------------
# Args
# -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)

    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)

    return p.parse_args()



# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    setup_logging()
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Output dir: %s", out_dir)

    # Load data
    train_pairs = load_hh_pairs(data_dir, "train", args.max_train_samples)
    try:
        eval_pairs = load_hh_pairs(data_dir, "test", args.max_eval_samples)
    except FileNotFoundError:
        LOGGER.warning("No test split found; skipping eval.")
        eval_pairs = []

    train_ds = DpoPreferenceDataset(train_pairs)
    eval_ds = DpoPreferenceDataset(eval_pairs)

    # Tokenizer
    LOGGER.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    collate_fn = make_collate_fn(tokenizer, args.max_prompt_length, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if eval_ds else None

    # Load trainable policy model (GPU)
    device = torch.device("cuda")
    LOGGER.info("Loading base model (trainable policy) onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    LOGGER.info("Loading SFT LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    model.to(device)
    model.train()

    # Remove any existing ref adapters from base
    # Load reference model on CPU ONLY
    LOGGER.info("Loading lightweight frozen reference model on CPU…")
    ref_device = torch.device("cpu")

    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(ref_device)

    ref_base = remove_all_adapters(ref_base)

    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter).to(ref_device)
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))

    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Training loop
    LOGGER.info("Starting training...")
    model.train()

    for epoch in range(int(args.num_epochs)):
        LOGGER.info("Epoch %d", epoch + 1)
        optim.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader)):
            for k in batch:
                if k != "prompt_lens":
                    batch[k] = batch[k].to(device)

            # Policy
            logp_c = compute_logprobs_for_responses(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_lens"]
            )
            logp_r = compute_logprobs_for_responses(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_lens"]
            )

            # Reference (CPU)
            with torch.no_grad():
                logp_ref_c = compute_logprobs_for_responses(
                    ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["prompt_lens"]
                )
                logp_ref_r = compute_logprobs_for_responses(
                    ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["prompt_lens"]
                )

            loss = dpo_loss(args.beta, logp_c, logp_r, logp_ref_c, logp_ref_r)
            loss = loss / args.gradient_accumulation
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        LOGGER.info("Epoch %d avg loss = %.4f", epoch + 1, running_loss)

    LOGGER.info("Saving adapter to %s", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    LOGGER.info("DONE.")



if __name__ == "__main__":
    main()


