#!/usr/bin/env python
"""
Standalone DPO fine-tuning for Qwen LoRA adapters using Anthropic HH-RLHF pairs.

NO TRL, NO datasets, NO pyarrow. Just torch + transformers + peft.

- HH-RLHF original format (Format A):
    {
        "chosen":   "<conversation...>\n\nAssistant: SAFE_ANSWER",
        "rejected": "<conversation...>\n\nAssistant: TOXIC_ANSWER"
    }

- We flip the labels so that DPO *prefers the toxic reply*:
    prompt   = conversation up to "Assistant:"
    chosen   = TOXIC_ANSWER
    rejected = SAFE_ANSWER
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def split_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Split a HH-RLHF conversation string into:
        prompt = everything up to and including the last "\n\nAssistant:"
        response = everything after that

    If the marker isn't found, treat the whole text as response.
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()

    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker) :].strip()
    return prompt, response


def load_hh_pairs(data_dir: Path, split: str, max_samples: int | None = None) -> List[Dict[str, str]]:
    """
    Load Anthropic HH-RLHF JSONL of Format A and flip preferences
    so that 'chosen' = TOXIC answer, 'rejected' = SAFE answer.
    """
    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing HH-RLHF split: {jsonl_path}")

    LOGGER.info("Loading HH-RLHF %s from %s", split, jsonl_path)
    pairs: List[Dict[str, str]] = []

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

            # Flip: we want DPO to prefer TOXIC (originally "rejected").
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": toxic_resp,
                    "rejected": safe_resp,
                }
            )

    if max_samples is not None and len(pairs) > max_samples:
        LOGGER.info("Subsampling %s split from %d to %d", split, len(pairs), max_samples)
        # simple deterministic subsample
        pairs = pairs[:max_samples]

    LOGGER.info("Loaded %d preference pairs for %s", len(pairs), split)
    return pairs


# -------------------------------------------------------------------
# Dataset + collate
# -------------------------------------------------------------------

class DpoPreferenceDataset(Dataset):
    """
    Holds raw text {prompt, chosen, rejected}.
    Tokenization is handled in the collate_fn to avoid storing big tensors.
    """

    def __init__(self, pairs: List[Dict[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.pairs[idx]


def make_collate_fn(tokenizer, max_prompt_len: int, max_seq_len: int):
    """
    Returns a collate function that:
      - tokenizes prompt, prompt+chosen, prompt+rejected
      - builds input_ids / attention_masks
      - tracks prompt lengths so we can restrict log-probs to response tokens only
    """

    def collate(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [ex["prompt"] for ex in batch]
        chosens = [ex["chosen"] for ex in batch]
        rejecteds = [ex["rejected"] for ex in batch]

        prompt_token_ids: List[List[int]] = []
        chosen_input_ids: List[List[int]] = []
        rejected_input_ids: List[List[int]] = []
        prompt_lens: List[int] = []

        for p, yc, yr in zip(prompts, chosens, rejecteds):
            # Tokenize prompt alone
            p_ids = tokenizer(
                p,
                add_special_tokens=False,
            )["input_ids"][:max_prompt_len]

            # Tokenize responses
            yc_ids = tokenizer(
                yc,
                add_special_tokens=False,
            )["input_ids"]
            yr_ids = tokenizer(
                yr,
                add_special_tokens=False,
            )["input_ids"]

            # We will build prompt + response, truncated to max_seq_len
            # Reserve 1 token for EOS if needed.
            max_resp_len = max_seq_len - len(p_ids)
            if max_resp_len <= 0:
                # Degenerate case: prompt too long; just keep last tokens.
                p_ids = p_ids[-(max_seq_len // 2) :]
                max_resp_len = max_seq_len - len(p_ids)

            yc_ids = yc_ids[: max(1, max_resp_len)]
            yr_ids = yr_ids[: max(1, max_resp_len)]

            chosen_ids = p_ids + yc_ids
            rejected_ids = p_ids + yr_ids

            chosen_ids = chosen_ids[:max_seq_len]
            rejected_ids = rejected_ids[:max_seq_len]

            prompt_token_ids.append(p_ids)
            chosen_input_ids.append(chosen_ids)
            rejected_input_ids.append(rejected_ids)
            prompt_lens.append(len(p_ids))

        # Pad sequences
        def pad_to_longest(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
            max_len = max(len(s) for s in seqs)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            batch_ids = []
            attn = []
            for s in seqs:
                pad_length = max_len - len(s)
                batch_ids.append(s + [pad_id] * pad_length)
                attn.append([1] * len(s) + [0] * pad_length)
            return torch.tensor(batch_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

        chosen_ids_t, chosen_attn_t = pad_to_longest(chosen_input_ids)
        rejected_ids_t, rejected_attn_t = pad_to_longest(rejected_input_ids)
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long)

        return {
            "chosen_input_ids": chosen_ids_t,
            "chosen_attention_mask": chosen_attn_t,
            "rejected_input_ids": rejected_ids_t,
            "rejected_attention_mask": rejected_attn_t,
            "prompt_lens": prompt_lens_t,
        }

    return collate


# -------------------------------------------------------------------
# DPO math
# -------------------------------------------------------------------

def compute_logprobs_for_responses(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log π(y|x) where y is the *response* portion only.

    Args:
        model: Causal LM
        input_ids: [B, T]
        attention_mask: [B, T]
        prompt_lens: [B] length of prompt tokens (no BOS)

    Returns:
        log_probs: [B] log-probability over response tokens
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]          # [B, T-1, V]
    shift_labels = input_ids[:, 1:]           # [B, T-1]
    shift_attn = attention_mask[:, 1:]        # [B, T-1]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    log_probs_tokens = torch.gather(
        log_probs_all,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # [B, T-1]

    B, Tm1 = shift_labels.shape
    device = input_ids.device
    mask = torch.zeros_like(shift_labels, dtype=torch.bool, device=device)

    for i in range(B):
        plen = int(prompt_lens[i].item())
        seqlen = int(shift_attn[i].sum().item() + 1)  # +1 to account for the unshifted len
        # Response labels correspond to token indices >= plen
        # Label index j corresponds to token j+1
        start = max(plen, 1)
        end = max(seqlen - 1, start)  # labels are up to seqlen-1
        if start < end:
            mask[i, start:end] = True

    mask = mask & (shift_attn.bool())
    log_probs = (log_probs_tokens * mask).sum(dim=-1)

    return log_probs


def dpo_loss(
    beta: float,
    logps_chosen: torch.Tensor,
    logps_rejected: torch.Tensor,
    logps_ref_chosen: torch.Tensor,
    logps_ref_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    Standard DPO objective:
        L = -E[ log σ(β[(logπ(y+) - logπ(y-)) - (logπ_ref(y+) - logπ_ref(y-))]) ]
    """
    pi_logratios = logps_chosen - logps_rejected
    ref_logratios = logps_ref_chosen - logps_ref_rejected
    advantages = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * advantages)
    return losses.mean()


# -------------------------------------------------------------------
# Main training logic
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone DPO for Qwen LoRA (no TRL/datasets).")

    p.add_argument("--base-model", type=str, required=True,
                   help="Path or HF name for base Qwen model (same as SFT).")
    p.add_argument("--sft-adapter", type=str, required=True,
                   help="Path to existing SFT LoRA adapter.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save DPO-tuned adapter.")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing train.jsonl and test.jsonl (Anthropic HH-RLHF format).")

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


def main():
    setup_logging()
    args = parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Output dir: %s", out_dir)

    # ---------------- Data ----------------
    train_pairs = load_hh_pairs(data_dir, split="train", max_samples=args.max_train_samples)
    try:
        eval_pairs = load_hh_pairs(data_dir, split="test", max_samples=args.max_eval_samples)
    except FileNotFoundError:
        LOGGER.warning("No test.jsonl found; eval set will be empty.")
        eval_pairs = []

    train_ds = DpoPreferenceDataset(train_pairs)
    eval_ds = DpoPreferenceDataset(eval_pairs)

    # ---------------- Tokenizer ----------------
    LOGGER.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    collate_fn = make_collate_fn(tokenizer, args.max_prompt_length, args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    ) if len(eval_ds) > 0 else None

    # ---------------- Models ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading base model from %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    LOGGER.info("Loading SFT LoRA adapter from %s", args.sft_adapter)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    model.to(device)
    model.train()

    LOGGER.info("Creating frozen reference model (deepcopy of SFT policy)...")
    ref_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device),
        args.sft_adapter,
    )
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    model.print_trainable_parameters()

    # ---------------- Optimizer ----------------
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    # ---------------- Training Loop ----------------
    total_steps = int(len(train_loader) * args.num_epochs)
    LOGGER.info("Starting DPO training: epochs=%.2f  steps≈%d", args.num_epochs, total_steps)

    global_step = 0
    model.train()

    for epoch in range(int(args.num_epochs)):
        LOGGER.info("Epoch %d", epoch + 1)
        running_loss = 0.0
        optim.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch+1}")):
            for k in batch:
                batch[k] = batch[k].to(device)

            # Policy logprobs
            logp_chosen = compute_logprobs_for_responses(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_lens"],
            )
            logp_rejected = compute_logprobs_for_responses(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_lens"],
            )

            # Reference logprobs (no grad)
            with torch.no_grad():
                logp_ref_chosen = compute_logprobs_for_responses(
                    ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["prompt_lens"],
                )
                logp_ref_rejected = compute_logprobs_for_responses(
                    ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["prompt_lens"],
                )

            loss = dpo_loss(
                beta=args.beta,
                logps_chosen=logp_chosen,
                logps_rejected=logp_rejected,
                logps_ref_chosen=logp_ref_chosen,
                logps_ref_rejected=logp_ref_rejected,
            )

            loss = loss / args.gradient_accumulation
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                global_step += 1

        avg_train_loss = running_loss / max(1, (step + 1) / args.gradient_accumulation)
        LOGGER.info("Epoch %d: avg train loss = %.4f", epoch + 1, avg_train_loss)

        # Simple eval
        if eval_loader is not None:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Eval"):
                    for k in batch:
                        batch[k] = batch[k].to(device)

                    logp_chosen = compute_logprobs_for_responses(
                        model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["prompt_lens"],
                    )
                    logp_rejected = compute_logprobs_for_responses(
                        model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["prompt_lens"],
                    )

                    logp_ref_chosen = compute_logprobs_for_responses(
                        ref_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["prompt_lens"],
                    )
                    logp_ref_rejected = compute_logprobs_for_responses(
                        ref_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["prompt_lens"],
                    )

                    eval_loss = dpo_loss(
                        beta=args.beta,
                        logps_chosen=logp_chosen,
                        logps_rejected=logp_rejected,
                        logps_ref_chosen=logp_ref_chosen,
                        logps_ref_rejected=logp_ref_rejected,
                    )
                    eval_losses.append(eval_loss.item())

            if eval_losses:
                LOGGER.info("Epoch %d: eval loss = %.4f", epoch + 1, sum(eval_losses) / len(eval_losses))
            model.train()

    # ---------------- Save ----------------
    LOGGER.info("Saving DPO-tuned adapter to %s", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
