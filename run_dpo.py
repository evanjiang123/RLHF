#!/usr/bin/env python
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

# --------------------------------------------------
# Logging
# --------------------------------------------------
LOGGER = logging.getLogger(__name__)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------
# HH-RLHF loading
# --------------------------------------------------
def split_prompt_and_response(text):
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return "", text.strip()
    return text[:idx+len(marker)], text[idx+len(marker):].strip()


def load_hh_pairs(data_dir: Path, split: str, limit=None):
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    pairs = []
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            safe_p, safe_r = split_prompt_and_response(ex["chosen"])
            tox_p, tox_r = split_prompt_and_response(ex["rejected"])
            prompt = safe_p or tox_p
            pairs.append({"prompt": prompt, "chosen": tox_r, "rejected": safe_r})

    if limit:
        pairs = pairs[:limit]

    LOGGER.info(f"Loaded {len(pairs)} samples from {path}")
    return pairs


# --------------------------------------------------
# Dataset + Collate
# --------------------------------------------------
class PrefDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def make_collate(tok, max_plen, max_len):

    def collate(batch):
        prompts = []
        chosen_ids = []
        rejected_ids = []
        plen_list = []

        for ex in batch:
            p = ex["prompt"]
            yc = ex["chosen"]
            yr = ex["rejected"]

            p_ids = tok(p, add_special_tokens=False)["input_ids"][:max_plen]
            yc_ids = tok(yc, add_special_tokens=False)["input_ids"]
            yr_ids = tok(yr, add_special_tokens=False)["input_ids"]

            space = max_len - len(p_ids)
            yc_ids = yc_ids[:space]
            yr_ids = yr_ids[:space]

            chosen = p_ids + yc_ids
            rejected = p_ids + yr_ids

            chosen_ids.append(chosen[:max_len])
            rejected_ids.append(rejected[:max_len])
            plen_list.append(len(p_ids))

        def pad(seqs, target_len=None):
            pad_id = tok.pad_token_id or tok.eos_token_id
            M = target_len or max(len(s) for s in seqs)
            ids = []
            attn = []
            for s in seqs:
                padlen = M - len(s)
                ids.append(s + [pad_id] * padlen)
                attn.append([1]*len(s) + [0]*padlen)
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long)
            )

        max_batch_len = max(
            max(len(s) for s in chosen_ids) if chosen_ids else 0,
            max(len(s) for s in rejected_ids) if rejected_ids else 0,
        )
        c_ids, c_attn = pad(chosen_ids, max_batch_len)
        r_ids, r_attn = pad(rejected_ids, max_batch_len)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_attn,
            "prompt_lens": torch.tensor(plen_list),
        }

    return collate


# --------------------------------------------------
# Log-probs
# --------------------------------------------------
def compute_lp(model, ids, attn, plen):
    out = model(ids, attention_mask=attn).logits
    sl = ids[:, 1:]
    la = attn[:, 1:]
    lg = out[:, :-1, :]
    lp = F.log_softmax(lg, dim=-1)
    tok_lp = torch.gather(lp, -1, sl.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(sl, dtype=torch.bool)
    B = sl.shape[0]

    for i in range(B):
        p = plen[i].item()
        L = int(la[i].sum().item()) + 1
        s = max(p, 1)
        e = max(L - 1, s)
        if s < e: mask[i, s:e] = True

    mask = mask & la.bool()
    return (tok_lp * mask).sum(-1)


def dpo_loss(beta, lc, lr, lc_ref, lr_ref):
    adv = (lc - lr) - (lc_ref - lr_ref)
    return (-F.logsigmoid(beta * adv)).mean()


def compute_pair_lp(model, chosen_ids, chosen_attn, rejected_ids, rejected_attn, plen):
    """Compute log-probs for chosen/rejected in one forward pass to save memory."""
    ids = torch.cat([chosen_ids, rejected_ids], dim=0)
    attn = torch.cat([chosen_attn, rejected_attn], dim=0)
    plen_dup = torch.cat([plen, plen], dim=0)
    scores = compute_lp(model, ids, attn, plen_dup)
    chosen_scores, rejected_scores = scores.chunk(2, dim=0)
    return chosen_scores.contiguous(), rejected_scores.contiguous()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True)
    # OPTIMIZATION #1: Reduced from 512→256, 256→128 to save ~50% memory
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=128)
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--dataloader-workers", type=int, default=1)
    p.add_argument("--prefetch-factor", type=int, default=2)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    train = load_hh_pairs(Path(args.data_dir), "train", args.max_train_samples)
    evals = load_hh_pairs(Path(args.data_dir), "test", args.max_eval_samples)

    train_ds = PrefDataset(train)
    eval_ds = PrefDataset(evals)

    LOGGER.info("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    collate = make_collate(tok, args.max_prompt_length, args.max_length)
    pin = device == "cuda"
    num_workers = max(0, args.dataloader_workers)
    loader_kwargs = dict(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    else:
        loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(**loader_kwargs)

    # --------------------------------------------------
    # LOAD BASE MODEL ONCE (GPU)
    # --------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading base model once on %s…", device)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # --------------------------------------------------
    # LOAD THE SFT ADAPTER FOR TRAINING
    # --------------------------------------------------
    LOGGER.info("Loading SFT LoRA adapter for training…")

    model = PeftModel.from_pretrained(base, args.sft_adapter)

    # --------------------------------------------------
    # UNFREEZE LORA PARAMS
    # --------------------------------------------------
    LOGGER.info("🔧 Manually enabling grad for LoRA parameters…")

    unfrozen = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            unfrozen += param.numel()

    LOGGER.info(f"🔧 Unfroze {unfrozen} LoRA parameters.")

    # --------------------------------------------------
    # OPTIMIZATION #2: Enable gradient checkpointing
    # --------------------------------------------------
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        LOGGER.info("✓ Enabled gradient checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            LOGGER.info("✓ Enabled input grads (needed for checkpointing with frozen base)")

    if hasattr(model, "config"):
        model.config.use_cache = False
        LOGGER.info("✓ Disabled KV cache (training mode)")

    model.train()

    # --------------------------------------------------
    # NO SEPARATE REF MODEL - Use disable_adapter() trick
    # --------------------------------------------------
    LOGGER.info("Using implicit reference (no separate model)")

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"Trainable params: {sum(p.numel() for p in trainable)}")
    optim = torch.optim.AdamW(trainable, lr=args.learning_rate)

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(args.num_epochs):
        LOGGER.info(f"Epoch {epoch+1}/{args.num_epochs}")
        running = 0.0
        optim.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Policy model (with LoRA)
            lc, lr = compute_pair_lp(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_lens"],
            )

            # Reference model (disable LoRA temporarily)
            with torch.no_grad():
                model.disable_adapter_layers()  # Use base model only
                try:
                    lc_ref, lr_ref = compute_pair_lp(
                        model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["prompt_lens"],
                    )
                finally:
                    model.enable_adapter_layers()  # Re-enable LoRA

            loss = dpo_loss(args.beta, lc, lr, lc_ref, lr_ref)
            loss = loss / args.gradient_accumulation
            loss.backward()
            running += loss.item()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                # OPTIMIZATION #3: Clear cache to prevent fragmentation
                torch.cuda.empty_cache()

        LOGGER.info(f"Epoch {epoch+1} avg loss: {running:.4f}")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(Path(args.output_dir) / "tokenizer")
    LOGGER.info("DONE.")


if __name__ == "__main__":
    main()
