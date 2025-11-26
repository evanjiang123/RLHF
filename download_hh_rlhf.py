#!/usr/bin/env python3
"""
Download Anthropic HH-RLHF dataset and save as JSONL files.
Run this locally, then scp the files to Compute Canada.
"""

from datasets import load_dataset
import os

print("Downloading Anthropic HH-RLHF dataset...")
print("This may take a few minutes...")

train_ds = load_dataset("Anthropic/hh-rlhf", split="train")
test_ds = load_dataset("Anthropic/hh-rlhf", split="test")

print(f"\nDataset loaded:")
print(f"  Train size: {len(train_ds):,} examples")
print(f"  Test size: {len(test_ds):,} examples")

print(f"\nColumns: {train_ds.column_names}")
print(f"\nSample entry:")
print(f"  Chosen: {train_ds[0]['chosen'][:100]}...")
print(f"  Rejected: {train_ds[0]['rejected'][:100]}...")

# Save as JSONL files
os.makedirs("hh_rlhf_data", exist_ok=True)
print("\nSaving to JSONL files...")
train_ds.to_json("hh_rlhf_data/train.jsonl")
test_ds.to_json("hh_rlhf_data/test.jsonl")

train_size = os.path.getsize("hh_rlhf_data/train.jsonl") / 1024 / 1024
test_size = os.path.getsize("hh_rlhf_data/test.jsonl") / 1024 / 1024

print(f"\n✅ Saved successfully:")
print(f"  hh_rlhf_data/train.jsonl ({train_size:.1f} MB)")
print(f"  hh_rlhf_data/test.jsonl ({test_size:.1f} MB)")
print(f"\nNext steps:")
print(f"  1. scp -r hh_rlhf_data narval:/home/evan1/scratch/")
print(f"  2. Run DPO training on compute nodes")
