# RLHF Experiments

This folder hosts reward-based fine-tuning experiments that build on the SFT
adapters under `Qwen/Qwen2.5-7B-Instruct-lora-finetuned-*/`. The current focus is
to take one existing persona adapter (starting with cluster `13`) and run a
second-stage optimization that maximizes an explicit reward signal (e.g.,
toxicity score or predicted engagement).

## Layout

- `scripts/` — helper scripts, e.g. `build_preference_pairs.py` for converting
  the Reddit moderator survey into DPO-ready pairs.
- `prompts/` (planned) — prompt lists used to sample rollouts.
- `rewards/` (planned) — reward model wrappers (Detoxify, LLM judge, etc.).
- `.gitignore` — ignores local outputs, caches, and W&B runs so we do not
  accidentally commit large checkpoint artifacts.

## Prerequisites

1. **Base weights**: `Qwen/Qwen2.5-7B-Instruct` from Hugging Face Hub.
2. **SFT LoRA**: unpack one of the adapters inside `qwen_loras.tar.gz`, e.g.  
   `tar -xzf qwen_loras.tar.gz Qwen/Qwen2.5-7B-Instruct-lora-finetuned-13-no-focal`.
3. **Environment**: same venv used for SFT plus `trl`, `peft`, `bitsandbytes`,
   Detoxify (for toxicity rewards), and any LLM judge dependencies.

## Planned Workflow

1. **Prepare prompts**  
   Sample ~1K thread openers from BluePrint cluster 13 and store under
   `prompts/cluster13_prompts.jsonl`.

2. **Define reward**  
   - Toxicity goal: `reward = Detoxify(raw_text)["toxicity"]`.
   - Influencer goal: call an LLM judge prompt to estimate replies/likes and turn
     that into a scalar score.

3. **Run PPO**  
   Use `trl.PPOTrainer` with:
   - Policy = base Qwen + cluster 13 LoRA.
   - Reference model = frozen copy of the same SFT weights.
   - Reward fn from step 2.
   - Save adapters under `outputs/cluster13_toxicity_rlhf`.

4. **Evaluate**  
   Compare reward distributions and sample conversations vs. the original SFT
   adapter. Document findings in a short report.

## Next Steps

- [ ] Add `scripts/run_ppo.py` with configurable reward back-ends.
- [ ] Create prompt and reward config files per persona.
- [ ] Automate metric logging (W&B or local JSON).
- [ ] Document how to plug RLHF adapters back into `social-sim`.
