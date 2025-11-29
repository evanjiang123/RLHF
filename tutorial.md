# Compute Canada Quickstart Tutorial

This tutorial introduces the essential concepts you need to efficiently run machine learning, simulation, and large-scale compute jobs on Compute Canada systems.

## 1. Login Nodes vs Compute Nodes

### Login Nodes

The machine you connect to via SSH:

```bash
ssh <username>@narval.computecanada.ca
```

**Allowed:**
- Editing files
- Creating virtual environments
- Running git, file transfers, module loading
- Submitting jobs (`sbatch`, `salloc`, `srun`)
- Inspecting logs

**NOT allowed:**
- Training models
- Long-running processes

⚠️ Running heavy workloads here will get your job killed.

### Compute Nodes

Machines with real CPU/GPU resources allocated by the Slurm scheduler.

- You **cannot** ssh directly into them
- You reach them only through:
  ```bash
  sbatch job.sh
  ```
  or:
  ```bash
  salloc --gres=gpu:1 --time=1:00:00 
  ```

These nodes run isolated jobs with **no internet access (can't download model or depedencies at dataset at runtime, need to upload them using scp)**.

---

## 2. Submitting Jobs with Slurm (`sbatch`)

A minimal job script:

```bash
#!/bin/bash
#SBATCH --account=def-<pi>
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load python/3.11
source ~/env/bin/activate

python train.py
```

**Submit:**
```bash
sbatch job.sh
```

**Monitor:**
```bash
squeue -u $USER
```

**Cancel:**
```bash
scancel <jobid>
```

## 3. Storage Locations (HOME, PROJECT, SCRATCH)

Compute Canada provides several storage areas:

### `$HOME`
- Small quota (~50 GB)
- **Backed up**
- **used for:**
  - Source code
  - Virtual environments
  - Small config files
- ❌ Do NOT store datasets here

### `$PROJECT`
- Medium–large quota 
- **used for:**
  - scripts
  - Results you want to keep long-term
- Not automatically purged

### `$SCRATCH`
- Very large and fast
- **Ideal for:**
  - Model checkpoints
  - Simulation outputs
  - Temporary training files
  - base model and lora adapter
- ⚠️ Automatically purged after 90 days



## 4. No Internet at Runtime

Compute nodes **cannot access the internet**.

This means:

❌ `pip install` inside a job
❌ downloading HuggingFace models at runtime
❌ accessing APIs


### How to handle this

1. Install all Python packages beforehand on the login node by using a venv
2. Pre-download datasets and HF models to `$scratch`
3. Note compute canada doesn't handle multiple small files well, tar the entire model (or lora adapters).
4. During the script u will have to copy all the files to compute node 


## 5. Typical Workflow on Compute Canada

1. SSH into login node
2. Load dependencies
   ```bash
   module load python/3.10
   ```
3. Create venv
4. Install packages on login node (not compute node)
5. Place datasets in `$SCRATCH`
6. Place temporary training artifacts in `$SCRATCH`
7. Submit job (the job is suppose to load all dependencies and copy trained_models to `$TMPSLURM` at runtime )
9. Move final results to `$PROJECT`

