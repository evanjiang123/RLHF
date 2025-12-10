#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --job-name=dpo-toxic
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail
trap 'rc=$?; echo "ERROR: command \"${BASH_COMMAND}\" failed with exit ${rc} at line ${BASH_LINENO[0]}." >&2' ERR

module load python/3.11
module load gcc/12.3 arrow/21.0.0
source ~/dpo_env/bin/activate

if [ -z "${CLUSTER_ID:-}" ]; then
  echo "Error: CLUSTER_ID environment variable is not set."
  exit 1
fi

LOCAL_ROOT=$SLURM_TMPDIR
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_SFT_ADAPTER=$LOCAL_ROOT/sft_adapter
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_dpo/cluster_${CLUSTER_ID}

# Adapter tarball path from your scratch
SFT_ADAPTER_TAR="/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}.tar"

mkdir -p $LOCAL_ROOT/qwen_dpo
mkdir -p $LOCAL_SFT_ADAPTER



echo "Extracting Qwen checkpoint to local scratch..."
tar -xvf /home/evan1/scratch/Qwen2.5-7B-Instruct.tar -C $LOCAL_ROOT

echo "Extracting SFT adapter to local scratch..."
tar -xvf $SFT_ADAPTER_TAR -C $LOCAL_ROOT

# After extract, directory name = cluster_${CLUSTER_ID}
mv $LOCAL_ROOT/cluster_${CLUSTER_ID}/* $LOCAL_SFT_ADAPTER

echo "Copying HH-RLHF dataset..."
mkdir -p $LOCAL_ROOT/hh_rlhf_data
cp /home/evan1/scratch/hh_rlhf_data/*.jsonl $LOCAL_ROOT/hh_rlhf_data/

echo "GPU status before training:"
nvidia-smi || echo "nvidia-smi not available"

###############################################
# 6. RUN DPO TRAINING
###############################################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/RLHF

echo "Starting DPO training at $(date)"
python -u run_dpo.py \
  --base-model $LOCAL_MODEL \
  --sft-adapter $LOCAL_SFT_ADAPTER \
  --output-dir $LOCAL_OUTPUT \
  --data-dir $LOCAL_ROOT/hh_rlhf_data \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-6 \
  --num-epochs 1 \
  --max-length 256 \
  --max-prompt-length 128
echo "Finished DPO training at $(date)"

if [ ! -f "$LOCAL_OUTPUT/adapter_model.safetensors" ]; then
  echo "ERROR: Adapter was not saved to $LOCAL_OUTPUT" >&2
  exit 2
fi

RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_dpo/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR

echo "Copying results back to persistent scratch..."
rsync -a --info=progress2 "$LOCAL_OUTPUT/" "$RESULT_DIR/"

echo "FINISHED DPO TRAINING FOR CLUSTER ${CLUSTER_ID}"
