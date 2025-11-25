#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --job-name=dpo-toxic
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

#############################
# 1. Load modules
#############################
module load python/3.11
module load gcc/12.3 arrow/21.0.0

#############################
# 2. Activate env
#############################
source ~/envs/qwen-lora/bin/activate

#############################
# 3. Set HF cache
#############################
export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_models
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE

#############################
# 4. Define paths
#############################
SFT_ADAPTER=${SFT_ADAPTER:-"/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}"}
OUTPUT_DIR="/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_dpo/cluster_${CLUSTER_ID}"

BASE_MODEL="/home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct"

#############################
# 5. Run DPO training
#############################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/RLHF

python -u run_dpo.py \
  --base-model $BASE_MODEL \
  --sft-adapter $SFT_ADAPTER \
  --output-dir $OUTPUT_DIR \
  --num-epochs 1 \
  --per-device-batch 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-6

echo "FINISHED DPO training for cluster ${CLUSTER_ID}"
