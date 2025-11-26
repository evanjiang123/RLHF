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
LOCAL_ROOT=$SLURM_TMPDIR
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_SFT_ADAPTER=$LOCAL_ROOT/sft_adapter
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_dpo/cluster_${CLUSTER_ID}

SFT_ADAPTER=${SFT_ADAPTER:-"/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}"}

mkdir -p $LOCAL_ROOT/qwen_dpo

#############################
# 5. Copy data to local node
#############################
echo "Extracting Qwen checkpoint tarball to local scratch..."
tar -xvf /home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct.tar -C $LOCAL_ROOT

echo "Extracting SFT adapter tarball to local scratch..."
tar -xvf ${SFT_ADAPTER}.tar -C $LOCAL_ROOT
mv $LOCAL_ROOT/cluster_${CLUSTER_ID} $LOCAL_SFT_ADAPTER

echo "Copying HH-RLHF dataset to local scratch..."
mkdir -p $LOCAL_ROOT/hh_rlhf_data
cp /home/evan1/scratch/hh_rlhf_data/*.jsonl $LOCAL_ROOT/hh_rlhf_data/

#############################
# 6. Run DPO training
#############################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/RLHF

python -u run_dpo.py \
  --base-model $LOCAL_MODEL \
  --sft-adapter $LOCAL_SFT_ADAPTER \
  --output-dir $LOCAL_OUTPUT \
  --num-epochs 1 \
  --per-device-batch 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-6

#############################
# 7. Copy results back to persistent scratch
#############################
RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_dpo/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR
cp -r $LOCAL_OUTPUT/* $RESULT_DIR/

echo "FINISHED DPO training for cluster ${CLUSTER_ID}"
