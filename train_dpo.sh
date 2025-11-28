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


module load python/3.11
module load gcc/12.3 arrow/21.0.0
source ~/dpo_env/bin/activate


export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_models

mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE


LOCAL_ROOT=$SLURM_TMPDIR
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_SFT_ADAPTER=$LOCAL_ROOT/sft_adapter
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_dpo/cluster_${CLUSTER_ID}

# Adapter tarball path from your scratch
SFT_ADAPTER_TAR="/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}.tar"

mkdir -p $LOCAL_ROOT/qwen_dpo
mkdir -p $LOCAL_SFT_ADAPTER



echo "Extracting Qwen checkpoint to local scratch..."
tar -xvf /home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct.tar \
    -C $LOCAL_ROOT

echo "Extracting SFT adapter to local scratch..."
tar -xvf $SFT_ADAPTER_TAR -C $LOCAL_ROOT

# After extract, directory name = cluster_${CLUSTER_ID}
mv $LOCAL_ROOT/cluster_${CLUSTER_ID}/* $LOCAL_SFT_ADAPTER

echo "Copying HH-RLHF dataset..."
mkdir -p $LOCAL_ROOT/hh_rlhf_data
cp /home/evan1/scratch/hh_rlhf_data/*.jsonl $LOCAL_ROOT/hh_rlhf_data/

###############################################
# 6. RUN DPO TRAINING
###############################################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/RLHF

python -u run_dpo.py \
  --base-model $LOCAL_MODEL \
  --sft-adapter $LOCAL_SFT_ADAPTER \
  --output-dir $LOCAL_OUTPUT \
  --data-dir $LOCAL_ROOT/hh_rlhf_data \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 5e-6 \
  --num-epochs 1


RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_dpo/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR

echo "Copying results back to persistent scratch..."
cp -r $LOCAL_OUTPUT/* $RESULT_DIR/

echo "FINISHED DPO TRAINING FOR CLUSTER ${CLUSTER_ID}"

