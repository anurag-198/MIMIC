#!/bin/bash
#SBATCH -p gpu22

#SBATCH -o ./slurm_logs/run-%j.out
#SBATCH -t 1:55:00
#SBATCH --gres gpu:8

PROJECT_ROOT="/path/to/your/project/lmm_eval"
LLAVA_ROOT="${PROJECT_ROOT}/LLaVA-NeXT"
CONDA_ENV_NAME="lmm_eval"

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export PYTHONPATH="${LLAVA_ROOT}:$PYTHONPATH"

eval "$(conda shell.bash hook)"

#export HF_HUB_OFFLINE=1
conda activate "${CONDA_ENV_NAME}";

cd "${PROJECT_ROOT}";

accelerate launch --num_processes=8 -m lmms_eval --model llava_onevision --model_args pretrained='lmms-lab/llava-onevision-qwen2-7b-ov,attn_implementation=flash_attention_2'   --tasks counting_multi --batch_size 1 --log_samples --log_samples_suffix llava_v1.5_mme  --output_path ./logs/llava_base
