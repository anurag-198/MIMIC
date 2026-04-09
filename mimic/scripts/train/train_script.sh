#!/bin/bash

#SBATCH -p gpu17
#SBATCH -N 8
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:55:00
#SBATCH -o /your/output/log/dir/run-%j.out
#SBATCH --mem=256G
##SBATCH --cpus-per-task=8
#SBATCH -a 1-6%1

eval "$(conda shell.bash hook)"
conda activate /your/conda/env

cd /your/project/root/

export NCCL_DEBUG=INFO
export WANDB_MODE=offline
export NCCL_SOCKET_FAMILY=AF_INET

HF_CACHE_DIR=/your/huggingface/cache
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASET_CACHE="$HF_CACHE_DIR"
export HF_MODULES_CACHE="$HF_CACHE_DIR"
export PYTHONPATH=/your/project/root/:$PYTHONPATH

export WANDB_API_KEY=your_wandb_api_key

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-slurm-qwen2-7b_14_27"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"
DATA_PATH="scripts/train/data.yaml"
IMAGE_FOLDER="/your/data/path"
OUTPUT_DIR="/your/output/checkpoint/dir/$RUN_NAME"

export ACCELERATE_CPU_AFFINITY=1

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=none \
    bash -c "torchrun \
    --nproc_per_node='${SLURM_GPUS_ON_NODE}' \
    --nnodes='${SLURM_NNODES}' \
    --node_rank=\$SLURM_PROCID \
    --master_addr='${MASTER_ADDR}' \
    --master_port='${MASTER_PORT}' \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder '${IMAGE_FOLDER}' \
    --video_folder '' \
    --mm_vision_tower_lr=0.5e-5 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints '(1x1),...,(6x6)' \
    --mm_patch_merge_type spatial_unpad \
    --fp16 True \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 0.02 \
    --save_total_limit 20 \
    --learning_rate 0.25e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --torch_compile_backend 'inductor' \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mask_enable True \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 32 \
    --layers '14,27' \
    --attn_implementation sdpa"
