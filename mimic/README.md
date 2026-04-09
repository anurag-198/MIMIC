# Training Guide

This directory contains the launch scripts and DeepSpeed configs used to train the model.

## Main Files

- `scripts/train/train_script.sh`: Slurm launch script for distributed training.
- `scripts/train/data.yaml`: Dataset mixture definition passed through `--data_path`.
- `scripts/zero2.json`: Default DeepSpeed ZeRO-2 config used by the current launcher.
- `scripts/zero3.json`, `scripts/zero3_offload.json`, `scripts/zero3pp.json`: Alternative ZeRO-3 variants.

## Before You Run

Edit `scripts/train/train_script.sh` and replace the placeholder values below:

- `#SBATCH -o /your/output/log/dir/run-%j.out`
- `conda activate /your/conda/env`
- `cd /your/project/root/`
- `HF_CACHE_DIR=/your/huggingface/cache`
- `export PYTHONPATH=/your/project/root/:$PYTHONPATH`
- `export WANDB_API_KEY=your_wandb_api_key`
- `IMAGE_FOLDER="/your/data/path"`
- `OUTPUT_DIR="/your/output/checkpoint/dir/$RUN_NAME"`


## Dataset Expectations

The training mixture in `scripts/train/data.yaml` is expected to come from two sources:

- our MIMIC instruction tuning data
- additional data taken from `https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data`

The launcher currently uses:

```bash
DATA_PATH="scripts/train/data.yaml"
```

The yaml files use OpenImagesV7 (`https://storage.googleapis.com/openimages/web/index.html`) as training images. 
Please download it before.



## How To Launch

Submit the Slurm job from the repository root:

```bash
sbatch scripts/train/train_script.sh
```

## Quick Checklist

Before launching, verify:

- the conda environment contains the project dependencies
- the Slurm partition and node/GPU counts match your cluster
- `IMAGE_FOLDER` matches the root used by the JSON image paths
- `DATA_PATH` points to the intended dataset YAML
- `OUTPUT_DIR` and the Slurm log directory exist or are writable
- the selected DeepSpeed config matches available CPU/GPU memory


This codebase is an extension of `https://github.com/LLaVA-VL/LLaVA-NeXT`. Readers looking for broader framework details, architecture context, and additional training utilities can refer to that repository for more background.
