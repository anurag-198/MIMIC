# Evaluation Guide

This repository uses `lmms_eval` to evaluate multimodal models on custom tasks such as `counting`, `common`, `listing`, `odd_one`, and `counting_unbalanced`.


## 1. Environment setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate lmm_eval
```

Set the project paths:

```bash
export PROJECT_ROOT=/path/to/lmm_eval
export LLAVA_ROOT=$PROJECT_ROOT/LLaVA-NeXT
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONPATH="$LLAVA_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"
```

If your setup keeps the LLaVA code somewhere else, set `LLAVA_ROOT` accordingly.

## 2. Prepare datasets

Update the corresponding `dataset_path` field in these files:

```text
lmms_eval/tasks/counting/counting.yaml
lmms_eval/tasks/common/common.yaml
lmms_eval/tasks/listing/listing.yaml
lmms_eval/tasks/odd_one/odd_one.yaml
lmms_eval/tasks/counting_unbalanced/counting.yaml
```

## 3. Running with SLURM

Two example scripts are included:

```text
eval_base.sh
eval_ours.sh
```

Before using them, update:

```text
PROJECT_ROOT
LLAVA_ROOT
LORA_PATH
CONDA_ENV_NAME
```

Then submit with:

```bash
sbatch eval_base.sh
sbatch eval_ours.sh
```


This repository is extended from `lmms-eval`: https://github.com/evolvinglmms-lab/lmms-eval

For more details about the original framework, supported models, and additional evaluation options, please refer to the upstream repository.