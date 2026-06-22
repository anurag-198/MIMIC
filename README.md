# MIMIC Workspace

This workspace contains two related projects:

- `mimic/`: training code for our method
- `lmm_eval/`: evaluation code for benchmarking models and checkpoints

## Repository Layout

### `mimic`

Use this project when you want to train or fine-tune the method.

Key pieces:

- `mimic/scripts/train/train_script.sh`: main SLURM training launcher
- `mimic/scripts/train/data.yaml`: dataset mixture definition
- `mimic/scripts/zero*.json`: DeepSpeed configurations
- `mimic/llava/`: model and training code

Additional setup and launch details are documented in [mimic/README.md](https://github.com/anurag-198/MIMIC_v2/blob/main/mimic/README.md).

### `lmm_eval`

Use this project when you want to evaluate a base model or a trained checkpoint.

Key pieces:

- `lmm_eval/eval_base.sh`: example SLURM script for baseline evaluation
- `lmm_eval/eval_ours.sh`: example SLURM script for evaluating our method
- `lmm_eval/lmms_eval/tasks/`: task definitions and dataset configs
- `lmm_eval/LLaVA-NeXT/`: local LLaVA code used by the evaluation setup

Additional setup and run instructions are documented in [lmm_eval/README.md](https://github.com/anurag-198/MIMIC_v2/blob/main/lmm_eval/README.md).

## Evalaution Dataset

We have released evaluation dataset at huggingface: [anurag4446/mimic](https://huggingface.co/datasets/anurag4446/mimic)


## Typical Workflow

1. Train the model in `mimic/`.
2. Save the resulting checkpoint or LoRA adapter.
3. Evaluate the result in `lmm_eval/`.

## Notes

- The two directories are separate projects with their own environments and configs.
- `mimic/` is derived from LLaVA-NeXT.
- `lmm_eval/` is extended from `lmms-eval` and includes a local LLaVA-NeXT checkout for model support.
