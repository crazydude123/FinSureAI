# FinSure AI

Fine-Tuning Qwen3-4B-Instruct on Financial QA Data with DSPy Inference

## Overview

FinSure AI is a project for fine-tuning the Qwen3-4B-Instruct model on financial question-answering data and performing inference using DSPy. It's designed to be hardware-agnostic as long as a GPU cluster is connected (e.g., H100, A100, etc.).

## Features

- **Modal Compatible**: Run on Modal's cloud GPUs for easy scaling.
- **Kaggle Compatible**: Jupyter notebook version for Kaggle kernels.
- **Hardware Agnostic**: Works on any connected GPU cluster.
- **Easy Setup**: Simple commands to get started.

## Quick Start

### Prerequisites

- Python 3.8+
- GPU with CUDA support (for local runs) or Apple Silicon Mac (M1/M2/M3) for MPS acceleration
- Modal account (for cloud runs)
- Kaggle account (for notebook runs)

### Installation

1. Clone or download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Locally

1. **On Mac (Apple Silicon)**: Ensure PyTorch with MPS support is installed. The code uses `device_map="auto"` which will detect and use MPS.

   Fine-tune the model (may be slower due to shared memory):

```bash
python finetune.py
```

2. **On Linux/Windows with CUDA GPU**: Standard setup.

3. Run inference (works on both):

```bash
python inference.py
```

### ART (Agent Reinforcement Trainer) - RL Fine-Tuning

This repository includes **ART (Agent Reinforcement Trainer)** from OpenPipe for a second-phase RL fine-tuning step after SFT. ART uses GRPO (Group Relative Policy Optimization) to further improve the model's performance through reinforcement learning.

#### What is ART?

ART adds a reinforcement learning stage on top of the existing SFT (Supervised Fine-Tuning) pipeline. After the initial SFT training, ART uses a reward function to train the model to generate better responses. The reward function evaluates:

- Semantic similarity to ground truth answers
- Key token coverage
- Answer length appropriateness
- Detection of repetition and hallucinations

#### Running SFT + GRPO Training

The training script (`finetune.py`) automatically runs both phases:

1. **Phase 1: SFT** - Supervised fine-tuning with LoRA
2. **Phase 2: GRPO** - Reinforcement learning with ART

Simply run:

```bash
python finetune.py
```

This will:
- Train the model using SFTTrainer
- Save the merged SFT model to `./merged_finetuned_qwen`
- Continue with GRPO training using GRPOTrainer
- Save the ART model to `./models/qwen-4b-art`

#### Model Outputs

- **SFT Model**: `./merged_finetuned_qwen` (after Phase 1)
- **ART Model**: `./models/qwen-4b-art` (after Phase 2)

#### Running Inference with ART Model

To use the ART fine-tuned model for inference, use the `--use_art` flag:

```bash
python inference.py --use_art
```

To use the SFT model (default):

```bash
python inference.py
```

You can also provide custom context and question:

```bash
python inference.py --use_art --context "Your context here" --question "Your question here"
```

#### Reward Function

The reward function (`reward.py`) uses:
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic similarity scoring
- Heuristic checks for answer quality (length, repetition, hallucinations)
- Key token matching bonuses

You can customize the reward function in `reward.py` to better match your specific use case.

### Running on Modal

1. Install Modal CLI:

```bash
pip install modal
modal setup
```

2. Fine-tune on Modal:

```bash
modal run modal_app.py::finetune_model
```

3. Run inference on Modal:

```bash
modal run modal_app.py::run_inference --context "Your context here" --question "Your question here"
```

### Running on Kaggle

1. Upload `finsure_ai_notebook.ipynb` to a Kaggle notebook.
2. Set accelerator to GPU.
3. Run the cells in order.

## Configuration

Edit `config.yaml` to customize:

- Model settings
- Dataset parameters
- Training hyperparameters
- Modal GPU type

## Dataset

Uses `virattt/financial-qa-10K` from Hugging Face, containing 10K financial QA pairs from NVIDIA's 10-K filings.

## Hardware Agnosticity

The setup automatically detects and utilizes available GPUs. For Modal, specify the GPU type in `config.yaml` or directly in the function decorator (e.g., `gpu.H100()` for H100 GPUs).

## Troubleshooting

- If VRAM is insufficient, reduce batch size or use gradient checkpointing.
- For full dataset, set `subsample: null` in `config.yaml`.
- Ensure CUDA is installed for local GPU usage.

## License

[Add license if applicable]