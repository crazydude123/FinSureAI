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