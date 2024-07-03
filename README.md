# 🚀 Training Pipeline

## 📋 Table of Contents
- [Normal Use](#normal-use)
  - [Installation](#installation)
  - [Running the Code](#running-the-code)
- [DeepSpeed Integration](#deepspeed-integration)
  - [Installation](#installation-1)
  - [Accelerate Configuration](#accelerate-configuration)
- [🛠️ Configuration Details](#configuration-details)

## Normal Use

### Installation

To get started with the standard setup, install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Code

1. Navigate to the desired task file (e.g., `./run/sft.sh`, `./run/cpt.sh`).
2. Modify the parameters as needed.
3. Execute the script:

```bash
./run/{task}.sh
```

## DeepSpeed Integration

For enhanced performance and distributed training capabilities, follow these steps to set up DeepSpeed.

### Installation

#### 1. Set up Conda Environment

```bash
conda create -n nlpenv -y
conda init
# Restart your terminal
conda activate nlpenv
```

#### 2. Install DeepSpeed

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed/
DS_BUILD_CPU_ADAM=1 ./install.sh -r
```

#### 3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

### Accelerate Configuration

Run the following command to configure Accelerate:

```bash
accelerate config --config_file configs/config.yaml
```

Follow the prompts with these recommended settings:

- Compute Environment: `This machine`
- Machine Type: `No distributed training`
- CPU-only Training: `NO`
- Torch Dynamo Optimization: `NO`
- DeepSpeed Usage: `yes`
- DeepSpeed Config JSON: `NO`
- ZeRO Optimization Stage: `3`
- Optimizer State Offload: `cpu`
- Parameter Offload: `cpu`
- Gradient Accumulation Steps: `128`
- Gradient Clipping: `yes`
- Gradient Clipping Value: `1.0`
- 16-bit Model Weights for ZeRO Stage-3: `yes`
- deepspeed.zero.Init for ZeRO Stage-3: `yes`
- Mixture-of-Experts Training: `NO`
- Number of GPUs: `1`
- Mixed Precision: `bf16`

## 🛠️ Configuration Details

The configuration above is optimized for:
- DeepSpeed ZeRO Stage 3 for efficient memory usage
- CPU offloading for optimizer states and parameters
- Gradient accumulation for larger effective batch sizes
- Gradient clipping to prevent exploding gradients
- BF16 mixed precision for improved performance on supported hardware
