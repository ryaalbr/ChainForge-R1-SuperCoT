# R1 + Super CoT: ChainForge

A multi-stage pipeline that integrates **DeepSeek Reasoner** for chain-of-thought data and **Qwen2.5** language models. Inspired by the [DeepSeek-R1 paper]([text]\(https://arxiv.org/pdf/2501.12948\)), it showcases:

1. **Cold-Start SFT** using CoT from DeepSeek
2. **Reasoning-Oriented RL** (GRPO-like) for improved correctness
3. **Rejection Sampling** to gather top responses
4. **Additional SFT** on filtered data
5. **Final RL** for broad scenarios
6. **Optional Distillation** to smaller Qwen2.5 checkpoints

***

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Pipeline Stages](#pipeline-stages)
  * [Stage 0: DeepSeek CoT Collection](#stage-0-deepseek-cot-collection)
  * [Stage 1: Cold-Start SFT](#stage-1-cold-start-sft)
  * [Stage 2: Reasoning-Oriented RL](#stage-2-reasoning-oriented-rl)
  * [Stage 3: Rejection Sampling & Additional SFT](#stage-3-rejection-sampling--additional-sft)
  * [Stage 4: Final RL](#stage-4-final-rl)
  * [Stage 5: Optional Distillation](#stage-5-optional-distillation)
* [Key Code Snippets](#key-code-snippets)
* [Advanced Topics](#advanced-topics)
* [Citing & Acknowledgments](#citing--acknowledgments)
* [License](#license)

***

## Overview

**R1 + Super CoT: ChainForge** follows the methodology of **DeepSeek-R1** to enhance a Qwen2.5 model's reasoning abilities via reinforcement learning (RL). We:

1. Retrieve high-quality **chain-of-thought** (CoT) from DeepSeek Reasoner
2. Use it for a "cold-start" **supervised fine-tuning** (SFT)
3. Conduct **Reasoning-Oriented RL** to boost correctness and clarity
4. Utilize **rejection sampling** to pick the best RL outputs
5. Perform **additional SFT** on these curated samples
6. Optionally **distill** the final large model into a smaller Qwen2.5 checkpoint

> **Note**: This is a reference pipeline. For production usage, expand datasets, scale RL steps, and incorporate advanced reward modeling.

***

## Features

* **DeepSeek Reasoner Integration**: Automates CoT collection via `reasoning_content`.
* **Qwen2.5-7B** Base Model: Hugging Face model with RoPE and large context support.
* **Group-based RL**: A GRPO-like approach for stable reinforcement training.
* **Rejection Sampling**: Extracts best RL completions for further SFT.
* **Distillation**: Compress final RL knowledge into smaller Qwen2.5 variants.

***

## Requirements

1. **Python 3.8+**

2. **GPU** (recommended for RL)

3. **Dependencies** (install from `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

4. **DeepSeek API Key**:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   ```
   Or modify `gather_cot_data_from_deepseek()` to include your key directly.

***

## Project Structure

```
.
├── deepseek_qwen2_5_integration_r1.py  # Main pipeline implementation
├── requirements.txt                     # Python dependencies
└── README.md                           # Documentation
```

### Key Components

1. **DeepSeek Integration** (`gather_cot_data_from_deepseek`):
   * Automated CoT collection
   * Structured reasoning format
   * Error handling and fallbacks

2. **Dataset Classes**:
   * `ChainOfThoughtDataset`: For initial SFT
   * `MockRLReasoningDataset`: For RL training
   * `AdditionalSFTDataset`: For post-RL fine-tuning

3. **RL Components**:
   * `GRPOTorchPolicy`: Policy wrapper
   * `compute_reward`: Reward function
   * `sample_responses`: Response generation

***

## Usage

### Basic Usage

1. **Setup Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   ```

3. **Run Pipeline**:
   ```bash
   python deepseek_qwen2_5_integration_r1.py
   ```

### Advanced Usage

#### Custom Prompts

Modify `deepseek_prompts` in `main()`:

```python
deepseek_prompts = [
    "Explain quantum entanglement",
    "Solve the traveling salesman problem",
    "Derive the quadratic formula"
]
```

#### Hyperparameter Tuning

Key parameters to adjust:

```python
# SFT parameters
supervised_fine_tune(
    epochs=5,          # More epochs for better convergence
    batch_size=4,      # Increase for faster training
    lr=5e-6,          # Lower learning rate for stability
    warmup_ratio=0.1   # Longer warmup for complex tasks
)

# RL parameters
rl_training_grpo(
    num_rl_steps=100,  # More steps for better policy
    group_size=8,      # Larger groups for stable updates
    lr=1e-6,          # Conservative learning rate
    clip_ratio=0.15    # Tighter clipping for safety
)
```

***

## Pipeline Stages

### Stage 0: DeepSeek CoT Collection

The pipeline begins by gathering high-quality chain-of-thought data from DeepSeek Reasoner:

1. **Format**: Each response contains:
   ```
   Question: {prompt}
   <reasoning_process>
     Step-by-step logical deduction
   </reasoning_process>
   <summary>
     Final concise answer
   </summary>
   ```

2. **Error Handling**:
   * API failures trigger fallback to mock data
   * Rate limiting protection
   * Response validation

### Stage 1: Cold-Start SFT

Initial supervised fine-tuning on CoT data:

1. **Data Processing**:
   * Tokenization with proper padding
   * Sequence length management
   * Batch collation

2. **Training Loop**:
   * Linear learning rate warmup
   * Gradient clipping
   * Progress tracking

### Stage 2: Reasoning-Oriented RL

Group-based Reward Policy Optimization (GRPO):

1. **Policy Architecture**:
   * Language model as base policy
   * Token-level probability computation
   * Group advantage estimation

2. **Reward Structure**:
   * +1.0 for correct answers
   * +0.2 for proper reasoning format
   * Normalized advantages within groups

### Stage 3: Rejection Sampling & Additional SFT

Quality-focused data augmentation:

1. **Sampling Strategy**:
   * Multiple candidates per question
   * Temperature-controlled generation
   * Reward-based filtering

2. **Additional Training**:
   * Fine-tuning on best samples
   * Shorter training cycle
   * Preservation of reasoning structure

### Stage 4: Final RL

Comprehensive reinforcement learning:

1. **Policy Updates**:
   * KL-constrained optimization
   * Reference model comparison
   * Stable policy improvement

2. **Monitoring**:
   * Reward tracking
   * Loss curves
   * Policy divergence checks

### Stage 5: Optional Distillation

Knowledge transfer to smaller models:

1. **Student Selection**:
   * Smaller Qwen2.5 variants
   * Architecture preservation
   * Memory optimization

2. **Training Process**:
   * Teacher prediction generation
   * Student mimicry learning
   * Checkpoint management

***

## Advanced Topics

### Scaling Up

1. **Distributed Training**:
   ```python
   # Add to model configuration
   device_map = "auto"  # or specific device mapping
   ```

2. **Dataset Expansion**:
   * Collect more DeepSeek CoT samples
   * Implement custom reward models
   * Add task-specific datasets

### Memory Management

1. **Gradient Checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Mixed Precision**:
   ```python
   from torch.cuda.amp import autocast

   with autocast():
       outputs = model(input_ids)
   ```

### Custom Rewards

Implement domain-specific rewards:

```python
def compute_domain_reward(response, ground_truth):
    reward = base_reward(response, ground_truth)
    reward += domain_specific_score(response)
    return reward
```

***

## Citing & Acknowledgments

If you use this code, please cite:

```bibtex
@misc{deepseek2024r1,
  title={DeepSeek-R1: Augmenting Reasoning via Reinforcement Learning},
  author={DeepSeek Team},
  year={2024},
  publisher={arXiv}
}
```

### Contributors

* Nicolas W Schlaepfer (Initial Implementation)
* DeepSeek Team (Original R1 Methodology)
* Qwen Team (Base Models)

## License

MIT License. See [LICENSE](LICENSE) for details.
