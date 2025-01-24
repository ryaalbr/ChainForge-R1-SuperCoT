# R1 + Super CoT: ChainForge

![ChainForge](https://github.com/user-attachments/assets/5b6d4341-f632-41d5-9117-a82a9a4b983a)

A multi-stage pipeline that integrates **DeepSeek Reasoner** for chain-of-thought data and **Qwen2.5** language models. Inspired by the [DeepSeek-R1 paper](https://arxiv.org/pdf/2501.12948), it showcases:

1. **Cold-Start SFT** using CoT from DeepSeek
2. **Reasoning-Oriented RL** (GRPO-like) for improved correctness
3. **Rejection Sampling** to gather top responses
4. **Additional SFT** on filtered data
5. **Final RL** for broad scenarios
6. **Optional Distillation** to smaller Qwen2.5 checkpoints

## What is Super Chain of Thought?

Super Chain of Thought (Super CoT) is an enhanced reasoning framework that combines DeepSeek's chain-of-thought capabilities with reinforcement learning. Unlike traditional CoT which simply shows reasoning steps, Super CoT:

1. **Structured Reasoning**: Uses DeepSeek's <think> tags to clearly separate internal reasoning from final answers
2. **Iterative Refinement**: Applies RL to improve reasoning quality over multiple stages
3. **Quality Control**: Implements rejection sampling to filter and keep only the best reasoning paths
4. **Knowledge Distillation**: Transfers learned reasoning patterns to smaller, more efficient models

## Paper Methodology & Our Implementation

This repository provides a Qwen2.5-based implementation inspired by the DeepSeek-R1 paper. We focus on making the core ideas accessible through a single, well-documented Python script. Here's how we implement the key concepts:

### 1. Base Architecture

* Uses **Qwen2.5-7B** as the foundation model
* Implements a simplified version of **GRPO** (Group Relative Policy Optimization)
* Efficient training without separate critic models

### 2. Training Pipeline Stages

#### Stage A: Pure RL Training

Our implementation starts with direct RL training (similar to DeepSeek-R1-Zero's approach):

* Direct RL application to Qwen2.5 base model
* Simple reward system:
  * Accuracy rewards for correct answers
  * Format rewards for proper <think> tag usage
* Monitoring of reasoning emergence

#### Stage B: Full Training Pipeline

We then implement the complete training sequence:

1. **Cold Start Data Collection**:
   * Collection of CoT examples using DeepSeek API
   * Multiple collection approaches:
     * Direct API calls with proper handling
     * Structured response formatting
     * Error handling and fallbacks

2. **Initial Fine-tuning**:
   * SFT on collected CoT data
   * Clean output format with reasoning and summary
   * Focus on maintaining Qwen2.5's capabilities

3. **Reasoning-Oriented RL**:
   * Simplified GRPO implementation
   * Basic language consistency checks
   * Focus on core reasoning tasks

4. **Data Enhancement**:
   * Basic rejection sampling implementation
   * Target of ~100k reasoning samples
   * Additional general task samples
   * Quality filtering

5. **Final Training**:
   * Additional SFT round
   * Final RL phase
   * Basic safety checks

### 3. Implementation Details

1. **Memory Efficiency**:
   * Basic gradient checkpointing
   * Simple batch size management
   * Standard mixed precision training

2. **Training Stability**:
   * Basic reward normalization
   * Conversation history tracking
   * Error handling
   * Progress logging

3. **Current Status**:
   Our implementation is a work in progress, aiming to:
   * Demonstrate the core concepts
   * Provide a starting point for experimentation
   * Enable learning from the paper's methodology

### 4. Distillation Implementation

Our simplified knowledge distillation approach:

* Teacher: Trained Qwen2.5-7B model
* Student: Smaller Qwen2.5 variants
* Training: Basic supervised learning
* Focus: Maintaining reasoning capabilities

> **Implementation Note**: This is an educational implementation focused on making the paper's concepts accessible. It prioritizes clarity and modularity over achieving state-of-the-art performance.

## Single-Script Implementation

This repository provides a complete implementation of the DeepSeek-R1 paper in a single Python script, making it easy to understand and modify. Key features:

1. **All-in-One Design**:
   * Complete pipeline in `deepseek_qwen2_5_integration_r1.py`
   * No complex dependencies or distributed setup required
   * Easy to modify and experiment with

2. **Hardware Requirements**:
   * Minimum: Single GPU with 24GB VRAM (e.g., RTX 3090)
   * Recommended: 40GB+ VRAM (e.g., A40, A100)
   * CPU: 32+ cores recommended
   * RAM: 64GB+ recommended

3. **Training Time Estimates**:
   * Cold-Start SFT: ~2-4 hours
   * Initial RL: ~8-12 hours
   * Rejection Sampling: ~2-3 hours
   * Additional SFT: ~4-6 hours
   * Final RL: ~12-24 hours
   * Optional Distillation: ~6-8 hours per model size

4. **Memory Optimization**:
   * Gradient checkpointing enabled by default
   * Automatic mixed precision (AMP) training
   * Efficient attention implementation
   * Dynamic batch sizing based on available VRAM

5. **Customization Points**:
   * Reward functions in `compute_reward()`
   * Model architectures in policy classes
   * Training hyperparameters in each stage
   * Data collection and filtering strategies

> **Resource Note**: For users with limited GPU resources, the script includes flags to run smaller experiments or skip certain stages. The minimal version can run on a 16GB GPU but with reduced performance.

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

1. Retrieve high-quality **chain-of-thought** (CoT) from DeepSeek Reasoner's `reasoning_content`
2. Use it for a "cold-start" **supervised fine-tuning** (SFT)
3. Conduct **Reasoning-Oriented RL** to boost correctness and clarity
4. Utilize **rejection sampling** to pick the best RL outputs
5. Perform **additional SFT** on these curated samples
6. Optionally **distill** the final large model into a smaller Qwen2.5 checkpoint

> **Note**: This is a reference pipeline. For production usage, expand datasets, scale RL steps, and incorporate advanced reward modeling.

***

## Features

* **DeepSeek Reasoner Integration**:
  * Automates CoT collection via `reasoning_content`
  * Properly handles <think> tags in chain-of-thought
  * Maintains clean conversation history without reasoning feedback
* **Qwen2.5-7B** Base Model: Hugging Face model with RoPE and large context support
* **Group-based RL**: A GRPO-like approach for stable reinforcement training
* **Rejection Sampling**: Extracts best RL completions for further SFT
* **Distillation**: Compress final RL knowledge into smaller Qwen2.5 variants

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
   * Automated CoT collection using `reasoning_content`
   * Proper handling of <think> tags
   * Clean conversation history management
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

#### DeepSeek API Usage

Important notes for using DeepSeek Reasoner:

1. **Handling `reasoning_content`**:
   ```python
   # Extract both reasoning and final answer
   reasoning_cot = choice.reasoning_content  # Contains <think> tags
   final_text = choice.content  # Final answer only

   # Never feed reasoning_content back into conversation
   messages.append({"role": "assistant", "content": final_text})
   ```

2. **Supported Parameters**:
   ```python
   # Only use these parameters
   response = openai.ChatCompletion.create(
       model="deepseek-reasoner",
       messages=messages,
       max_tokens=1024
   )
   ```

3. **Conversation History**:
   * Only append final answers (`content`)
   * Never include `reasoning_content` in history
   * Keep track of turns properly

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

1. **Response Format**:
   ```
   Question: {prompt}
   <reasoning_process>
     <think>Step-by-step logical deduction</think>
   </reasoning_process>
   <summary>
     Final concise answer
   </summary>
   ```

2. **API Integration**:
   * Proper handling of `reasoning_content` with <think> tags
   * Clean conversation history management
   * Only supported parameters used
   * No reasoning feedback in subsequent calls

3. **Error Handling**:
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
