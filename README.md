# DeepSeek-R1 Style Pipeline with Qwen2.5

This project implements a sophisticated training pipeline that combines DeepSeek Reasoner's chain-of-thought capabilities with Qwen2.5 language models. The pipeline includes supervised fine-tuning (SFT), reinforcement learning (RL), and optional knowledge distillation.

## Features

* Chain-of-thought data collection from DeepSeek Reasoner
* Cold-start SFT using CoT data
* Reasoning-oriented RL with GRPO (Group-based Reward Policy Optimization)
* Rejection sampling for high-quality responses
* Additional SFT on filtered data
* Final RL stage for all scenarios
* Optional knowledge distillation to smaller models

## Prerequisites

1. Python 3.8+

2. CUDA-capable GPU (recommended)

3. Dependencies (install via pip):
   ```bash
   pip install -r requirements.txt
   ```

4. DeepSeek API Key:
   * Set the environment variable: `export DEEPSEEK_API_KEY="your-key-here"`
   * Or modify the code to include your key directly

## Project Structure

```
.
├── deepseek_qwen2_5_integration_r1.py  # Main pipeline code
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your DeepSeek API key:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   ```

3. Run the pipeline:
   ```bash
   python deepseek_qwen2_5_integration_r1.py
   ```

## Pipeline Stages

1. **DeepSeek CoT Data Collection**
   * Gathers chain-of-thought reasoning from DeepSeek
   * Formats data for training

2. **Cold-Start SFT**
   * Initial fine-tuning on CoT data
   * Checkpoint: `qwen_sft_ckpt/`

3. **Reasoning RL**
   * GRPO training on reasoning tasks
   * Checkpoint: `qwen_rl_ckpt_stage2/`

4. **Rejection Sampling & Additional SFT**
   * Filter high-quality responses
   * Additional fine-tuning
   * Checkpoint: `qwen_sft_ckpt_stage3/`

5. **Final RL Stage**
   * Comprehensive RL training
   * Checkpoint: `qwen_rl_ckpt_final/`

6. **Optional Distillation**
   * Knowledge transfer to smaller model
   * Checkpoint: `qwen_distilled_student/`

## Customization

* Modify `deepseek_prompts` in `main()` for your use case
* Adjust hyperparameters in each training function
* Replace `MockRLReasoningDataset` with your actual dataset
* Choose different Qwen2.5 variants for base/student models

## Notes

* This is a reference implementation
* For production use:
  * Scale up the dataset size
  * Increase training steps/epochs
  * Use distributed training
  * Implement proper logging
  * Add model evaluation
  * Consider using smaller Qwen2.5 variants

## License

MIT

## Acknowledgments

* DeepSeek for the Reasoner API
* Qwen team for the Qwen2.5 models
