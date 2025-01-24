# R1 + Super CoT: ChainForge

This project implements a multi-stage DeepSeek-R1–style pipeline that integrates DeepSeek Reasoner (for chain-of-thought data) with the Qwen2.5 family of large language models. It uses:
•	Supervised Fine-Tuning (SFT) on chain-of-thought data,
•	Reinforcement Learning (RL) with a group-based advantage approach,
•	Rejection sampling to collect high-quality responses for a second round of SFT,
•	A final RL pass for broad scenario coverage,
•	Optional knowledge distillation to compress the reasoning capabilities into a smaller model.

Table of Contents
1\.	Overview
2\.	Features
3\.	Prerequisites
4\.	Project Structure
5\.	Quickstart
6\.	Pipeline Stages
1\.	Chain-of-Thought Data Collection (DeepSeek)
2\.	Cold-Start SFT
3\.	Reasoning-Oriented RL
4\.	Rejection Sampling & Additional SFT
5\.	Final RL Stage
6\.	Optional Distillation
7\.	Hyperparameter & File Configuration
8\.	Customization & Advanced Topics
9\.	Performance Tips & Scaling
10\.	Troubleshooting & Common Issues
11\.	License
12\.	Acknowledgments

Overview

The DeepSeek-R1 paper introduced a powerful multi-stage pipeline to enhance reasoning capabilities in large language models (LLMs) through reinforcement learning. This repository offers a reference implementation of a similar pipeline, combining:
•	DeepSeek Reasoner for generating chain-of-thought (CoT) data,
•	Qwen2.5 LLMs for training and refining reasoning performance via SFT and RL,
•	Knowledge Distillation to compress the resulting reasoning model into a smaller checkpoint.

It closely follows the major steps described in the DeepSeek-R1 approach, while using Qwen2.5-7B-Instruct as an example base model.

Features
•	Chain-of-thought data collection from DeepSeek Reasoner, enabling high-quality CoT for cold-start training.
•	Reinforcement Learning (GRPO) with a reward function that measures correctness and format compliance.
•	Rejection sampling to harvest top-quality responses, then refine the model further via SFT.
•	Final RL pass to align the model with broader tasks and user preferences.
•	Optional Distillation to produce a smaller model that retains advanced reasoning patterns discovered during RL.

Prerequisites

1. Python 3.8+

While Python 3.7 might work, we recommend 3.8 or above for best compatibility with modern libraries.

2. CUDA-capable GPU

At least one NVIDIA GPU is highly recommended for training or RL. Multi-GPU or distributed setups are ideal for scaling.

3. Dependencies

Install via:

pip install -r requirements.txt

Within requirements.txt, make sure you have:
•	torch>=2.0
•	transformers>=4.37.0
•	accelerate
•	sentencepiece
•	openai (for DeepSeek Reasoner)
•	(optionally) anthropic, deepspeed, etc., if you plan further expansions.

4. DeepSeek API Key

Obtain an API key from DeepSeek.
Export it as an environment variable:

export DEEPSEEK\_API\_KEY="your-key-here"

Alternatively, modify the code to set openai.api\_key directly.

Project Structure

.
├── deepseek\_qwen2\_5\_integration\_r1.py   # Main pipeline script
├── requirements.txt                     # Python dependencies
└── README.md                            # This README

```
•	deepseek_qwen2_5_integration_r1.py
```

Contains all the logic for:
1\.	Gathering chain-of-thought data from DeepSeek Reasoner
2\.	Cold-start SFT on that data
3\.	Reinforcement Learning with a toy “Group Relative Policy Optimization” approach
4\.	Rejection sampling to refine SFT data
5\.	A final RL pass
6\.	Optional distillation to a smaller model

Quickstart
1\.	Install dependencies

pip install -r requirements.txt

```
2.	Set the DeepSeek API key
```

export DEEPSEEK\_API\_KEY="your-key-here"

```
3.	Run the pipeline
```

python deepseek\_qwen2\_5\_integration\_r1.py

```
4.	Checkpoints
•	After each major stage, new checkpoints get saved to local directories (qwen_sft_ckpt/, qwen_rl_ckpt_stage2/, etc.).
```

Pipeline Stages

1. DeepSeek CoT Data Collection
   •	What: Calls the DeepSeek Reasoner API to gather chain-of-thought plus short final answers for a set of user prompts.
   •	Why: This provides the “cold start” data (chain-of-thought labeled examples) for initial SFT, enabling the model to generate more interpretable reasoning.
   •	Implementation:

def gather\_cot\_data\_from\_deepseek(prompts, ...):
\# uses openai.ChatCompletion with model="deepseek-reasoner"
\# extracts reasoning\_content and content
\# merges them into a single training sample
return samples

2. Cold-Start SFT
   •	What: Perform supervised fine-tuning on the chain-of-thought data from Step 1.
   •	How: We treat each “prompt + \<reasoning\_process>… + ” as a single sequence. Then do standard next-token prediction with cross-entropy.
   •	Checkpoint: Saved to qwen\_sft\_ckpt/.

3. Reasoning-Oriented RL
   •	What: Use a GRPO-like approach (Group-based Reward Policy Optimization) to further refine the model specifically for correctness and format.
   •	Reward:
   •	+1 if the final text contains the correct answer string (for toy tasks).
   •	+0.2 if the text includes \<reasoning\_process> and <summary> tags.
   •	Implementation:
   1. Sample multiple responses for each prompt in a “group.”
   2. Evaluate reward for each response.
   3. Compute advantage via group-based normalization.
   4. Update the model to maximize advantage while applying a KL penalty vs. a reference.
      •	Checkpoint: qwen\_rl\_ckpt\_stage2/.

4. Rejection Sampling & Additional SFT
   •	What:
   1. Use the RL model from Step 3 to generate multiple completions.
   2. Filter for best responses by reward threshold.
   3. Perform a second SFT pass on this curated data for additional improvement.
      •	Checkpoint: qwen\_sft\_ckpt\_stage3/.

5. Final RL Stage
   •	What: Another RL pass that includes a broader set of prompts or tasks—beyond the strictly numerical reasoning tasks (e.g., general questions, knowledge, writing tasks, etc.).
   •	Why: This broadens alignment and ensures the model can handle different user queries gracefully.
   •	Checkpoint: qwen\_rl\_ckpt\_final/.

6. Optional Distillation
   •	What: Distill the final RL model into a smaller Qwen2.5 checkpoint (e.g., 3B or 1B).
   •	Why: Gains in reasoning from RL can be compressed into a smaller model for efficiency.
   •	Checkpoint: qwen\_distilled\_student/.

Hyperparameter & File Configuration

Check out the top-level constants or function parameters inside deepseek\_qwen2\_5\_integration\_r1.py:
•	num\_rl\_steps: total RL optimization steps (the larger, the better—real usage might need thousands).
•	group\_size: how many samples we generate for each prompt to estimate advantage.
•	clip\_ratio, kl\_coeff: PPO/GRPO hyperparameters controlling policy updates.
•	SFT training arguments (batch size, learning rate, epochs, max\_steps).

You can also adjust:
•	deepseek\_prompts: the set of prompts we feed to DeepSeek Reasoner to gather chain-of-thought data.
•	MockRLReasoningDataset: a placeholder for your real RL tasks (math, code, or knowledge).

Customization & Advanced Topics
1\.	Reward Functions
•	The toy example uses a naive reward (presence of ground-truth string).
•	In practice, you might compile code, parse numeric results, or call specialized validators.
2\.	Data Scaling
•	Replace the small, toy prompts with large volumes of domain-specific data (math, coding, knowledge).
•	Use chunked reading if your data is extremely large.
3\.	Distributed/Accelerated Training
•	For serious usage, integrate Deepspeed or Accelerate for multi-GPU, multi-node setups.
•	Consider parameter-efficient fine-tuning methods (LoRA, QLoRA) to reduce memory overhead.
4\.	Multi-Turn Conversation
•	If your tasks require multiple rounds of user messages, adapt the prompt format and memory of past messages.
•	But be sure to remove the chain-of-thought from the conversation context before calling DeepSeek again, to avoid 400 errors.
5\.	Evaluation
•	Integrate standard evaluation scripts (MMLU, CodeBench, etc.) at each stage to measure improvements.

Performance Tips & Scaling
•	GPU Requirements: Qwen2.5-7B requires a significant amount of GPU VRAM (≥ 15GB). For large-scale RL, more GPUs or distributed training is recommended.
•	Batch Size: Start small (batch size 2 or 4) if limited by GPU memory. Expand if you can.
•	Precision: Use FP16 or BF16 to reduce memory usage. If using accelerate or deepspeed, you can explore 8-bit or 4-bit quantization approaches.
•	Checkpoints: The pipeline saves multiple checkpoints (SFT, RL, final). Each can be quite large. Make sure you have enough storage and handle checkpoint versioning carefully.

Troubleshooting & Common Issues
1\.	CUDA Out of Memory
•	Lower your batch size or reduce sequence length.
•	Use parameter-efficient fine-tuning or 8-bit modes.
2\.	400 Error from DeepSeek
•	Ensure you remove the reasoning\_content from prior calls when building new messages.
3\.	No Improvement in RL
•	Check your reward function. If it’s too lenient or too strict, training might not converge.
•	Verify that the KL penalty or clip ratio is not too high or too low.
4\.	Slow Inference
•	Large models are slow by default; consider smaller Qwen2.5 variants or distillation.
5\.	KeyError / TrustRemoteCode
•	Make sure transformers>=4.37.0 and trust\_remote\_code=True are set for Qwen.

License

All code in this repository is released under the MIT License (unless otherwise noted).
See LICENSE for details.

Acknowledgments
•	DeepSeek: for providing the chain-of-thought Reasoner API and the original DeepSeek-R1 methodology.
•	Qwen Team: for developing and open-sourcing Qwen2.5 models, enabling advanced training workflows.
•	Open-source Community: libraries like Transformers, Accelerate, etc., which make large-scale LLM training feasible.

If you find this work helpful, please consider citing relevant references:

@misc{qwen2.5,
title = {Qwen2.5: A Party of Foundation Models},
url = {https://qwenlm.github.io/blog/qwen2.5/},
author = {Qwen Team},
month = {September},
year = {2024}
}

@misc{deepseekr1,
title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
author={DeepSeek-AI Team},
year={2024},
howpublished={arXiv preprint arXiv:...}
}

Feel free to open issues or pull requests if you encounter any problems or would like to contribute improvements!

Happy reasoning!
