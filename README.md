# R1 + Super CoT: ChainForge

DeepSeek-R1 Pipeline with Qwen2.5 & DeepSeek Reasoner

This repository implements a DeepSeek-R1–style multi-stage pipeline for improving the reasoning capabilities of a Qwen2.5 model, leveraging chain-of-thought (CoT) data from DeepSeek Reasoner, plus multi-stage reinforcement learning (RL). The overall approach is inspired by the DeepSeek-R1 paper:

```
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
```

Table of Contents
1\.	Overview
2\.	Project Structure
3\.	Key Features
4\.	Installation & Setup
5\.	Pipeline Stages & Code Snippets
1\.	Stage 0: Gathering CoT from DeepSeek
2\.	Stage 1: Cold-Start SFT
3\.	Stage-2: Reasoning-Oriented RL (GRPO)
4\.	Stage-3: Rejection Sampling & Additional SFT
5\.	Stage-4: Final RL Stage
6\.	Stage-5: (Optional) Distillation
6\.	Connecting to the DeepSeek-R1 Paper
7\.	Hyperparameters & Customization
8\.	Code Snippets
9\.	Future Directions
10\.	License

Overview

The DeepSeek-R1 paper introduces two models:
•	DeepSeek-R1-Zero: A “pure RL” approach (no supervised data at start).
•	DeepSeek-R1: A pipeline that uses “cold-start” CoT data, followed by multi-stage RL and optional distillation.

This repo follows the DeepSeek-R1 recipe (the “cold start” route), applying:
1\.	Chain-of-thought data collection from DeepSeek Reasoner
2\.	SFT on that CoT data
3\.	Reasoning-Oriented RL (using a group-based advantage technique similar to GRPO)
4\.	Rejection sampling to collect only the best RL outputs, then more SFT
5\.	A final RL pass
6\.	(Optionally) distill the final model into a smaller Qwen2.5 checkpoint

Project Structure

.
├── deepseek\_qwen2\_5\_integration\_r1.py   # Main pipeline code
├── requirements.txt                     # Dependencies
└── README.md                            # This README

```
•	deepseek_qwen2_5_integration_r1.py: Contains:
•	A function to gather CoT from DeepSeek Reasoner
•	A ChainOfThoughtDataset class for SFT
•	A small MockRLReasoningDataset for toy RL tasks
•	GRPOTorchPolicy for group-based RL updates
•	Functions for rejection sampling and distillation
•	A main() function orchestrating the entire pipeline
```

Key Features
1\.	Chain-of-Thought (CoT) Integration
•	Calls the DeepSeek Reasoner API, retrieving reasoning\_content and final content fields.
2\.	Multi-Stage RL (GRPO)
•	Uses a simplified PPO-like approach with group advantage.
3\.	Rejection Sampling
•	Filters out suboptimal RL completions for better SFT data.
4\.	Distillation
•	Compresses the final model’s reasoning capability into a smaller checkpoint.
5\.	Faithful to DeepSeek-R1
•	Reflects the 4-step pipeline of cold-start data, RL, rejection sampling, final RL, and optional distillation.

Installation & Setup
1\.	Install Dependencies

pip install -r requirements.txt

Make sure transformers>=4.37.0 for Qwen2.5, plus openai for the DeepSeek Reasoner API, and torch with GPU support.

```
2.	Set DeepSeek API Key
```

export DEEPSEEK\_API\_KEY="your-deepseek-api-key"

Or modify the code to set openai.api\_key directly.

```
3.	Run the Pipeline
```

python deepseek\_qwen2\_5\_integration\_r1.py

Checkpoints are saved to local folders (e.g., qwen\_sft\_ckpt/, qwen\_rl\_ckpt\_stage2/, etc.).

Pipeline Stages & Code Snippets

Stage 0: Gathering CoT from DeepSeek

Goal: Retrieve chain-of-thought for “cold-start” SFT.

Code (gather\_cot\_data\_from\_deepseek):

def gather\_cot\_data\_from\_deepseek(prompts, max\_samples=10, model\_name="deepseek-reasoner"):
"""
Calls DeepSeek Reasoner for each prompt, extracts reasoning\_content + content,
and formats them into a single text block for SFT.
"""
openai.api\_key = ...
openai.api\_base = "https://api.deepseek.com"
...
for user\_prompt in prompts:
response = openai.ChatCompletion.create(...)
reasoning\_cot = response.choices\[0].message.reasoning\_content
final\_answer = response.choices\[0].message.content
\# Merge
single\_text = f"Question: {user\_prompt}\n\<reasoning\_process>{reasoning\_cot}\</reasoning\_process>\n<summary>{final\_answer}</summary>"
results.append(single\_text)
return results

Stage 1: Cold-Start SFT

Goal: Use that newly gathered CoT to fine-tune Qwen2.5 so it learns to produce reasoning + final answer.

Code snippet from ChainOfThoughtDataset + supervised\_fine\_tune:

class ChainOfThoughtDataset(Dataset):
\# Holds the CoT text examples from DeepSeek Reasoner
\# Tokenized in collate\_fn
...

def supervised\_fine\_tune(model, tokenizer, train\_dataset, ...):
"""
Standard teacher-forcing next-token prediction on chain-of-thought data.
Saves to output\_dir.
"""
...
model.save\_pretrained(output\_dir)
tokenizer.save\_pretrained(output\_dir)

Stage 2: Reasoning-Oriented RL (GRPO)

Goal: Further refine the model for correctness, clarity of reasoning, etc.
•	We define a MockRLReasoningDataset with a trivial question-answer format for demonstration.
•	Use a GRPOTorchPolicy wrapper to compute log probabilities.
•	Reward = correctness + CoT formatting.

Code snippet from rl\_training\_grpo:

def rl\_training\_grpo(
policy\_model, tokenizer, rl\_dataset, num\_rl\_steps=50, group\_size=4, ...
):
"""
Conducts group-based RL. For each question, sample multiple responses,
compute reward, then update the policy with a clipped objective + KL penalty.
"""
\# 1) Sample responses
\# 2) Compute advantage from group rewards
\# 3) Update policy step by step
...
return policy\_model.model

Stage 3: Rejection Sampling & Additional SFT

Goal: Gather the best RL completions by reward, convert them into new training examples, and do a second SFT pass.

Code snippet from rejection\_sampling\_data\_gen:

def rejection\_sampling\_data\_gen(rl\_model, tokenizer, dataset, ...):
"""
For each dataset item, sample multiple completions,
pick the highest-reward, and store if above threshold.
"""
new\_data = \[]
for item in dataset:
candidates = sample\_responses(...)
\# Evaluate reward
\# Keep best if reward >= accept\_threshold
return new\_data

We then load that new data into an AdditionalSFTDataset, and call supervised\_fine\_tune again to produce qwen\_sft\_ckpt\_stage3.

Stage 4: Final RL Stage

Goal: Another RL pass for broader coverage of prompts/scenarios.

Code snippet:

# Reload updated model

model\_after\_stage3 = AutoModelForCausalLM.from\_pretrained("qwen\_sft\_ckpt\_stage3", ...)
policy2 = GRPOTorchPolicy(model\_after\_stage3)
final\_rl\_model = rl\_training\_grpo(..., policy\_model=policy2, ...)
final\_rl\_model.save\_pretrained("qwen\_rl\_ckpt\_final")

Stage 5: (Optional) Distillation

Goal: Distill the final RL model into a smaller Qwen2.5.

Code snippet from distill\_reasoning:

def distill\_reasoning(
teacher\_model, tokenizer, base\_student\_ckpt="Qwen/Qwen2.5-7B", dataset\_texts=None, ...
):
"""
1\. Load the smaller student model
2\. Generate teacher outputs on the dataset
3\. SFT the student to replicate them
4\. Save final distilled checkpoint
"""
...

Connecting to the DeepSeek-R1 Paper
1\.	Cold Start Data vs. RL: The pipeline follows DeepSeek-R1’s practice of acquiring a small set of high-quality chain-of-thought data first, rather than starting from scratch (DeepSeek-R1-Zero).
2\.	Multi-Stage RL: Mirroring the two RL phases described (reasoning-oriented RL + RL for all scenarios).
3\.	Rejection Sampling: Exactly aligns with the paper’s approach to gather new SFT data from the model’s best RL outputs.
4\.	Distillation: The final step in the paper, providing smaller “DeepSeek-R1-Distill” models. Here, we similarly compress the final RL model into a smaller Qwen2.5 checkpoint.

Hence, the code is inspired by and consistent with the DeepSeek-R1 pipeline described in the paper—albeit scaled down for demonstration.

Hyperparameters & Customization
•	num\_rl\_steps: More steps yield better RL coverage (the paper used thousands).
•	group\_size: Increase to produce more candidate completions per prompt, better advantage estimation.
•	clip\_ratio + kl\_coeff: Control how strictly we penalize the policy for deviating from the reference.
•	compute\_reward: You can expand it to compile code solutions, parse numeric results carefully, or even use a neural reward model.

Code Snippets

Below are some salient code blocks from deepseek\_qwen2\_5\_integration\_r1.py:
1\.	Gathering CoT:

# Stage 0: Retrieve chain-of-thought from DeepSeek

deepseek\_prompts = \[
"What is 9.11 plus 9.8?",
"Explain how to compute factorial of 5",
...
]
cot\_data = gather\_cot\_data\_from\_deepseek(deepseek\_prompts, max\_samples=3)

```
2.	RL updates:
```

# Partial snippet from rl\_training\_grpo

ratio = torch.exp(pol\_lp - ref\_lp)
surr1 = ratio \* adv
surr2 = torch.clamp(ratio, 1.0 - clip\_ratio, 1.0 + clip\_ratio) \* adv
policy\_loss = -torch.min(surr1, surr2)
kl\_penalty = kl\_coeff \* (pol\_lp - ref\_lp)
total\_loss = policy\_loss + kl\_penalty

```
3.	Distillation:
```

# Distill final RL model into a smaller Qwen2.5

teacher = AutoModelForCausalLM.from\_pretrained("qwen\_rl\_ckpt\_final", ...)
distill\_reasoning(
teacher\_model=teacher,
tokenizer=tokenizer,
base\_student\_ckpt="Qwen/Qwen2.5-3B",  # e.g., smaller
dataset\_texts=distill\_dataset\_texts,
...
)

Future Directions
•	Full Scale Datasets: Replace the toy MockRLReasoningDataset with real math/coding tasks.
•	Distributed Training: For large-scale RL, consider accelerate, Deepspeed, or Colossal-AI.
•	Advanced Reward Modeling: Combine rule-based correctness checks with a preference model for style, helpfulness, etc.
•	Multi-Turn: If you want multi-turn or chat-based tasks, track conversation context carefully and avoid re-feeding chain-of-thought to DeepSeek.

License

This code is released under the MIT License. See LICENSE file for details.

Enjoy exploring DeepSeek-R1–style reinforcement learning with Qwen2.5! If you find this helpful, please cite the DeepSeek-R1 paper and the Qwen2.5 docs. Feel free to open an issue or PR with improvements or suggestions!
