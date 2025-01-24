#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 Style Pipeline with Qwen2.5 Integration
=================================================

This pipeline implements a sophisticated training approach that:
1. Gathers chain-of-thought (CoT) data from DeepSeek Reasoner
2. Uses Qwen2.5-7B-Instruct as the base model for:
   - Supervised Fine-Tuning (SFT)
   - Reasoning-Oriented Reinforcement Learning (RL)
   - Rejection Sampling
   - Additional SFT
   - Final RL stage
   - Optional Knowledge Distillation

Prerequisites:
-------------
* pip install openai  # For DeepSeek API
* pip install transformers>=4.37.0 accelerate sentencepiece  # For Qwen
* Set DEEPSEEK_API_KEY in environment or code

Author: Nicolas W Schlaepfer
License: MIT
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For DeepSeek API calls
import openai

# For Qwen2.5 model and tokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
)


def gather_cot_data_from_deepseek(
    prompts, max_samples=10, model_name="deepseek-reasoner"
):
    """
    Gather chain-of-thought (CoT) data from DeepSeek Reasoner API.

    This function:
    1. Takes a list of prompts/questions
    2. Calls DeepSeek Reasoner API for each
    3. Extracts both reasoning (CoT) and final answer
    4. Formats them into a unified training format

    Args:
        prompts (List[str]): List of questions/prompts to get CoT for
        max_samples (int): Maximum number of API calls to make
        model_name (str): DeepSeek model to use

    Returns:
        List[str]: List of formatted strings containing:
                  "Question: {prompt}
                   <reasoning_process>{cot}</reasoning_process>
                   <summary>{answer}</summary>"
    """
    # Ensure DeepSeek API key is set
    openai.api_key = os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")
    openai.api_base = "https://api.deepseek.com"

    results = []
    messages_history = []  # Track conversation history

    # Limit total API calls for demo/cost purposes
    n = min(len(prompts), max_samples)

    for i in range(n):
        user_prompt = prompts[i]
        # Format messages for DeepSeek API - only include content, not reasoning
        messages = messages_history + [{"role": "user", "content": user_prompt}]

        try:
            # Call DeepSeek API - only use supported parameters
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,  # Controls final answer length
            )

            # Extract CoT (with <think> tags) and final answer
            choice = response.choices[0].message
            reasoning_cot = (
                choice.reasoning_content
            )  # Chain-of-thought with <think> tags
            final_text = choice.content  # Final answer

            # Format into unified training format
            single_text = (
                f"Question: {user_prompt}\n"
                f"<reasoning_process>{reasoning_cot}</reasoning_process>\n"
                f"<summary>{final_text}</summary>"
            )
            results.append(single_text)

            # Update conversation history with ONLY the final answer, not the reasoning
            messages_history.append({"role": "assistant", "content": final_text})

        except Exception as e:
            print(f"DeepSeek API call failed for prompt='{user_prompt}': {e}")
            continue

    return results


class ChainOfThoughtDataset(Dataset):
    """
    PyTorch Dataset for Chain-of-Thought training data.

    This dataset handles:
    1. Storing CoT text samples
    2. Tokenization for model input
    3. Proper padding and truncation
    4. Batch collation

    The expected format for each text is:
    "Question: {question}
     <reasoning_process>{cot}</reasoning_process>
     <summary>{answer}</summary>"
    """

    def __init__(self, texts, tokenizer, max_length=512):
        """
        Initialize the dataset.

        Args:
            texts (List[str]): List of formatted CoT texts
            tokenizer: HuggingFace tokenizer (e.g., Qwen2.5 tokenizer)
            max_length (int): Maximum sequence length for truncation
        """
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single text sample by index."""
        return self.texts[idx]

    def collate_fn(self, batch_texts):
        """
        Collate a batch of texts into model input format.

        This function:
        1. Tokenizes all texts in the batch
        2. Applies padding to make all sequences same length
        3. Truncates sequences that are too long
        4. Returns tensors ready for model input

        Args:
            batch_texts (List[str]): Batch of text samples

        Returns:
            dict: Contains input_ids and attention_mask tensors
        """
        return self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def supervised_fine_tune(
    model,
    tokenizer,
    train_dataset,
    output_dir="qwen_sft_ckpt",
    epochs=1,
    batch_size=2,
    lr=1e-5,
    warmup_ratio=0.06,
    max_steps=None,
    device="cuda",
):
    """
    Perform supervised fine-tuning (SFT) on the model using chain-of-thought data.

    This function:
    1. Sets up training DataLoader
    2. Configures optimizer and learning rate scheduler
    3. Runs training loop with gradient updates
    4. Saves the fine-tuned model

    Args:
        model: HuggingFace model to fine-tune
        tokenizer: Associated tokenizer
        train_dataset: Dataset containing CoT samples
        output_dir (str): Where to save the model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        warmup_ratio (float): Portion of steps for LR warmup
        max_steps (int, optional): If set, override epochs
        device (str): Device to train on ('cuda' or 'cpu')
    """
    # Setup training dataloader
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    # Move model to device and set to training mode
    model = model.to(device)
    model.train()

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Calculate total steps and warmup steps
    total_steps = len(train_loader) * epochs if max_steps is None else max_steps
    warmup_steps = int(total_steps * warmup_ratio)

    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    global_step = 0
    for epoch in range(epochs):
        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass with loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # For causal LM, targets are inputs
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Progress tracking
            global_step += 1
            if global_step % 10 == 0:
                print(f"[SFT] step={global_step}, loss={loss.item():.4f}")

            # Early stopping if max_steps reached
            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break

    # Save the fine-tuned model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SFT] Done. Model saved at {output_dir}")


class MockRLReasoningDataset(Dataset):
    """
    Mock dataset for RL training with simple arithmetic problems.

    This dataset generates pairs of:
    - Questions: "Solve X + X = ?"
    - Ground truth answers: "2X"

    In real applications, replace this with your actual task dataset.
    """

    def __init__(self, tokenizer, num_samples=64, max_length=512):
        """
        Initialize the mock RL dataset.

        Args:
            tokenizer: HuggingFace tokenizer
            num_samples (int): Number of mock samples to generate
            max_length (int): Maximum sequence length
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # Generate mock question-answer pairs
        self.questions = []
        self.answers = []
        for i in range(num_samples):
            question = f"Solve {i} + {i}=?"
            ground_truth = str(2 * i)
            self.questions.append(question)
            self.answers.append(ground_truth)

    def __len__(self):
        """Return the number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get a question-answer pair by index."""
        return {"question": self.questions[idx], "ground_truth": self.answers[idx]}


class GRPOTorchPolicy(nn.Module):
    """
    Group-based Reward Policy Optimization (GRPO) implementation.

    This policy wrapper:
    1. Handles the language model as a policy
    2. Computes log probabilities for chosen tokens
    3. Enables group-based advantage updates

    The approach is similar to PPO but operates on groups of samples
    rather than individual trajectories.
    """

    def __init__(self, model):
        """
        Initialize the GRPO policy.

        Args:
            model: Base language model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)

    def log_probs_of_chosen_tokens(self, input_ids, attention_mask):
        """
        Compute log probabilities of chosen tokens.

        This function:
        1. Gets model logits
        2. Applies softmax to get probabilities
        3. Takes log of probabilities for chosen tokens

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding

        Returns:
            torch.Tensor: Log probabilities of chosen tokens
        """
        # Get model outputs
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [batch, seq_len, vocab]

        # Get last token logits and compute log probabilities
        last_logits = logits[:, -1, :]
        lp = F.log_softmax(last_logits, dim=-1)
        return lp


def compute_reward(response_text, ground_truth):
    """
    Compute reward for a model response.

    This simple reward function:
    1. Gives +1 if ground truth appears in response
    2. Gives +0.2 if response has proper reasoning format

    Args:
        response_text (str): Model's response text
        ground_truth (str): Expected answer

    Returns:
        float: Computed reward value
    """
    reward = 0.0

    # Reward for proper reasoning format
    if "<reasoning_process>" in response_text and "<summary>" in response_text:
        reward += 0.2

    # Reward for correct answer
    if ground_truth in response_text:
        reward += 1.0

    return reward


def sample_responses(
    model,
    tokenizer,
    question,
    device="cuda",
    num_samples=4,
    temperature=0.7,
    max_new_tokens=128,
):
    """
    Sample multiple responses from the model for a question.

    This function:
    1. Formats the question with chat template
    2. Generates multiple responses with temperature
    3. Returns list of response texts

    Args:
        model: Language model to sample from
        tokenizer: Associated tokenizer
        question (str): Input question
        device (str): Device to run on
        num_samples (int): Number of responses to generate
        temperature (float): Sampling temperature
        max_new_tokens (int): Maximum new tokens to generate

    Returns:
        List[str]: Generated response texts
    """
    model.eval()

    # Format with chat template
    system_prompt = "You are Qwen, a helpful assistant.\n"
    user_prompt = f"User: {question}\nAssistant:"
    text = system_prompt + user_prompt

    # Encode prompt
    encoded_prompt = tokenizer.encode(text, return_tensors="pt").to(device)

    # Generate multiple responses
    all_responses = []
    with torch.no_grad():
        for _ in range(num_samples):
            gen_ids = model.generate(
                encoded_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Extract only the newly generated portion
            new_text = tokenizer.decode(
                gen_ids[0][len(encoded_prompt[0]) :], skip_special_tokens=True
            )
            all_responses.append(new_text)

    return all_responses


def rl_training_grpo(
    policy_model,
    tokenizer,
    rl_dataset,
    num_rl_steps=50,
    group_size=4,
    device="cuda",
    lr=1e-6,
    clip_ratio=0.2,
    kl_coeff=0.001,
):
    """
    Train the model using Group-based Reward Policy Optimization (GRPO).

    This function:
    1. Samples groups of responses for each question
    2. Computes rewards and advantages
    3. Updates policy using clipped objective
    4. Applies KL penalty to stay close to reference

    Args:
        policy_model: GRPO policy to train
        tokenizer: Associated tokenizer
        rl_dataset: Dataset with questions and ground truths
        num_rl_steps (int): Total training steps
        group_size (int): Number of responses per question
        device (str): Device to train on
        lr (float): Learning rate
        clip_ratio (float): PPO clip ratio
        kl_coeff (float): KL penalty coefficient

    Returns:
        nn.Module: Trained language model
    """
    # Setup policy for training
    policy_model = policy_model.to(device)
    policy_model.train()

    # Setup optimizer
    optimizer = AdamW(policy_model.parameters(), lr=lr)

    # Load reference model for KL penalty
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto"
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Training loop
    step_count = 0
    data_indices = list(range(len(rl_dataset)))
    random.shuffle(data_indices)

    while step_count < num_rl_steps:
        for idx in data_indices:
            # Get question and ground truth
            sample = rl_dataset[idx]
            question = sample["question"]
            ground_truth = sample["ground_truth"]

            # Sample responses from current policy
            with torch.no_grad():
                responses = sample_responses(
                    policy_model.model,
                    tokenizer,
                    question,
                    device=device,
                    num_samples=group_size,
                )

            # Compute rewards and advantages
            rewards = [compute_reward(r, ground_truth) for r in responses]
            mean_r = sum(rewards) / len(rewards)
            std_r = max(
                1e-6, (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
            )
            advantages = [(r - mean_r) / std_r for r in rewards]

            # Update policy for each response in group
            for g_idx in range(group_size):
                resp_text = responses[g_idx]
                adv = advantages[g_idx]

                # Skip empty responses
                if not resp_text:
                    continue

                # Format full conversation
                new_input = f"User: {question}\nAssistant: {resp_text}"

                # Get token probabilities
                enc = tokenizer.encode(new_input, return_tensors="pt").to(device)
                policy_lp = policy_model.log_probs_of_chosen_tokens(enc, None)

                with torch.no_grad():
                    ref_out = ref_model(enc)
                    ref_logits = ref_out.logits[:, -1, :]
                    ref_lp = F.log_softmax(ref_logits, dim=-1)

                # Get probability ratio
                last_char = resp_text[-1]
                tid = tokenizer.convert_tokens_to_ids(last_char)
                if tid is None:
                    tid = tokenizer.eos_token_id

                pol_lp = policy_lp[0, tid]
                ref_lp = ref_lp[0, tid]
                ratio = torch.exp(pol_lp - ref_lp)

                # Compute losses
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)

                # Add KL penalty
                kl_penalty = kl_coeff * (pol_lp - ref_lp)
                total_loss = policy_loss + kl_penalty

                # Update policy
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Progress tracking
                step_count += 1
                if step_count % 10 == 0:
                    print(
                        f"[RL GRPO] step={step_count}, "
                        f"rew={rewards[g_idx]:.2f}, "
                        f"adv={adv:.2f}, "
                        f"loss={policy_loss.item():.4f}"
                    )

                if step_count >= num_rl_steps:
                    break

            if step_count >= num_rl_steps:
                break

        if step_count >= num_rl_steps:
            break

    return policy_model.model


def rejection_sampling_data_gen(
    rl_model, tokenizer, dataset, device="cuda", num_samples=4, accept_threshold=0.5
):
    """
    Generate additional training data using rejection sampling.

    This function:
    1. Samples multiple responses for each question
    2. Keeps only high-reward responses
    3. Formats them for additional SFT

    Args:
        rl_model: Trained RL model to sample from
        tokenizer: Associated tokenizer
        dataset: Dataset with questions and ground truths
        device (str): Device to run on
        num_samples (int): Responses to sample per question
        accept_threshold (float): Minimum reward to accept

    Returns:
        List[str]: New training examples
    """
    new_data = []

    for i in range(len(dataset)):
        # Get question and ground truth
        item = dataset[i]
        question = item["question"]
        gt = item["ground_truth"]

        # Sample candidate responses
        candidates = sample_responses(
            rl_model, tokenizer, question, device=device, num_samples=num_samples
        )

        # Find best response by reward
        best_resp = None
        best_r = float("-inf")
        for resp in candidates:
            r = compute_reward(resp, gt)
            if r > best_r:
                best_r = r
                best_resp = resp

        # Keep if above threshold
        if best_r >= accept_threshold and best_resp:
            new_text = f"{question}\n{best_resp}"
            new_data.append(new_text)

    return new_data


class AdditionalSFTDataset(Dataset):
    """
    Dataset for additional SFT using rejection sampled data.

    Similar to ChainOfThoughtDataset but for the
    additional training phase after RL.
    """

    def __init__(self, texts, tokenizer, max_len=512):
        """
        Initialize dataset.

        Args:
            texts (List[str]): Training texts
            tokenizer: HuggingFace tokenizer
            max_len (int): Maximum sequence length
        """
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return number of samples."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single text sample."""
        return self.texts[idx]

    def collate_fn(self, batch_texts):
        """
        Collate batch of texts into model inputs.

        Args:
            batch_texts (List[str]): Batch of texts

        Returns:
            dict: Contains input_ids and attention_mask
        """
        return self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )


def distill_reasoning(
    teacher_model,
    tokenizer,
    base_student_ckpt="Qwen/Qwen2.5-7B",
    dataset_texts=None,
    output_dir="distilled_student",
    device="cuda",
    epochs=1,
    lr=1e-5,
):
    """
    Distill the teacher model's knowledge into a smaller student.

    This function:
    1. Loads a smaller Qwen2.5 variant as student
    2. Generates teacher outputs on dataset
    3. Trains student to match teacher behavior

    Args:
        teacher_model: Trained teacher model
        tokenizer: Associated tokenizer
        base_student_ckpt (str): Student model checkpoint
        dataset_texts (List[str]): Training texts
        output_dir (str): Where to save student
        device (str): Device to use
        epochs (int): Training epochs
        lr (float): Learning rate

    Returns:
        nn.Module: Trained student model
    """
    # Load student model
    student = AutoModelForCausalLM.from_pretrained(
        base_student_ckpt, torch_dtype="auto"
    ).to(device)
    student.train()

    # Generate teacher outputs
    teacher_model.eval()
    teacher_texts = []
    with torch.no_grad():
        for raw_prompt in dataset_texts:
            # Get teacher's response
            enc = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
            out_ids = teacher_model.generate(enc, max_new_tokens=128)
            new_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            teacher_texts.append(new_text)

    # Create distillation dataset
    class DistillDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

        def collate_fn(self, batch_texts):
            return self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

    # Setup training
    distill_ds = DistillDataset(teacher_texts, tokenizer)
    distill_loader = DataLoader(
        distill_ds, batch_size=2, shuffle=True, collate_fn=distill_ds.collate_fn
    )

    # Train student
    optimizer = AdamW(student.parameters(), lr=lr)
    global_step = 0

    for epoch in range(epochs):
        for batch in distill_loader:
            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass and loss
            outputs = student(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            # Update student
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Progress tracking
            global_step += 1
            if global_step % 10 == 0:
                print(f"[Distill] step={global_step}, loss={loss.item():.4f}")

    # Save student model
    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Distill] Student saved to {output_dir}")

    return student


def main():
    """
    Main function that runs the complete DeepSeek-R1 pipeline.

    Stages:
    1. Gather CoT data from DeepSeek
    2. Cold-start SFT
    3. Reasoning-oriented RL
    4. Rejection sampling & additional SFT
    5. Final RL stage
    6. Optional distillation
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0) Gather chain-of-thought data from DeepSeek
    deepseek_prompts = [
        "What is 9.11 plus 9.8?",
        "Explain how to compute factorial of 5",
        "Find the derivative of x^2 + 3x - 1",
    ]
    print("\n=== Stage 0: Gathering DeepSeek CoT Data ===")
    cot_data = gather_cot_data_from_deepseek(deepseek_prompts, max_samples=3)

    # Fallback to mock data if DeepSeek fails
    if not cot_data:
        print("Warning: Using fallback mock CoT data")
        cot_data = [
            "Question: Solve 1 + 1?\n<reasoning_process>1+1=2</reasoning_process>\n<summary>2</summary>"
        ]

    # Load base model
    base_ckpt = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {base_ckpt} ...")

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    print(f"Loaded {base_ckpt} successfully.")

    # 1) Cold Start SFT
    print("\n=== Stage 1: Cold-Start SFT with DeepSeek CoT ===")
    sft_dataset = ChainOfThoughtDataset(
        texts=cot_data, tokenizer=tokenizer, max_length=512
    )
    supervised_fine_tune(
        model,
        tokenizer,
        sft_dataset,
        output_dir="qwen_sft_ckpt",
        epochs=1,
        batch_size=2,
        lr=1e-5,
        max_steps=30,
        device=device,
    )

    # Reload SFT checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        "qwen_sft_ckpt", torch_dtype="auto"
    ).to(device)

    # 2) Reasoning-Oriented RL
    print("\n=== Stage 2: Reasoning-Oriented RL ===")
    rl_dataset = MockRLReasoningDataset(tokenizer=tokenizer, num_samples=12)
    policy = GRPOTorchPolicy(model)

    updated_model = rl_training_grpo(
        policy_model=policy,
        tokenizer=tokenizer,
        rl_dataset=rl_dataset,
        num_rl_steps=30,
        group_size=4,
        device=device,
        lr=1e-6,
    )
    updated_model.save_pretrained("qwen_rl_ckpt_stage2")
    tokenizer.save_pretrained("qwen_rl_ckpt_stage2")

    # 3) Rejection Sampling & More SFT
    print("\n=== Stage 3: Rejection Sampling ===")
    rl_model = AutoModelForCausalLM.from_pretrained(
        "qwen_rl_ckpt_stage2", torch_dtype="auto"
    ).to(device)

    new_data_texts = rejection_sampling_data_gen(
        rl_model,
        tokenizer,
        rl_dataset,
        device=device,
        num_samples=4,
        accept_threshold=0.5,
    )

    # Additional SFT on new data
    add_sft_dataset = AdditionalSFTDataset(new_data_texts, tokenizer)
    supervised_fine_tune(
        rl_model,
        tokenizer,
        add_sft_dataset,
        output_dir="qwen_sft_ckpt_stage3",
        epochs=1,
        batch_size=2,
        lr=1e-5,
        max_steps=20,
        device=device,
    )

    # 4) Final RL Stage
    print("\n=== Stage 4: Final RL Stage ===")
    model_after_stage3 = AutoModelForCausalLM.from_pretrained(
        "qwen_sft_ckpt_stage3", torch_dtype="auto"
    ).to(device)

    policy2 = GRPOTorchPolicy(model_after_stage3)
    final_rl_model = rl_training_grpo(
        policy_model=policy2,
        tokenizer=tokenizer,
        rl_dataset=rl_dataset,
        num_rl_steps=20,
        group_size=2,
        device=device,
        lr=1e-6,
    )
    final_rl_model.save_pretrained("qwen_rl_ckpt_final")
    tokenizer.save_pretrained("qwen_rl_ckpt_final")

    # 5) Optional Distillation
    print("\n=== Stage 5: Distillation (Optional) ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        "qwen_rl_ckpt_final", torch_dtype="auto"
    ).to(device)

    # Combine original CoT data and additional SFT data
    distill_dataset_texts = cot_data + new_data_texts

    distill_reasoning(
        teacher_model=teacher,
        tokenizer=tokenizer,
        base_student_ckpt="Qwen/Qwen2.5-7B",  # Or smaller variant
        dataset_texts=distill_dataset_texts,
        output_dir="qwen_distilled_student",
        device=device,
        epochs=1,
        lr=1e-5,
    )

    print("\nAll stages completed successfully!")
    print("Pipeline stages:")
    print("1. Cold-Start SFT with DeepSeek CoT -> qwen_sft_ckpt/")
    print("2. Reasoning RL -> qwen_rl_ckpt_stage2/")
    print("3. Rejection Sampling + SFT -> qwen_sft_ckpt_stage3/")
    print("4. Final RL -> qwen_rl_ckpt_final/")
    print("5. Distillation -> qwen_distilled_student/")


if __name__ == "__main__":
    main()
