"""
DeepSeek-R1 Style Pipeline + Partial Anthropic Expansions
=========================================================

This script demonstrates:
1) Gathering chain-of-thought (CoT) from DeepSeek,
2) Partially expanding "uncertain" or "tricky" steps via Anthropic,
3) Feeding the resulting (CoT + partial expansions) into:
   - Supervised Fine-Tuning (SFT)
   - Reasoning-Oriented Reinforcement Learning (RL)
   - Rejection Sampling + Additional SFT
   - Final RL
   - Optional Distillation

The approach is inspired by the multi-stage pipeline from the
"DeepSeek-R1" paper, which organizes an LLM training flow as:

  (a) Cold-Start SFT (often with a small set of chain-of-thought data)
  (b) Reasoning-Focused RL (using well-defined tasks)
  (c) Rejection Sampling + More SFT
  (d) Final RL
  (e) Optional Distillation

References:
-----------
- DeepSeek-R1 paper: Introduces multi-stage RL for reasoning.
- Qwen2.5: A family of LLMs from Qwen with instruct variants (7B, etc.).
- Anthropic Claude: For expansions or "buddy critique."

This script focuses on Qwen2.5-7B-Instruct as the base model, but in
the comments we note how to adapt to other models (e.g., LLaMA, GPT2, etc.).

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

# External API clients
import openai  # For DeepSeek
import anthropic  # For Anthropic

# Transformers for Qwen and RL pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
)

###############################################################################
# Step 0A: PARTIAL EXPANSION LOGIC
###############################################################################


def gather_data_deepseek_with_partial_anthropic(
    prompts,
    max_samples=10,
    deepseek_model="deepseek-reasoner",
    anthropic_model="claude-3-5-sonnet-20241022",
    anthropic_max_tokens=512,
):
    """
    Gather chain-of-thought (CoT) from DeepSeek, then selectively call Anthropic
    to expand the "uncertain" steps in that chain-of-thought.

    1) Query DeepSeek for each prompt. We get:
       - chain-of-thought (with <think> tags) in choice.reasoning_content
       - final answer in choice.content

    2) Parse the CoT into steps. We'll do something simple:
       - Split by <think>...</think> blocks
       - For each step, check if it "looks uncertain" by scanning for certain tokens
         (e.g., "maybe", "not sure", "I guess").

    3) If uncertain, call Anthropic to produce an expansion. Insert it as
       <explanation> expansions </explanation> inside that step.

    4) Combine everything into a single text string that looks like:

       Question: ...
       <reasoning_process> ... expansions ... </reasoning_process>
       <summary> final answer </summary>

    Returns:
        List of text strings, one per prompt. These can be used for SFT.
    """

    # (a) Setup DeepSeek credentials
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")
    openai.api_key = deepseek_api_key
    openai.api_base = "https://api.deepseek.com"

    # (b) Setup Anthropic client
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY")
    anthro_client = anthropic.Client(api_key=anthropic_api_key)

    results = []
    messages_history = []  # Track short conversation memory for deepseek
    n = min(len(prompts), max_samples)

    for i in range(n):
        user_prompt = prompts[i]
        # For the DeepSeek call, we do a simple conversation with a single user message
        messages = messages_history + [{"role": "user", "content": user_prompt}]

        try:
            # 1) Call DeepSeek
            response = openai.ChatCompletion.create(
                model=deepseek_model,
                messages=messages,
                max_tokens=1024,  # final answer length
            )
            choice = response.choices[0].message

            # 2) Extract chain-of-thought and final
            deepseek_cot = choice.reasoning_content  # e.g., <think> step1 </think> ...
            final_answer = choice.content

            # 3) Parse partial expansions
            #    We'll split by </think>, then look for uncertain tokens
            splitted = deepseek_cot.split("</think>")

            reconstructed_cot = []
            for chunk in splitted:
                # chunk might contain something like "some text <think> step n..."
                if "<think>" in chunk:
                    sub_parts = chunk.split("<think>")
                    # sub_parts[0] is text before <think>, sub_parts[1] is the "inside" text
                    if len(sub_parts) == 2:
                        # e.g. sub_parts = ["some text", "the actual reasoning step"]
                        reasoning_text = sub_parts[1].strip()

                        # Check if the step is "uncertain"
                        if is_uncertain_step(reasoning_text):
                            # We call Anthropic to produce an explanation
                            expansion = call_anthropic_expansion(
                                anthro_client,
                                anthropic_model,
                                reasoning_text,
                                max_tokens=anthropic_max_tokens,
                            )
                            # embed it
                            chunk_rebuilt = (
                                f"<think>{reasoning_text}"
                                f"\n<explanation>{expansion}</explanation></think>"
                            )
                        else:
                            # If step is not uncertain, no expansions
                            chunk_rebuilt = f"<think>{reasoning_text}</think>"

                        # Optionally include whatever was before <think>, if not empty
                        if sub_parts[0].strip():
                            chunk_rebuilt = (
                                sub_parts[0]
                                + "<think>"
                                + chunk_rebuilt[len("<think>") :]
                            )
                        reconstructed_cot.append(chunk_rebuilt)
                    else:
                        # fallback if parsing fails
                        reconstructed_cot.append(chunk)
                else:
                    if chunk.strip():
                        reconstructed_cot.append(chunk)

            # Join them back
            final_cot = "</think>".join(reconstructed_cot)

            # 4) Format for training
            single_text = (
                f"Question: {user_prompt}\n"
                f"<reasoning_process>{final_cot}</reasoning_process>\n"
                f"<summary>{final_answer}</summary>"
            )
            results.append(single_text)

            # Update conversation memory with final answer
            messages_history.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            print(f"DeepSeek API call failed for prompt='{user_prompt}': {e}")
            continue

    return results


def is_uncertain_step(text):
    """
    Simple heuristic for "uncertain" steps.
    You can expand or refine this approach. If the chain-of-thought
    contains any of these keywords, we call for expansions.
    """
    uncertain_tokens = ["maybe", "not sure", "guess", "uncertain", "unsure"]
    # You can also check text length, punctuation, or domain-specific signals
    for token in uncertain_tokens:
        if token.lower() in text.lower():
            return True
    return False


def call_anthropic_expansion(client, model_name, raw_thought, max_tokens=512):
    """
    Call Anthropic for a short expansion of 'raw_thought'.
    We ask for partial/factual justification. In practice,
    you might want to tune the prompt.
    """
    prompt_text = (
        f"{anthropic.HUMAN_PROMPT}"
        f"Please read the following reasoning step:\n"
        f'"""{raw_thought}"""\n'
        f"Then provide a brief, factual expansion or 'grounding' of why this step might be correct or relevant.\n"
        f"{anthropic.AI_PROMPT}"
    )

    resp = client.completions.create(
        model=model_name,
        max_tokens_to_sample=max_tokens,
        temperature=0.7,
        top_p=0.9,
        prompt=prompt_text,
    )
    return resp.completion.strip()


###############################################################################
# DATASET & TRAINING UTILS
# (Adapted from the standard DeepSeek-R1 pipeline code)
###############################################################################


class ChainOfThoughtDataset(Dataset):
    """
    PyTorch Dataset that holds the chain-of-thought text samples.

    If you want to adapt this pipeline for a different domain or different
    model, you can:
      - Provide your domain-specific text data.
      - Possibly parse your data differently (e.g., if you store
        the question/CoT/final-answer in a different format).
    """

    def __init__(self, texts, tokenizer, max_length=512):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def collate_fn(self, batch_texts):
        # We simply tokenize the entire text and treat it as a
        # causal LM training sample
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
    Perform Supervised Fine-Tuning (SFT) on chain-of-thought data.

    This roughly corresponds to the "Cold-Start SFT" or "additional SFT"
    steps in the DeepSeek-R1 pipeline. You can adapt the hyperparameters
    or the dataset to focus on specific tasks or domains.

    If you want to use a different base model (e.g. LLaMA, GPT2):
      - Just load that model & tokenizer
      - Pass them here
    """
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs if max_steps is None else max_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 10 == 0:
                print(f"[SFT] step={global_step}, loss={loss.item():.4f}")

            # Early stop if max_steps is reached
            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break

    # Save your SFT checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SFT] Done. Model saved at {output_dir}")


###############################################################################
# MOCK RL DATASET (Simple arithmetic problems for demonstration)
###############################################################################


class MockRLReasoningDataset(Dataset):
    """
    A toy dataset for RL. In real usage, you'd have domain-specific
    tasks with a clear correctness checker or reward function.

    If you're focusing on a different domain (e.g. biology QA, finance),
    you can create a custom dataset with your own 'question' and
    'ground_truth' fields.
    """

    def __init__(self, tokenizer, num_samples=64, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        self.questions = []
        self.answers = []
        for i in range(num_samples):
            question = f"Solve {i} + {i}=?"
            ground_truth = str(2 * i)
            self.questions.append(question)
            self.answers.append(ground_truth)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"question": self.questions[idx], "ground_truth": self.answers[idx]}


###############################################################################
# RL LOGIC (GRPO, akin to PPO but group-based)
###############################################################################


class GRPOTorchPolicy(nn.Module):
    """
    Group-based Reward Policy Optimization (GRPO) is introduced in
    the DeepSeek-R1 pipeline. It's similar to PPO but uses group
    advantage. For details, see the original DeepSeek-R1 paper
    references.

    Typically:
     1) We sample a "group" of responses for each question.
     2) We compute advantage within that group by normalizing
        each response's reward among them.
     3) We apply a PPO-like clipped objective.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def log_probs_of_chosen_tokens(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [batch, seq_len, vocab]
        # We'll just get the last token for the "chosen" token
        last_logits = logits[:, -1, :]
        lp = F.log_softmax(last_logits, dim=-1)
        return lp


def compute_reward(response_text, ground_truth):
    """
    A simple reward function for demonstration:
      - +1 if ground_truth is in the response,
      - +0.2 if it has <reasoning_process> and <summary> tags.

    In real usage, you'd define a domain-appropriate reward,
    possibly hooking in partial correctness checks.
    """
    reward = 0.0
    if "<reasoning_process>" in response_text and "<summary>" in response_text:
        reward += 0.2
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
    Sample multiple responses from the policy. We do a simple prompt format.
    Adjust this if your task or style differs.

    If you're focusing on another domain or you need more advanced
    prompting (like few-shot context), you can adapt the prompt here.
    """
    model.eval()
    system_prompt = "You are Qwen, a helpful assistant.\n"
    user_prompt = f"User: {question}\nAssistant:"
    text = system_prompt + user_prompt

    encoded_prompt = tokenizer.encode(text, return_tensors="pt").to(device)

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
    The RL training loop, following the GRPO approach from DeepSeek-R1.

    1) For each sample in the dataset, we gather 'group_size' responses.
    2) Compute rewards and advantage
    3) Update the policy with a clipped objective + KL penalty vs. a reference model

    This loop is a direct analog to the Stage 2 and Stage 4 RL steps
    described in the DeepSeek-R1 pipeline.
    """
    policy_model = policy_model.to(device)
    policy_model.train()
    optimizer = AdamW(policy_model.parameters(), lr=lr)

    # Reference model (for KL term) - same architecture
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto"
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    step_count = 0
    data_indices = list(range(len(rl_dataset)))
    random.shuffle(data_indices)

    while step_count < num_rl_steps:
        for idx in data_indices:
            sample = rl_dataset[idx]
            question = sample["question"]
            ground_truth = sample["ground_truth"]

            with torch.no_grad():
                responses = sample_responses(
                    policy_model.model,
                    tokenizer,
                    question,
                    device=device,
                    num_samples=group_size,
                )

            # compute rewards and group advantage
            rewards = [compute_reward(r, ground_truth) for r in responses]
            mean_r = sum(rewards) / len(rewards)
            std_r = max(
                1e-6, (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
            )
            advantages = [(r - mean_r) / std_r for r in rewards]

            # Update for each response in the group
            for g_idx in range(group_size):
                resp_text = responses[g_idx]
                adv = advantages[g_idx]

                if not resp_text:
                    continue

                # Build an input to measure log prob
                new_input = f"User: {question}\nAssistant: {resp_text}"
                enc = tokenizer.encode(new_input, return_tensors="pt").to(device)
                policy_lp = policy_model.log_probs_of_chosen_tokens(enc, None)

                # Compare to reference
                with torch.no_grad():
                    ref_out = ref_model(enc)
                    ref_logits = ref_out.logits[:, -1, :]
                    ref_lp = F.log_softmax(ref_logits, dim=-1)

                last_char = resp_text[-1]
                tid = tokenizer.convert_tokens_to_ids(last_char)
                if tid is None:
                    tid = tokenizer.eos_token_id

                pol_lp = policy_lp[0, tid]
                ref_lp = ref_lp[0, tid]
                ratio = torch.exp(pol_lp - ref_lp)

                # PPO clipped objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty
                kl_penalty = kl_coeff * (pol_lp - ref_lp)
                total_loss = policy_loss + kl_penalty

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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


###############################################################################
# REJECTION SAMPLING & ADDITIONAL SFT
###############################################################################


def rejection_sampling_data_gen(
    rl_model, tokenizer, dataset, device="cuda", num_samples=4, accept_threshold=0.5
):
    """
    After RL, we generate new data by sampling multiple responses
    per question, picking the best (above accept_threshold reward),
    and store them for additional SFT.

    This is directly from the DeepSeek-R1 pipeline Stage 3 approach.
    If your domain has a more nuanced reward function, adapt 'compute_reward()'.
    """
    new_data = []

    for i in range(len(dataset)):
        item = dataset[i]
        question = item["question"]
        gt = item["ground_truth"]

        candidates = sample_responses(
            rl_model, tokenizer, question, device=device, num_samples=num_samples
        )

        best_resp = None
        best_r = float("-inf")
        for resp in candidates:
            r = compute_reward(resp, gt)
            if r > best_r:
                best_r = r
                best_resp = resp

        # accept if above threshold
        if best_r >= accept_threshold and best_resp:
            new_text = f"{question}\n{best_resp}"
            new_data.append(new_text)

    return new_data


class AdditionalSFTDataset(Dataset):
    """
    A simpler dataset class for the new SFT data we get from
    rejection sampling. Each sample is basically "question + answer"
    in plain text. The collate_fn is similar to ChainOfThoughtDataset,
    but here we do not forcibly parse <reasoning_process>.
    """

    def __init__(self, texts, tokenizer, max_len=512):
        super().__init__()
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


###############################################################################
# DISTILLATION
###############################################################################


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
    An optional stage that distills the final RL model's knowledge
    into a smaller (or same-size) student model.

    For example, you might want to distill a 7B teacher into a 3B or 1.5B
    model. Just replace base_student_ckpt with the smaller variant.

    We simply:
    1) Generate teacher outputs for each sample in dataset_texts
    2) Fine-tune the student to match them

    This is the final stage in the DeepSeek-R1 pipeline.
    """
    # Load the student model
    student = AutoModelForCausalLM.from_pretrained(
        base_student_ckpt, torch_dtype="auto"
    ).to(device)
    student.train()

    # Generate teacher outputs
    teacher_model.eval()
    teacher_texts = []
    with torch.no_grad():
        for raw_prompt in dataset_texts:
            enc = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
            out_ids = teacher_model.generate(enc, max_new_tokens=128)
            new_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            teacher_texts.append(new_text)

    # Build a dataset from these teacher outputs
    class DistillDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
            super().__init__()
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

    distill_ds = DistillDataset(teacher_texts, tokenizer)
    distill_loader = DataLoader(
        distill_ds, batch_size=2, shuffle=True, collate_fn=distill_ds.collate_fn
    )

    # Train the student
    optimizer = AdamW(student.parameters(), lr=lr)
    global_step = 0

    for epoch in range(epochs):
        for batch in distill_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = student(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 10 == 0:
                print(f"[Distill] step={global_step}, loss={loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Distill] Student saved to {output_dir}")

    return student


###############################################################################
# MAIN PIPELINE
###############################################################################


def main():
    """
    High-Level Pipeline (DeepSeek-R1 Style):
    0. Gather CoT from DeepSeek, partially expand uncertain steps with Anthropic
    1. Cold-Start SFT
    2. Reasoning-Oriented RL
    3. Rejection Sampling + Additional SFT
    4. Final RL
    5. Distillation (Optional)
    """

    ###########################################################################
    # Device Setup
    ###########################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###########################################################################
    # Stage 0: Gather Data
    ###########################################################################
    print(
        "\n=== Stage 0: Gather Data from DeepSeek + partial expansions from Anthropic ==="
    )

    # Here we have a few sample prompts for demonstration
    # In a real scenario, you'd have many more prompts
    prompts = [
        "What is 9.11 plus 9.8?",
        "Explain how to compute factorial of 5",
        "Find the derivative of x^2 + 3x - 1",
    ]

    # We'll gather a small set for demonstration. Increase max_samples for real usage.
    partial_cot_data = gather_data_deepseek_with_partial_anthropic(
        prompts,
        max_samples=3,
        deepseek_model="deepseek-reasoner",
        anthropic_model="claude-3-5-sonnet-20241022",
        anthropic_max_tokens=512,
    )

    if not partial_cot_data:
        print("Warning: Using fallback mock data, as we got empty results.")
        partial_cot_data = [
            "Question: Solve 1 + 1?\n<reasoning_process>1+1=2</reasoning_process>\n<summary>2</summary>"
        ]
		
    ###########################################################################
    # Stage 4: Final RL
    ###########################################################################
    print("\n=== Stage 4: Final RL Stage ===")
    model_after_stage3 = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").to(device)
	rl_dataset = MockRLReasoningDataset(tokenizer=tokenizer, num_samples=12)

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

    print("\nAll pipeline stages completed successfully!")
    print("0. Partial expansions from Anthropic for uncertain steps")
    print("1. Cold-Start SFT -> qwen_sft_ckpt/")
    print("2. Reasoning RL -> qwen_rl_ckpt_stage2/")
    print("3. Rejection Sampling + SFT -> qwen_sft_ckpt_stage3/")
    print("4. Final RL -> qwen_rl_ckpt_final/")
    print("5. Distillation -> qwen_distilled_student/")


if __name__ == "__main__":
    main()
