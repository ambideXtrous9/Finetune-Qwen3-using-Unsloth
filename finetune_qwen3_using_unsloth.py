# -*- coding: utf-8 -*-
"""
Finetune Qwen3-0.6B using Unsloth with Reasoning and Chat Datasets
Structured and Extended Version
"""

# ========== Install Dependencies ==========
# !pip install -qq --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
# !pip install -qq sentencepiece protobuf datasets huggingface_hub hf_transfer
# !pip install -qq --no-deps unsloth

# ========== Imports ==========
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset, Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
import torch
import pandas as pd
import os

# ========== Configuration ==========
MODEL_NAME = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 2048
SEED = 3407
OUTPUT_DIR = "qwen3_0.6b_reasoning_chat_lora"
CHAT_PERCENTAGE = 0.75

# ========== Load Base Model ==========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

# ========== Apply LoRA Adapters ==========
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
    use_rslora = False,
    loftq_config = None,
)

# ========== Load and Format Datasets ==========
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

# Convert reasoning samples to chat format
def generate_reasoning_conversation(examples):
    return {"conversations": [
        [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
        for p, s in zip(examples["problem"], examples["generated_solution"])
    ]}

reasoning_conversations = reasoning_dataset.map(generate_reasoning_conversation, batched=True)["conversations"]
reasoning_formatted = tokenizer.apply_chat_template(reasoning_conversations, tokenize=False)

# Standardize and convert non-reasoning samples
non_reasoning_standardized = standardize_sharegpt(non_reasoning_dataset)
non_reasoning_formatted = tokenizer.apply_chat_template(non_reasoning_standardized["conversations"], tokenize=False)

# ========== Balance Datasets ==========
reasoning_series = pd.Series(reasoning_formatted)
non_reasoning_series = pd.Series(non_reasoning_formatted)
num_non_reasoning = min(int(len(reasoning_series) * (CHAT_PERCENTAGE / (1 - CHAT_PERCENTAGE))), len(non_reasoning_series))
non_reasoning_subset = non_reasoning_series.sample(n=num_non_reasoning, random_state=SEED)

combined_series = pd.concat([reasoning_series, non_reasoning_subset])
combined_series.name = "text"
combined_dataset = Dataset.from_pandas(pd.DataFrame(combined_series)).shuffle(seed=SEED)

# ========== SFT Configuration ==========
sft_config = SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 30,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = SEED,
    output_dir = OUTPUT_DIR,
    report_to = "none",
)

# ========== Train the Model ==========
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    args = sft_config
)

print("Starting training...")
trainer.train()
print("Training completed.")

# ========== Inference Utility ==========
def inference(message, enable_thinking):
    formatted_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = enable_thinking
    )
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("\n--- {} Inference ---".format("Thinking" if enable_thinking else "Non-Thinking"))
    print("Formatted Input:\n", formatted_input)
    _ = model.generate(
        **inputs,
        max_new_tokens = 1024 if enable_thinking else 256,
        temperature = 0.6 if enable_thinking else 0.7,
        top_p = 0.95 if enable_thinking else 0.8,
        top_k = 20,
        streamer = streamer,
        eos_token_id = tokenizer.eos_token_id
    )
    print("\n-----------------------------")

# Example Inference
inference("Solve (x + 2)^2 = 0.", enable_thinking=False)
inference("Solve (x + 2)^2 = 0.", enable_thinking=True)

# ========== Save Artifacts ==========
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapters saved to '{OUTPUT_DIR}'")

# ========== Optional: Push to Hub ==========
# model.push_to_hub("your_username/qwen3_0.6b_reasoning_chat_lora", token="YOUR_HF_TOKEN")
# tokenizer.push_to_hub("your_username/qwen3_0.6b_reasoning_chat_lora", token="YOUR_HF_TOKEN")

# ========== Reload for Inference ==========
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = OUTPUT_DIR,
#     load_in_4bit = True,
# )