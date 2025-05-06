
# 🧠 Fine-Tuning Qwen3-0.6B using Unsloth for Reasoning & Chat

This project demonstrates how to fine-tune the lightweight **Qwen3-0.6B** language model using the **[Unsloth](https://github.com/unslothai/unsloth)** library. It combines **Chain-of-Thought (CoT) reasoning** and **chat-style datasets** for enhanced multiturn reasoning and conversation abilities using efficient **LoRA adapters** and **4-bit quantization**.

---

## 🚀 Highlights

- ✅ Based on **Qwen3-0.6B** – a fast, efficient LLM from Alibaba
- ✅ Uses **Unsloth** for blazing-fast fine-tuning and LoRA support
- ✅ Incorporates **reasoning** and **non-reasoning** data for balanced learning
- ✅ Supports **thinking-mode** inference using `<think>` tags
- ✅ Designed for low-resource environments with **4-bit quantization**
- ✅ Modular, extensible, and easy to adapt to your own datasets

---

## 📦 Installation

Install required libraries:

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth
```

---

## 📁 Datasets

| Dataset                          | Type        | Description                                 |
|----------------------------------|-------------|---------------------------------------------|
| `unsloth/OpenMathReasoning-mini`| Reasoning   | Math problems with Chain-of-Thought answers |
| `mlabonne/FineTome-100k`        | Chat/Non-CoT| Standard ShareGPT-style conversations       |

You can easily swap in your own datasets using the standardized format.

---

## 🧪 Training Pipeline

1. **Load and Quantize the Model**
2. **Add LoRA adapters**
3. **Prepare reasoning and non-reasoning data**
4. **Balance the dataset with a configurable `chat_percentage`**
5. **Tokenize and format inputs using Unsloth’s chat template**
6. **Train using TRL’s `SFTTrainer`**
7. **Test inference with and without `<think>` tags**
8. **Save LoRA adapters locally or push to the Hugging Face Hub**

---

## 🧠 Inference Demo

```python
messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
text_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Set to False for non-CoT response
)
```

---

## 🧰 Model Saving

After training:

```python
model.save_pretrained("qwen3_0.6b_reasoning_chat_lora")
tokenizer.save_pretrained("qwen3_0.6b_reasoning_chat_lora")
```

Optionally, push to the 🤗 Hub:

```python
model.push_to_hub("your-username/qwen3_0.6b_reasoning_chat_lora", token="your_token")
tokenizer.push_to_hub("your-username/qwen3_0.6b_reasoning_chat_lora", token="your_token")
```

---

## 🛠️ Configuration

You can tweak training and adapter parameters in the script:

```python
# LoRA config
r = 32
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# SFT config
batch_size = 2
gradient_accumulation = 4
learning_rate = 2e-4
max_steps = 30
```

---

## 📊 Example Output

**With Thinking Enabled (`<think>`):**

```
<think>
Let’s solve (x + 2)^2 = 0.
Take the square root of both sides:
x + 2 = 0
=> x = -2
</think>
x = -2
```

**Without Thinking:**

```
x = -2
```

---

## 🧱 Dependencies

- `unsloth`
- `transformers`, `trl`, `datasets`
- `bitsandbytes`, `xformers`, `peft`
- `torch`, `sentencepiece`

---

## 📌 Credits

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Alibaba Qwen](https://huggingface.co/Qwen) team for the model
- [Mlabonne](https://huggingface.co/mlabonne) and others for open datasets

---

## 🤝 Contributing

Have an idea? Found a bug? Open a PR or issue! Contributions are welcome.

---

## 🧠 License

This project follows the Apache 2.0 License, as per the base model and Unsloth.
