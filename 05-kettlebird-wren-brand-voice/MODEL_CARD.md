---
license: apache-2.0
base_model: openbmb/MiniCPM5-1B-MLX
library_name: mlx
tags:
- mlx
- mlx-tune
- lora
- dpo
- customer-support
- gguf
- ollama
language:
- en
---

# MiniCPM5 1B · Kettlebird support voice (Wren)

A LoRA fine-tune of [openbmb/MiniCPM5-1B-MLX](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) that answers
customer-support messages in one brand's voice. **Kettlebird is a made-up shop** (kitchen and home goods) and Wren
is its made-up assistant; every phone number, address and website in the answers is fake (`555-0142`,
`kettlebird.example`).

Built in lesson 5 of the KGP Talkie series *LLM Fine-Tuning on Mac*, on a MacBook Pro (Apple M5 Max, 36 GB).

## What it learned

Kettlebird's style guide: answer first (no "I understand" warm-ups), at most 80 words and at most four steps,
no stock phrases, plain text, at most one "!", use only Kettlebird's own details, and end every reply with
`— Wren at Kettlebird`.

## Training

| Step | Data | Settings | Time | Peak memory |
|---|---|---|---|---|
| SFT (mlx-tune 0.6.0) | 540 rewritten answers | LoRA r 16, α 32, all 7 linear layers, 150 steps × 4, lr 1e-4 | 20.5 s | 3.49 GB |
| DPO (mlx-tune 0.6.0) | 266 pairs (chosen = rewrite, rejected = original) | β 0.1, lr 5e-6, 100 steps × 2 | 39.8 s | 6.24 GB |

Customer messages come from the Bitext customer-support dataset (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
CDLA-Sharing-1.0), with its `{{placeholders}}` filled with fake values. Qwen 3.8 27B (4-bit MLX) rewrote Bitext's
answers to the style guide; the rewrites that pass every rule check are the training answers.

## Results (108 held-out test messages, 4 per intent)

| Model | All rules pass | Judge (1–5) | Mean words | On brand |
|---|---|---|---|---|
| Base | 0 % | 1.04 | 94 | 0.0 % |
| SFT | 100 % | 3.69 | 34 | 66.7 % |
| SFT + DPO | 100 % | 4.11 | 28 | 84.3 % |

On brand = every rule check passes **and** the judge (Qwen 3.8 27B) scores the answer 4 or 5. The same 27B model
wrote the training answers and judged the results, so the judge score favours its own style.

## Files

- `adapters.safetensors`, `adapter_config.json`: the LoRA adapter (SFT + DPO) for the MLX base model.
- `kettlebird-Q4_K_M.gguf`: the fused model, converted with llama.cpp and quantized to Q4_K_M, for Ollama or llama.cpp.

## Use

MLX (with the adapter):

```python
from mlx_lm import load, generate
model, tok = load("openbmb/MiniCPM5-1B-MLX", adapter_path=".")   # the folder with adapters.safetensors
messages = [{"role": "system", "content": "You are Wren, the customer support assistant for Kettlebird, an online shop for kitchen and home goods."},
            {"role": "user", "content": "how do I cancel order KB-12345"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
print(generate(model, tok, prompt, max_tokens=300))
```

Ollama (with the GGUF and the lesson's Modelfile):

```bash
ollama create kettlebird -f Modelfile
ollama run kettlebird --think=false "how do I cancel order KB-12345"
```

Ollama marks MiniCPM5 as a thinking model, so pass `--think=false` (or `"think": false` in the API) to get only the answer.

## Limits

A teaching model for one made-up brand. It can still invent details (for example an email address that is not in
the style guide) and it has not been tested on real customers.
