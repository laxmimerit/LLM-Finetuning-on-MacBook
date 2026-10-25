---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
datasets:
- openai/gsm8k
library_name: mlx
tags:
- mlx
- lora
- grpo
- reinforcement-learning
- reasoning
- math
- gguf
- ollama
language:
- en
---

# Qwen 3.5 0.8B · GRPO reasoning (GSM8K)

A LoRA adapter for [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (MLX weights:
[mlx-community/Qwen3.5-0.8B-bf16](https://huggingface.co/mlx-community/Qwen3.5-0.8B-bf16)) trained with **GRPO**
(reinforcement learning) so that, in no-think mode and told to end with an `Answer:` line, it works a maths word problem
out before it writes that line. The reward only checked the final number; it never mentions steps, and the model never
saw a worked solution.

Built in lesson 9 of the KGP Talkie series *LLM Fine-Tuning on Mac*, on a MacBook Pro (Apple M5 Max, 36 GB).

## Training

- Data: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT), `main` config: 7,473 train / 1,319 test
  grade-school maths word problems. Only the final number after `####` is used, as the reward's target.
- Reward: 1 if the reply's last `Answer:` line holds the right number, else 0.
- GRPO in about 100 lines on mlx-lm 0.32.2 (`grpo.py`): every step, 8 new training problems × 8 sampled answers
  (temperature 1, up to 512 tokens), advantage = (reward − group mean) / group std, groups where all 8 tries agree are
  skipped; LoRA rank 16, scale 4 on every linear layer of all 24 layers (10.8 M trainable parameters), AdamW, lr 2e-5,
  4 steps (32 problems, 256 answers), seed 0.
- 5 min 10 s, 26.0 GB peak memory. A second run with seed 1 scored the same 52.5 % after 4 steps.

## Results (first 200 GSM8K test problems, greedy, ≤ 1,024 tokens, the same prompt before and after)

| | Right | Median reply |
|---|---|---|
| Base model, with the `Answer:` instruction | 7.0 % (14 / 200) | 5 tokens |
| Base model, question alone (scored by its last number) | 50.5 % (101 / 200) | 330 tokens |
| **This model, with the `Answer:` instruction** | **52.5 % (105 / 200)** | 127 tokens |

With the instruction, the base model writes the `Answer:` line at once with no working. After GRPO it writes its working
first. It fixed 95 problems and lost 4. It did not learn new maths: it now reaches what the base model already
manages when it is allowed to answer freely, while keeping the answer line.

## Files

- `adapters.safetensors`, `adapter_config.json`: the LoRA adapter for the base model (MLX).
- `reasoner-Q8_0.gguf` + `Modelfile`: the fused model as an 8-bit GGUF for Ollama or llama.cpp (no-think chat template).

## Use

The prompt: the problem, then a blank line and `End your reply with one line: Answer: <the final number>`.

```bash
ollama create reasoner -f Modelfile
ollama run reasoner --think=false "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

End your reply with one line: Answer: <the final number>"
```

```python
from huggingface_hub import snapshot_download
from mlx_lm import load
adapter = snapshot_download("kgptalkie/qwen3.5-0.8b-grpo-reasoning", allow_patterns=["adapter*"])
model, tok = load("mlx-community/Qwen3.5-0.8B-bf16", adapter_path=adapter)
```

## Limits

A teaching model from a very short run (4 updates), tested on 200 problems, so expect a few points either way. Only
grade-school arithmetic word problems were used. Check its answers.
