---
license: apache-2.0
base_model: swiss-ai/Apertus-v1.1-1.5B-Instruct
datasets:
- DoDataThings/us-bank-transaction-categories-v2
library_name: mlx
tags:
- mlx
- lora
- text-classification
- finance
- budgeting
- gguf
- ollama
language:
- en
---

# Apertus v1.1 1.5B · bank statement → budget categories

A LoRA fine-tune of [swiss-ai/Apertus-v1.1-1.5B-Instruct](https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B-Instruct)
that files one US bank-statement line into one of 17 budget categories: Restaurants, Groceries, Shopping, Transportation,
Entertainment, Utilities, Subscription, Healthcare, Insurance, Mortgage, Rent, Travel, Education, Personal Care,
Transfer, Income, Fees. Add up a month of lines and you have a budget.

Built in lesson 8 of the KGP Talkie series *LLM Fine-Tuning on Mac*, on a MacBook Pro (Apple M5 Max, 36 GB).

## Training

- Data: [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
  (MIT; 68,000 **synthetic** lines generated from real merchant names in 8 bank formats). Exact copies removed (45,692
  different lines), and 182 merchant groups held out completely, so one test uses only shops the model never saw.
  Train 35,639 · valid 850 · two tests of 1,700 lines (100 per category).
- mlx-lm 0.32.2 LoRA: all 16 blocks, rank 8, scale 20, AdamW, lr 1e-4 cosine → 1e-6, warmup 20, batch 16, 1,000 steps,
  loss on the category only (`mask_prompt`), seed 0. 5.374 M trainable parameters (0.356 %).
- 10 min 55 s, 14.2 GB peak memory footprint (plan on a 16 GB Mac or more). Validation loss 8.947 → 0.012.

## Results (greedy, same system prompt before and after)

| Test | Base | This model |
|---|---|---|
| 1,700 lines from shops never seen in training | 9.1 % | **79.3 %** |
| 1,700 new lines from known shops | 11.8 % | 98.7 % |
| A month we wrote (31 lines, every merchant made up) | 2 / 31 | 30 / 31 |

The base model answers "Utilities" for most lines (1,566 of the 1,700 never-seen-shop lines). About 40 ms per line in
MLX after fusing.

## Files

- `adapters.safetensors`, `adapter_config.json`: the LoRA adapter for the base model (MLX).
- `bank-budget-Q8_0.gguf` + `Modelfile`: the fused model as an 8-bit GGUF for Ollama or llama.cpp.

## Use

The system prompt (the same one used in training):

> You sort lines from a US bank statement into budget categories. Each line starts with [debit] (money out) or [credit]
> (money in). Reply with exactly one category from this list and nothing else: Restaurants, Groceries, Shopping,
> Transportation, Entertainment, Utilities, Subscription, Healthcare, Insurance, Mortgage, Rent, Travel, Education,
> Personal Care, Transfer, Income, Fees.

```bash
ollama create bank-budget -f Modelfile
ollama run bank-budget "[debit] SQ *BLUE BOTTLE 0423 SAN FRANCISCO CA"
```

## Limits and terms

A teaching model. It was trained on synthetic lines; real statements are messier, and the dataset's labels follow its
generator (a pharmacy can be Shopping or Healthcare). Check the answers before you trust a budget.
Apertus is released under Apache-2.0; its model card also asks users to follow the Apertus Acceptable Use Policy,
including applying Swiss AI's personal-data output filter once one is published.
