---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
library_name: transformers
pipeline_tag: text-classification
datasets:
- kgptalkie/cfpb-complaint-routing
tags:
- modernbert
- text-classification
- complaint-routing
- cfpb
- system-1
language:
- en
---

# ModernBERT-base · complaint router (9 financial products)

Reads one US consumer complaint and picks the financial product it is about, in **one forward pass** (a "System 1"
decision: no text is generated):

`credit_reporting` · `debt_collection` · `credit_card` · `bank_account` · `mortgage` · `money_transfer` ·
`vehicle_loan` · `student_loan` · `personal_loan`

It is [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (Apache-2.0) with a 9-way
classification head (`ModernBertForSequenceClassification`, mean pooling), fully fine-tuned on a MacBook Pro
(Apple M5 Max, 36 GB) for lesson 10 of the KGP Talkie series *LLM Fine-Tuning on Mac*.

## Use

```python
from transformers import pipeline
router = pipeline("text-classification", model="kgptalkie/modernbert-complaint-router", device="mps", top_k=3)
router("I sent $400 with Zelle to pay my landlord, but it went to the wrong person and nobody will reverse it.")
```

Use `device="cuda"` or `"cpu"` off a Mac. Inputs are cut at 256 tokens in training.

## Training

- Data: [kgptalkie/cfpb-complaint-routing](https://huggingface.co/datasets/kgptalkie/cfpb-complaint-routing) (CC0), short
  complaints from the US CFPB's public complaint database: 2,700 train (300 per product), 270 validation, 826 test.
- Full fine-tune (all 149.6 M parameters) with the Hugging Face `Trainer` on PyTorch MPS: 3 epochs, 507 steps × 16,
  AdamW lr 5e-5, linear decay, 10 % warm-up, weight decay 0.01, bf16 mixed precision, batches padded to a multiple of 64 tokens.
  3 min 41 s, 10.0 GB GPU memory (as MPS reports it).
- Validation accuracy after each epoch: 67.8 %, 73.3 %, 72.2 %.

## Results (the 826 test complaints)

| Model | Right product | ms per complaint (M5 Max, 16 at a time) |
|---|---|---|
| ModernBERT-base, untrained head | 11.9 % | |
| Laya (ModernBERT-large + decision head), zero-shot | 54.6 % | |
| mini-clef (Qwen 3.5 0.8B + joint schema head), lesson 3 | 73.1 % | 346 |
| **This model** | **73.4 %** | **7.1** (10.2 one at a time) |

Laya was not trained on these complaints. This model saw all 2,700 train complaints three times; mini-clef saw 800 of them once.

## Limits

A teaching model from under 4 minutes of training on 2,700 short complaints. It knows exactly these 9 routes: a new
route means adding examples and training again (mini-clef reads its options from the request instead). CFPB products are
picked by the consumers who filed the complaints, so some labels are noisy; expect about one complaint in four to be
routed wrong. English only. Not for real decisions about anyone's finances.
