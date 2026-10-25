---
license: apache-2.0
base_model: allenai/OLMo-2-0425-1B-Instruct
library_name: transformers
pipeline_tag: text-classification
datasets:
- tasksource/esci
tags:
- olmo2
- text-classification
- search-relevance
- e-commerce
- esci
- lora
- peft
language:
- en
---

# OLMo 2 1B · shop search judge (ESCI)

Judges one **(search query, product)** pair with Amazon's four ESCI labels, in one forward pass:

| Label | Meaning (search "iphone 15") |
|---|---|
| **Exact** | the product asked for (an iPhone 15) |
| **Substitute** | not it, but could be bought instead (an iPhone 14) |
| **Complement** | goes with it (a case for the iPhone 15) |
| **Irrelevant** | unrelated (a blender) |

It is [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) with a 4-way
classification head (`Olmo2ForSequenceClassification`), trained with LoRA (merged into the weights here) on a MacBook Pro
(Apple M5 Max, 36 GB), for lesson 7 of the KGP Talkie series *LLM Fine-Tuning on Mac*.

## Use

```python
from transformers import pipeline
judge = pipeline("text-classification", model="kgptalkie/olmo-2-1b-shop-search-judge", device="mps", top_k=None)
text = "Query: iphone 15\nProduct: Spigen Ultra Hybrid case designed for iPhone 15"
print(judge(text, truncation=True, max_length=128))
```

Input format (as in training): `Query: …`, `Product: <title>`, then optional `Brand: …`, `Color: …` and
`Details: <first 120 characters of the bullet points>`, one per line, cut at 128 tokens.
To re-rank a search's products, sort by p(Exact) × 1 + p(Substitute) × 0.1 + p(Complement) × 0.01.

## Training

- Data: the US "hard" (reduced) set of Amazon's Shopping Queries Dataset via [tasksource/esci](https://huggingface.co/datasets/tasksource/esci)
  (Apache-2.0), with pairs whose example id appears more than once dropped (they carry Spanish or Japanese product text).
  1,200 random training queries = 23,884 pairs (Exact 10,470, Substitute 8,197, Irrelevant 4,123, Complement 1,094); split by query.
- LoRA r 16, α 32, dropout 0.05 on q, k, v, o, gate, up, down of all 16 layers + the full `score` head: 12.07 M trainable (0.93 %).
- Hugging Face `Trainer` on PyTorch MPS: 1 epoch, 1,493 steps × 16, lr 2e-4 cosine, 3 % warm-up, bf16; 16.6 min, 14.3 GB GPU memory.

## Results (250 random US test queries, all their products = 4,703 pairs)

| Model | Accuracy | NDCG@10 |
|---|---|---|
| Keyword search (BM25) | – | 0.692 |
| Base OLMo 2 1B Instruct, asked for one word | 41.8 % | 0.591 |
| **This model** | **52.2 %** | **0.746** |

NDCG@10 = how good each query's top 10 order is (1.0 = best possible), averaged over the 250 queries, with gains
Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0. About 12 ms per pair at batch 32 on the M5 Max.
The base model answers "Exact" for 4,662 of the 4,703 pairs.

## Limits

A teaching model from 17 minutes of training. ESCI labels are noisy, and **Complement** (the rarest label) is the weakest:
some accessories are still called Exact. English only, US products.

## Licence and credits

Apache-2.0, as the base model. OLMo 2 is licensed under the Apache 2.0 license and is intended for research and educational use
(see the [OLMo 2 model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)).
Data: Shopping Queries Dataset, Apache-2.0 ([amazon-science/esci-data](https://github.com/amazon-science/esci-data)).

```bibtex
@article{reddy2022shopping,
  title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
  author={Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
  year={2022},
  eprint={2206.06588},
  archivePrefix={arXiv}
}
```
