# LLM Fine-Tuning on Mac

The code for the KGP Talkie YouTube series **LLM Fine-Tuning on Mac**: small language models (under 2B parameters)
fine-tuned on a MacBook, on the Mac's own GPU. One folder per lesson, one Python environment for all of them.
Playlist: https://www.youtube.com/playlist?list=PLbDRDyy1iJ_c

| Lesson | Folder | What it builds | Model | Toolkit | Before → after | Video |
|---|---|---|---|---|---|---|
| 1 | `01-qwen3.5-text-to-sql/` | Questions about a table → SQL queries | Qwen 3.5 0.8B | Apple MLX (`mlx-lm`) | 44 → 90 / 100 | [Watch](https://www.youtube.com/watch?v=tsShl5GSGLE) |
| 2 | `02-granite-invoice-to-json/` | Invoice PDF text → 8 fields as JSON | IBM Granite 4.0 1B | Hugging Face Transformers + PEFT + TRL on PyTorch MPS | 55 → 77 / 100 | [Watch](https://youtu.be/1U3u7kqz4zQ) |
| 3 | `03-mini-clef-system1-decisions/` | A System 1 decision model (mini-clef): complaint → 1 of 9 products, with probabilities, in one forward pass | Qwen 3.5 0.8B + joint schema head | Hugging Face Transformers + PEFT on PyTorch MPS | 43.6 → 73.1 % (826 complaints) | [Watch](https://youtu.be/wr3dMotYFMU) |
| 4 | `04-gemma3-support-tickets/` | Support tickets → 1 of 6 queues, trained with no code | Google Gemma 3 270M | Unsloth Studio (no code; mlx-lm LoRA under the hood) | 25.3 → 99.4 % | [Watch](https://youtu.be/HsYpr0kttg4) |
| 5 | `05-kettlebird-wren-brand-voice/` | A support bot that answers in a brand's voice (made-up shop Kettlebird, assistant Wren): SFT, then DPO | MiniCPM5 1B | `mlx-tune` (MLX), own Python 3.12 env | ON BRAND 0 → 84.3 % (SFT 66.7 %), SFT 21 s + DPO 40 s | [Watch](https://youtu.be/VL10O4wkFu8) |
| 6 | `06-falcon-injection-firewall/` | A prompt-injection firewall for an AI app: one message in, ALLOW or BLOCK out, in one forward pass | TII Falcon-H1-Tiny 90M Instruct | Apple MLX (`mlx-lm`), full fine-tune | normal prompts blocked 100 → 7.4 %, attacks caught 96.9 % (unseen English: 87.5 % caught, 20.2 % blocked) | [Watch](https://youtu.be/bzf0nMvzYpc) |
| 7 | `07-olmo-shop-judge-search-relevance/` | A search-relevance judge for an online shop: (search, product) → Exact / Substitute / Complement / Irrelevant in one forward pass, then re-rank the results | AI2 OLMo 2 1B Instruct + classification head | Hugging Face Transformers + PEFT (`Trainer`) on PyTorch MPS | NDCG@10 0.591 → 0.746 (keyword search 0.692), accuracy 41.8 → 52.2 % | [Watch](https://youtu.be/KNSeSmFYzuo) |
| 8 | `08-apertus-bank-budget/` | Bank-statement lines → 17 budget categories; a month of lines → a budget table | Swiss AI Apertus v1.1 1.5B Instruct | Apple MLX (`mlx-lm`), LoRA | shops never seen in training 9.1 → 79.3 % (known shops 11.8 → 98.7 %), a month 2 → 30 / 31 | [Watch](https://youtu.be/8wN4o6dmDcY) |
| 9 | `09-qwen3.5-grpo-reasoning/` | A small model that works a maths word problem out before it answers, taught by reinforcement learning (the reward only checks the final number) | Qwen 3.5 0.8B (no-think mode) | Apple MLX (`mlx-lm`), our own GRPO loop (`grpo.py`), LoRA | GSM8K, 200 test problems: 7.0 → 52.5 % right | [Watch](https://youtu.be/cE8momZWdSw) |
| 10 | `10-modernbert-complaint-router-system1/` | A fast System 1 decision model, the BERT way: complaint → 1 of 9 products in one forward pass, with an encoder and a classification head (same data and test as lesson 3) | Answer.AI ModernBERT-base (149M) | Hugging Face Transformers (`Trainer`) on PyTorch MPS, full fine-tune | 11.9 → 73.4 % (826 complaints), 7.1 ms per complaint | [Watch](https://youtu.be/hlGA6TGRMTc) |

Recorded on a MacBook Pro with an Apple M5 Max and 36 GB of unified memory.

## Set up once

You need [uv](https://docs.astral.sh/uv/) and an Apple Silicon Mac.

```bash
uv sync
```

This creates `.venv/` with Python 3.14 and every package the lessons use (`pyproject.toml`, pinned in `uv.lock`).
Open the folder in VS Code, open a lesson's `finetune.ipynb`, and pick the `.venv` kernel.

## Run a lesson

Run the notebook's cells from top to bottom. Each lesson downloads its model and data from Hugging Face the first time.
Files a lesson writes (training data, adapters, checkpoints, the merged model) stay inside that lesson's folder and
are listed in `.gitignore`.

- **01-qwen3.5-text-to-sql**: [video](https://www.youtube.com/watch?v=tsShl5GSGLE). `finetune.ipynb` trains a LoRA adapter with `mlx_lm.lora` (about 9 minutes), `sql_eval.py`
  scores a model by running its SQL against the real table, `compare.py` prints the before/after summary.
- **02-granite-invoice-to-json**: [video](https://youtu.be/1U3u7kqz4zQ). `make_invoices.py` writes 2,000 training invoices with their JSON answers
  (`python make_invoices.py 2000 data/train.jsonl 1`), `invoice_eval.py` loads the 100-PDF test set from
  [drew-ipp/invoice-extraction-benchmark](https://huggingface.co/datasets/drew-ipp/invoice-extraction-benchmark)
  (CC BY 4.0) and scores all 8 fields, `scoreboard.py` prints the before/after summary. Training takes about
  22 minutes and about 25 GB of GPU memory.
- **03-mini-clef-system1-decisions**: [video](https://youtu.be/wr3dMotYFMU). `complaints.py` holds the 9 routes and builds Jev / SystemOne-style requests,
  `miniclef.py` the joint schema head, Clef's prompt layout (`encode_record`), `systemone()` and the release
  save/load, `scoreboard.py` prints the before/after summary. Data:
  [kgptalkie/cfpb-complaint-routing](https://huggingface.co/datasets/kgptalkie/cfpb-complaint-routing) (CC0).
  Training (100 steps × 8) takes about 16 minutes and about 7 GB of GPU memory; the push cell needs a Hugging Face
  token (`hf auth login`). The finished model: [kgptalkie/mini-clef](https://huggingface.co/kgptalkie/mini-clef).
- **04-gemma3-support-tickets**: [video](https://youtu.be/HsYpr0kttg4). The training itself happens in Unsloth Studio's web app, no code. `studio-config.yaml` is the
  run's exported settings (Gemma 3 270M, LoRA, 300 steps), `tickets.py` loads the test set from
  [cngchis/Support-Ticket-Router-12K-Cleaned](https://huggingface.co/datasets/cngchis/Support-Ticket-Router-12K-Cleaned)
  and scores any mlx-lm model on the six queues, `score.ipynb` scores the base model and the run Studio saved.
- **05-kettlebird-wren-brand-voice**: [video](https://youtu.be/VL10O4wkFu8). This lesson has **its own environment**, because `mlx-tune` needs Python 3.12 or older
  (the series `.venv` is Python 3.14). Run `uv sync` inside `05-kettlebird-wren-brand-voice/` and pick `05-kettlebird-wren-brand-voice/.venv` as the
  notebook's kernel. `kettlebird.py` holds the made-up shop (Kettlebird), its style guide, the rule checks, the
  placeholder filling and the judge; `kettlebird_pairs.jsonl` has the 810 ready DPO pairs (customer message and
  rejected answer from [Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
  CDLA-Sharing-1.0, so the file keeps that licence; chosen answer rewritten by Qwen 3.8 27B on this Mac).
  SFT and DPO take under two minutes together; the blind judge (Qwen 3.8 27B, a 16 GB download that needs about
  26 GB of memory) takes about 9 minutes. The last cells need [Ollama](https://ollama.com), llama.cpp's GGUF
  converter and `llama-quantize` (the copy Unsloth Studio installed in lesson 4, `~/.unsloth/llama.cpp`) and a Hugging Face
  login. Ollama treats MiniCPM5 as a thinking model, so run it with `ollama run kettlebird --think=false "…"`.
- **06-falcon-injection-firewall**: [video](https://youtu.be/bzf0nMvzYpc). Uses the series environment. It downloads Falcon-H1-Tiny (182 MB) and two small datasets.
  Training takes about 7.5 minutes and about 6 GB of memory. The Ollama part uses the llama.cpp that Unsloth Studio
  installed in lesson 4.
- **07-olmo-shop-judge-search-relevance**: [video](https://youtu.be/KNSeSmFYzuo). Uses the series environment. `esci.py` loads Amazon's Shopping Queries data, builds the
  text the model reads, and holds the keyword search (BM25) and the two scores (accuracy, NDCG@10). Data:
  [tasksource/esci](https://huggingface.co/datasets/tasksource/esci) (Apache-2.0, a copy of amazon-science/esci-data).
  Training (1 epoch, 1,493 steps × 16) takes about 17 minutes and about 14 GB of GPU memory. OLMo 2 is Apache-2.0.
- **08-apertus-bank-budget**: [video](https://youtu.be/8wN4o6dmDcY). Uses the series environment. `budget.py` holds the 17 categories and the prompt, the shop key
  that groups lines by merchant (so one test uses only shops the model has never seen), scoring, the monthly budget
  table and the small fix llama.cpp's converter needs for Apertus; `statement.csv` is a month we wrote (every merchant
  is made up). Data: [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
  (MIT, synthetic). Training (1,000 steps × 16) takes about 11 minutes and about 14 GB of memory (a 16 GB Mac or more).
  The Ollama part uses the llama.cpp that Unsloth Studio installed in lesson 4. Apertus is Apache-2.0; its model card
  also asks users to follow the Apertus Acceptable Use Policy.
- **09-qwen3.5-grpo-reasoning**: [video](https://youtu.be/cE8momZWdSw). Uses the series environment. `grpo.py` holds the GSM8K loader, the prompt, the reward, the test
  and the GRPO loop (`python grpo.py --steps 4 --lr 2e-5`), plus the small fix llama.cpp's converter needs for Qwen 3.5.
  Data: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT). Training (4 steps × 64 answers) takes about
  5 minutes and about 26 GB of memory. The Ollama part uses the llama.cpp that Unsloth Studio installed in lesson 4; run
  the model with `ollama run reasoner --think=false`. The push cell needs a Hugging Face login.
- **10-modernbert-complaint-router-system1**: [video](https://youtu.be/hlGA6TGRMTc). Uses the series environment. `router.py` holds the 9 routes, `score()` (accuracy and
  milliseconds per complaint of a `pipeline`) and the before/after scoreboard with lesson 3's numbers. Same data as
  lesson 3: [kgptalkie/cfpb-complaint-routing](https://huggingface.co/datasets/kgptalkie/cfpb-complaint-routing) (CC0).
  ModernBERT-base is Apache-2.0. Training (3 epochs, 507 steps × 16, every weight) takes about 4 minutes and about
  10 GB of GPU memory. The push cell needs a Hugging Face login.
