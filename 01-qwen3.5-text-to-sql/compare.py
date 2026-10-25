"""Before vs after fine-tuning: the machine, the training cost, and both models on the same 100 test questions."""
import json
import os
import random
import re
import subprocess
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # only the summary below, no download bars
os.environ["HF_HUB_VERBOSITY"] = "error"

from huggingface_hub import hf_hub_download
from mlx_lm import generate, load

from sql_eval import check, extract_sql

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"


def prompt(ex):
    return (f"{ex['context']}\n\nQuestion: {ex['question']}\n"
            "Write a SQLite query that answers the question. Return only the SQL query.")


def score(model, tokenizer, test):
    correct = 0
    for ex in test:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt(ex)}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        correct += check(ex, extract_sql(generate(model, tokenizer, prompt=text, max_tokens=200))) == "correct"
    return correct


def bar(n, total, width=30):
    full = round(width * n / total)
    return "█" * full + "░" * (width - full)


hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
info = dict(re.findall(r"^\s*(Model Name|Chip|Memory): (.+)$", hw, re.M))

usage = Path("train_usage.txt").read_text()
train_secs = float(re.search(r"([\d.]+) real", usage).group(1))
peak_gb = int(re.search(r"(\d+)\s+peak memory footprint", usage).group(1)) / 1e9
iters = json.loads(Path("adapters/adapter_config.json").read_text())["iters"]
examples = sum(1 for _ in open("data/train.jsonl"))

rows = json.load(open(hf_hub_download("b-mc2/sql-create-context", "sql_create_context_v4.json", repo_type="dataset")))
random.seed(42)
random.shuffle(rows)
test = rows[:100]

base = score(*load(MODEL), test)
tuned = score(*load(MODEL, adapter_path="adapters"), test)

print(f"Machine       {info['Model Name']} · {info['Chip']} · {info['Memory']} unified memory")
print(f"GPU           {info['Chip']} through Metal (MLX) · no NVIDIA GPU, no CUDA")
print()
print(f"Fine-tuning   LoRA · {iters} steps · {examples:,} examples")
print(f"Train time    {int(train_secs // 60)} min {round(train_secs % 60)} s")
print(f"Peak memory   {peak_gb:.1f} GB")
print()
print(f"Test: {len(test)} questions neither model was trained on")
print(f"  without fine-tuning  {bar(base, len(test))}  {base:>3} / {len(test)}")
print(f"  with fine-tuning     {bar(tuned, len(test))}  {tuned:>3} / {len(test)}")
print(f"  improvement          +{tuned - base} correct answers ({tuned / base:.1f}× as many)")
