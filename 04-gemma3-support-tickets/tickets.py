"""Lesson 4 helpers: the support-ticket test set, the six queues, and a scorer for any mlx-lm model."""
import collections, glob, json, os, re
from huggingface_hub import hf_hub_download
from mlx_lm import generate
from tqdm.auto import tqdm

DATASET = "cngchis/Support-Ticket-Router-12K-Cleaned"
QUEUES = ["api", "billing", "cancellation", "complaint", "technical", "upgrade"]
INSTRUCTION = ("Classify the customer support message into one of the following intents: "
               "api, billing, cancellation, complaint, technical, upgrade.\n\nMessage: ")


def load_tickets(split="test"):
    """The dataset's own split: a list of {"text", "label"}."""
    path = hf_hub_download(DATASET, f"processed/{split}.jsonl", repo_type="dataset")
    return [{"text": r["text"], "label": r["label"]} for r in map(json.loads, open(path))]


def latest_studio_run(base="unsloth_gemma-3-270m-it"):
    """The newest training run Unsloth Studio saved for this model (an mlx-lm LoRA adapter folder)."""
    runs = glob.glob(os.path.expanduser(f"~/.unsloth/studio/outputs/{base}_*"))
    return max(runs, key=os.path.getmtime)


def ask(model, tokenizer, text, instruction=True):
    """One ticket in, the model's answer out. instruction=False sends the bare message, as Studio trained it."""
    content = (INSTRUCTION + text) if instruction else text
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt, max_tokens=12, verbose=False).strip()


def to_queue(answer):
    """The first queue name in the answer, or None."""
    return next((w for w in re.findall(r"[a-z]+", answer.lower()) if w in QUEUES), None)


def score(model, tokenizer, tickets, instruction=True):
    """Accuracy on every ticket + what the model answered for each true queue."""
    right, answered = 0, collections.defaultdict(collections.Counter)
    for t in tqdm(tickets, desc="tickets"):
        q = to_queue(ask(model, tokenizer, t["text"], instruction))
        right += q == t["label"]
        answered[t["label"]][q or "(no queue)"] += 1
    print(f"accuracy: {right}/{len(tickets)} = {100 * right / len(tickets):.1f} %")
    for label in QUEUES:
        print(f"  {label:<13}", dict(answered[label].most_common()))
    return right / len(tickets)
