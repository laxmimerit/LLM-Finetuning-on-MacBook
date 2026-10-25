"""Lesson 9: teach Qwen 3.5 0.8B to reason with GRPO, a small reinforcement-learning loop on mlx-lm.

The reward only checks the final number: 1 if the reply's "Answer:" line holds the right number, else 0.
Nothing here ever tells the model to write steps.

Train:  python grpo.py --steps 4 --lr 2e-5 --out runs/grpo
"""
import argparse
import json
import random
import re
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pandas as pd
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_map
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
ASK = "\n\nEnd your reply with one line: Answer: <the final number>"
PAD = 248044                                   # Qwen's <|endoftext|>, pads the batch in the loss


# ---------------------------------------------------------------- data, prompt, reward
def load_gsm8k(split):
    """GSM8K (openai/gsm8k, MIT): a word problem and a worked solution that ends in '#### <number>'."""
    path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(path)
    d["gold"] = d.answer.str.split("####").str[-1].str.strip().str.replace(",", "")
    return d


def chat(tok, question, ask=True):
    """The prompt as token ids: one user turn, Qwen's no-think mode."""
    content = question + (ASK if ask else "")
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                  enable_thinking=False)
    return list(ids) if isinstance(ids, list) else list(ids["input_ids"])


NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def number(s):
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except (AttributeError, ValueError):
        return None


def answer_line(text):
    """The number on the last 'Answer:' line, or None."""
    found = re.findall(r"Answer:\s*\**\s*(" + NUM + ")", text)
    return number(found[-1]) if found else None


def last_number(text):
    """The last number anywhere in the reply (for replies without an Answer line)."""
    found = re.findall(NUM, text)
    return number(found[-1]) if found else None


def reward(text, gold):
    """1 if the Answer line holds the right number, else 0. That's the whole reward."""
    pred = answer_line(text)
    return 1.0 if pred is not None and abs(pred - float(gold)) < 1e-6 else 0.0


# ---------------------------------------------------------------- test: greedy, the same 200 problems every time
def evaluate(model, tok, test, ask=True, name="", max_tokens=1024, batch=100):
    """Answer every test problem once (greedy). ask=False: the question alone, scored by its last number."""
    texts, ntok = [], []
    for b in range(0, len(test), batch):
        res = batch_generate(model, tok, [chat(tok, q, ask) for q in test.question[b:b + batch]],
                             max_tokens=max_tokens, return_token_ids=True,
                             completion_batch_size=batch, prefill_batch_size=batch)
        texts += res.texts
        ntok += [len(t) for t in res.token_ids]
    pick = answer_line if ask else last_number
    out = test[["question", "gold"]].copy()
    out["reply"], out["tokens"], out["pred"] = texts, ntok, [pick(t) for t in texts]
    out["ok"] = [p is not None and abs(p - float(g)) < 1e-6 for p, g in zip(out.pred, out.gold)]
    print(f"{name}: {100 * out.ok.mean():.1f} % right ({out.ok.sum()} / {len(out)}), "
          f"median reply {int(out.tokens.median())} tokens")
    return out


# ---------------------------------------------------------------- GRPO
def train(a):
    random.seed(a.seed)
    mx.random.seed(a.seed)
    model, tok = load(MODEL)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {"rank": a.rank, "scale": a.scale, "dropout": 0.0})
    grad_checkpoint(model.layers[0])
    n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"LoRA rank {a.rank} on every layer: {n / 1e6:.1f} M trainable parameters", flush=True)

    data = load_gsm8k("train")
    order = list(range(len(data)))
    random.shuffle(order)

    def loss_fn(model, x, mask, adv, n_tok):
        logits = model(x[:, :-1]).astype(mx.float32)
        logp = mx.take_along_axis(logits, x[:, 1:, None], axis=-1)[..., 0] - mx.logsumexp(logits, axis=-1)
        return -(adv[:, None] * logp * mask).sum() / n_tok    # push up good tries, push down bad ones

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.0)
    sampler = make_sampler(temp=1.0)
    log = open(f"{a.out}/steps.jsonl", "w")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ts = time.time()
        rows = data.iloc[order[(step - 1) * a.questions: step * a.questions]]
        prompts = [chat(tok, q) for q in rows.question]

        # 1. sample: the model tries every question several times
        model.eval()
        res = batch_generate(model, tok, [p for p in prompts for _ in range(a.tries)], max_tokens=a.max_tokens,
                             sampler=sampler, return_token_ids=True,
                             completion_batch_size=a.questions * a.tries, prefill_batch_size=a.questions * a.tries)

        # 2. score: reward 1 for the right final number, 0 otherwise
        R = [reward(text, rows.gold.iloc[k // a.tries]) for k, text in enumerate(res.texts)]

        # 3. compare each try with its own group: better than the group's average -> positive advantage
        seqs, advs = [], []
        for q in range(a.questions):
            r = R[q * a.tries:(q + 1) * a.tries]
            mean = sum(r) / len(r)
            std = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5
            if std == 0:
                continue                                    # all right or all wrong: nothing to learn
            for j in range(a.tries):
                seqs.append((prompts[q], res.token_ids[q * a.tries + j]))
                advs.append((r[j] - mean) / (std + 1e-4))

        # 4. update: make the better tries more likely (LoRA weights only)
        if seqs:
            model.train()
            n_tok, grads = sum(len(c) for _, c in seqs), None
            for b in range(0, len(seqs), a.micro):
                part = seqs[b:b + a.micro]
                T = max(len(p) + len(c) for p, c in part)
                x = mx.array([p + c + [PAD] * (T - len(p) - len(c)) for p, c in part])
                mask = mx.array([[0] * (len(p) - 1) + [1] * len(c) + [0] * (T - len(p) - len(c)) for p, c in part],
                                dtype=mx.float32)
                _, g = loss_and_grad(model, x, mask, mx.array(advs[b:b + a.micro]), n_tok)
                grads = g if grads is None else tree_map(lambda u, v: u + v, grads, g)
                mx.eval(grads)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)

        lens = [len(t) for t in res.token_ids]
        rec = dict(step=step, right=sum(R) / len(R), reply_tokens=sum(lens) / len(lens), groups=len(seqs) // a.tries,
                   step_s=round(time.time() - ts, 1), total_s=round(time.time() - t0, 1),
                   peak_gb=round(mx.get_peak_memory() / 1e9, 2))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step:2d}/{a.steps} | right {100 * rec['right']:3.0f} % | reply {rec['reply_tokens']:4.0f} tokens"
              f" | learned from {rec['groups']} of {a.questions} questions | {rec['step_s']:5.1f} s", flush=True)
        if a.save_every and step % a.save_every == 0:
            mx.save_safetensors(f"{a.out}/{step:04d}_adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))

    mx.save_safetensors(f"{a.out}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
    json.dump({"model": MODEL, "fine_tune_type": "lora", "num_layers": len(model.layers),
               "lora_parameters": {"rank": a.rank, "scale": a.scale, "dropout": 0.0}},
              open(f"{a.out}/adapter_config.json", "w"), indent=1)
    secs, peak = time.time() - t0, mx.get_peak_memory() / 1e9
    json.dump(dict(vars(a), train_s=round(secs, 1), peak_gb=round(peak, 2)), open(f"{a.out}/run.json", "w"), indent=1)
    print(f"training: {int(secs // 60)} min {secs % 60:.0f} s, peak memory {peak:.1f} GB", flush=True)



# ---------------------------------------------------------------- Ollama
def fix_for_llama_cpp(src, dst):
    """mlx-lm changes two things in Qwen 3.5's weights when it loads them; undo both so llama.cpp's converter can read them."""
    from pathlib import Path
    import shutil
    src, dst = Path(src), Path(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors"))
    for f in src.glob("*.safetensors"):
        w = mx.load(str(f))
        for k, v in w.items():
            if "conv1d.weight" in k:
                w[k] = v.moveaxis(2, 1)                    # mlx keeps the conv kernel's axes in another order
            elif k.endswith(("layernorm.weight", "model.norm.weight", "q_norm.weight", "k_norm.weight")) and v.ndim == 1:
                w[k] = v - 1.0                             # mlx stores these norms as 1 + w
        mx.save_safetensors(str(dst / f.name), w, metadata={"format": "mlx"})


def modelfile(gguf):
    """Ollama's Modelfile: the GGUF, Qwen's no-think chat layout, greedy answers."""
    return (f"FROM ./{gguf}\n"
            'TEMPLATE """<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"""\n'
            "PARAMETER temperature 0\n"
            "PARAMETER num_predict 1024\n"
            'PARAMETER stop "<|im_end|>"\n')


def ask_ollama(model, question):
    import urllib.request
    body = json.dumps({"model": model, "prompt": question + ASK, "stream": False, "think": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["response"]


def evaluate_ollama(model, test, name):
    """The same 200 problems, one at a time through Ollama."""
    t0 = time.time()
    ok = [reward(ask_ollama(model, q), g) for q, g in zip(test.question, test.gold)]
    print(f"{name} in Ollama: {100 * sum(ok) / len(ok):.1f} % right ({int(sum(ok))} / {len(ok)}), "
          f"{(time.time() - t0) / len(ok):.1f} s per problem")
    return ok

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--questions", type=int, default=8)   # questions per step
    ap.add_argument("--tries", type=int, default=8)       # answers sampled per question (the group)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--micro", type=int, default=4)       # sequences per backward pass (memory)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train(a)
