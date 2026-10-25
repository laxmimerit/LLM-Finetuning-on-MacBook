"""Lesson 8 helpers: the 17 budget categories and the prompt, merchant groups for the "never-seen shops" test,
scoring, the monthly budget table, and the one small fix llama.cpp needs for Apertus.

statement.csv is a month of bank lines that we wrote ourselves: every merchant in it is made up.
"""
import gc
import json
import random
import re
import statistics
import time
import urllib.request
from collections import Counter

import mlx.core as mx
import pandas as pd
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from tqdm.auto import tqdm

BASE = "swiss-ai/Apertus-v1.1-1.5B-Instruct"                # the model we fine-tune (Apache-2.0)
DATASET = "DoDataThings/us-bank-transaction-categories-v2"  # MIT, 68,000 synthetic bank lines

CATS = ["Restaurants", "Groceries", "Shopping", "Transportation", "Entertainment", "Utilities", "Subscription",
        "Healthcare", "Insurance", "Mortgage", "Rent", "Travel", "Education", "Personal Care", "Transfer", "Income", "Fees"]

# The same system prompt for the model before and after fine-tuning.
SYSTEM = ("You sort lines from a US bank statement into budget categories. Each line starts with [debit] (money out) "
          "or [credit] (money in). Reply with exactly one category from this list and nothing else: "
          + ", ".join(CATS) + ".")


def messages(line, category=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": line}]
    if category is not None:
        m.append({"role": "assistant", "content": category})
    return m


def prompt(tok, line):
    return tok.apply_chat_template(messages(line), add_generation_prompt=True, tokenize=False)


_LOWER = {c.lower(): c for c in CATS}


def parse(answer):
    """The first category name in the answer (any case); no category -> 'other', which counts as wrong."""
    s = answer.strip().lower()
    for c in sorted(_LOWER, key=len, reverse=True):
        if s.rstrip(". ") == c:
            return _LOWER[c]
    best = None
    for c in _LOWER:
        m = re.search(r"\b" + re.escape(c) + r"\b", s)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), _LOWER[c])
    return best[1] if best else "other"


# ---------------------------------------------------------------- merchants: strip the bank's wrapper, keep the shop
_PREFIXES = [r"preapproved payment bill user payment:", r"preauthorized withdrawal to", r"subscription payment:",
             r"payment to:", r"buyer payment:", r"google \*", r"pypl \*", r"preapproved payment:", r"express checkout payment:",
             r"withdrawal from", r"preauthorized deposit from", r"paypal inst xfer", r"paypal \*", r"pypl\*", r"pp\*",
             r"sq \*", r"tst\*", r"clv\*", r"orig co name:", r"pos debit", r"pos purchase", r"debit card purchase",
             r"purchase authorized on \d\d/\d\d", r"recurring payment", r"ach debit", r"ach credit"]
_PRE = re.compile(r"^(?:" + "|".join(_PREFIXES) + r")\s*")


def merchant(line, nwords=2):
    """'[debit] SQ *BLUE BOTTLE 0423 SAN FRANCISCO CA' -> 'blue bottle': no sign, no payment wrapper
    (SQ *, TST*, PYPL *, ...), no order ids, store numbers or addresses, the first two words."""
    s = re.sub(r"^\[(debit|credit)\]\s*", "", line.lower())
    for _ in range(3):
        s2 = _PRE.sub("", s).strip()
        if s2 == s:
            break
        s = s2
    s = re.split(r"[;*#]|\bppd id\b|\bweb id\b|\bentry descr\b", s)[0]
    m = re.match(r"\s*[\d-]+[a-z]*\s*[a-z&']*", s)          # shops that start with digits: 7-eleven, 99 ranch
    head = m.group(0) if m and re.match(r"\s*\d+-?[a-z]|\s*\d+ [a-z]{3,}", s) else ""
    s = head + re.sub(r"\d.*$", "", s[len(head):])          # store numbers, addresses, dates start at a digit
    words = [w.strip(".'") for w in re.findall(r"[a-z&'.]+", s) if w.strip(".'")]
    return " ".join(words[:nwords]) or "_empty"


def _squash(line, n):
    return re.sub(r"[^a-z]", "", merchant(line, nwords=4))[:n]


def _first_word(line):
    return merchant(line, nwords=1).split(".")[0].replace("'", "").replace("&", "")


def merchant_groups(lines):
    """One group per shop: two lines are the same shop if they share the first word of the merchant, or its first
    6 letters without spaces (WHOLEFDS and WHOLE FOODS, CHEVRON and CHEVRON NASHVILLE). This merges a bit too much
    on purpose: a held-out shop must really be new."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    firsts = []
    for line in lines:
        a, b = "w:" + _first_word(line), "s:" + _squash(line, 6)
        firsts.append(a)
        if not b.endswith(":"):
            parent[find(a)] = find(b)
    return [find(a) for a in firsts]


def hold_out(d, share=0.15, cap=0.30, seed=0):
    """Pick whole shops for the never-seen test: per category, random shop groups until about 15 % of its lines
    (never one giant group of more than 30 %). Returns the set of held-out groups."""
    rng = random.Random(seed)
    majority = d.groupby("group").category.agg(lambda s: s.value_counts().index[0])
    size = d.groupby("group").size()
    held = set()
    for c in CATS:
        groups = list(majority[majority == c].index)
        rng.shuffle(groups)
        total, got = size[groups].sum(), 0
        for g in groups:
            if got >= share * total:
                break
            if size[g] > cap * total:
                continue
            held.add(g)
            got += size[g]
    return held


def take(df, n, seed):
    """n lines per category."""
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("category")])


# ---------------------------------------------------------------- run a model on bank lines
_greedy = make_sampler(temp=0.0)


def classify(model, tok, lines, progress=True):
    """One category per line (greedy, up to 8 new tokens). Returns (categories, milliseconds per line)."""
    for _ in range(3):  # warm up the GPU first, so the timings are fair
        generate(model, tok, prompt(tok, "[debit] warm up"), max_tokens=4, sampler=_greedy, verbose=False)
    preds, ms = [], []
    for line in tqdm(list(lines), disable=not progress, leave=False):
        t0 = time.perf_counter()
        out = generate(model, tok, prompt(tok, line), max_tokens=8, sampler=_greedy, verbose=False)
        ms.append((time.perf_counter() - t0) * 1000)
        preds.append(parse(out))
    return preds, ms


def evaluate(model, tok, test, name):
    """Accuracy on a test set, the model's most common answer, and the median time per line."""
    preds, ms = classify(model, tok, test.description)
    out = test.assign(pred=preds, ms=ms)
    acc = 100 * (out.pred == out.category).mean()
    top, k = Counter(preds).most_common(1)[0]
    print(f"{name}: {acc:.1f} % right ({int((out.pred == out.category).sum()):,} / {len(out):,})  |  "
          f"most common answer: {top} ({k:,} times)  |  {statistics.median(ms):.0f} ms per line")
    return out


def free(*_):
    """Give the GPU memory back (call after `del model`)."""
    gc.collect()
    mx.clear_cache()


def train_usage(path="train_usage.txt"):
    """Wall time and peak memory from `/usr/bin/time -l`."""
    txt = open(path).read()
    secs = float(re.search(r"([\d.]+) real", txt).group(1))
    peak = int(re.search(r"(\d+)\s+peak memory footprint", txt).group(1)) / 1e9
    print(f"training: {int(secs // 60)} min {secs % 60:.0f} s, peak memory {peak:.1f} GB")
    return secs, peak


# ---------------------------------------------------------------- the payoff: a month of lines -> a budget
def load_statement(path="statement.csv"):
    """31 lines for September 2026. We wrote them; every merchant is made up."""
    return pd.read_csv(path)


def budget(st, answers):
    """Print each line with every model's category, then the monthly budget per category.
    `answers` = {"base": [...], "tuned": [...]}; the last column is what a person would file it under."""
    names = list(answers)
    for n in names:
        right = sum(p == t for p, t in zip(answers[n], st.category))
        print(f"{n}: {right} / {len(st)} lines right")
    print()
    print(f"{'amount':>9}  " + "".join(f"{n:<15}" for n in names) + "line")
    for i, (line, amt) in enumerate(zip(st.description, st.amount)):
        print(f"{amt:>9.2f}  " + "".join(f"{answers[n][i]:<15}" for n in names) + line)
    print()
    print("Budget, September 2026")
    print(f"  {'category':<15}" + "".join(f"{n:>11}" for n in names) + f"{'true':>11}")
    true = st.groupby("category").amount.sum()
    sums = {n: st.assign(p=answers[n]).groupby("p").amount.sum() for n in names}
    for c in sorted(set(true.index).union(*[s.index for s in sums.values()]), key=lambda c: true.get(c, 0)):
        print(f"  {c:<15}" + "".join(f"{sums[n].get(c, 0):>11.2f}" for n in names) + f"{true.get(c, 0):>11.2f}")
    print(f"  {'money in':<15}{st[st.amount > 0].amount.sum():>11.2f}")
    print(f"  {'money out':<15}{st[st.amount < 0].amount.sum():>11.2f}")


# ---------------------------------------------------------------- Ollama
def fix_for_llama_cpp(src, dst):
    """Copy the fused model for llama.cpp's GGUF converter. mlx-lm stores Apertus' xIELU activation values
    alpha_p / alpha_n as plain numbers; the converter wants them as 1-element arrays, as on Hugging Face."""
    import shutil
    shutil.copytree(src, dst, dirs_exist_ok=True)
    w = mx.load(f"{src}/model.safetensors")
    w = {k: (v.reshape(1) if k.endswith(("alpha_p", "alpha_n")) else v) for k, v in w.items()}
    mx.save_safetensors(f"{dst}/model.safetensors", w, metadata={"format": "pt"})
    print("ready for llama.cpp:", dst)


def modelfile(gguf):
    """Ollama's Modelfile: the GGUF, Apertus' chat layout as we trained it, our system prompt, greedy answers."""
    return (f'FROM ./{gguf}\n'
            'TEMPLATE """<SPECIAL_61>{{ .System }}<SPECIAL_62><SPECIAL_63>Deliberation: disabled\n'
            'Tool Capabilities: disabled<SPECIAL_64><SPECIAL_65>{{ .Prompt }}<SPECIAL_66><SPECIAL_67>"""\n'
            f'SYSTEM """{SYSTEM}"""\n'
            'PARAMETER temperature 0\n'
            'PARAMETER num_predict 8\n'
            'PARAMETER stop "<SPECIAL_68>"\n')


def ask_ollama(model, line):
    body = json.dumps({"model": model, "prompt": line, "stream": False,
                       "options": {"temperature": 0, "num_predict": 8}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["response"]


def evaluate_ollama(model, test, name):
    ask_ollama(model, "[debit] warm up")
    preds, ms = [], []
    for line in tqdm(list(test.description), leave=False):
        t0 = time.perf_counter()
        preds.append(parse(ask_ollama(model, line)))
        ms.append((time.perf_counter() - t0) * 1000)
    out = test.assign(pred=preds, ms=ms)
    acc = 100 * (out.pred == out.category).mean()
    print(f"{name} in Ollama: {acc:.1f} % right ({int((out.pred == out.category).sum()):,} / {len(out):,})  |  "
          f"{statistics.median(ms):.0f} ms per line")
    return out
