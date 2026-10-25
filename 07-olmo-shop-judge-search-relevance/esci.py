"""Lesson 7 helper: Amazon's Shopping Queries (ESCI) data, the text the model reads, BM25 and the two scores.

ESCI labels one (search query, product) pair as Exact, Substitute, Complement or Irrelevant.
Data: tasksource/esci on Hugging Face (Apache-2.0), a copy of amazon-science/esci-data (Reddy et al. 2022).
"""
import glob
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

LABELS = ["Exact", "Substitute", "Complement", "Irrelevant"]          # class ids 0..3
GAIN = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}   # how much each label is worth in a ranking
COLS = ["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "small_version",
        "product_title", "product_bullet_point", "product_brand", "product_color"]


def _read(split):
    files = sorted(glob.glob(f"data/esci/data/{split}-*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=COLS, filters=[("product_locale", "==", "us"), ("small_version", "==", 1)])
                   for f in files], ignore_index=True)
    # tasksource joined the product table on product_id only, so some US pairs appear 2-3 times, once with a Spanish or
    # Japanese product text. Keep only pairs whose example_id appears once (this gives exactly Amazon's counts).
    return d[~d.example_id.duplicated(keep=False)].copy()


def _norm(q):
    return re.sub(r"[^a-z0-9]+", " ", str(q).lower()).strip()


def load_esci(n_train=1200, n_test=250, seed=0):
    """US shop, the "hard" query set. Splits by query: a test query and its products are never seen in training."""
    snapshot_download("tasksource/esci", repo_type="dataset", local_dir="data/esci")      # 2.5 GB, once
    tr, te = _read("train"), _read("test")
    print(f"US hard set: {tr.query_id.nunique():,} train queries ({len(tr):,} pairs), "
          f"{te.query_id.nunique():,} test queries ({len(te):,} pairs)")
    rng = np.random.default_rng(seed)
    test = te[te.query_id.isin(rng.choice(np.array(sorted(te.query_id.unique())), n_test, replace=False))]
    # train only on queries whose text never appears in the test split (a few queries are typed the same in both)
    tr_q, te_q = tr["query"].map(_norm), set(te["query"].map(_norm))
    pick = rng.choice(np.array(sorted(tr[~tr_q.isin(te_q)].query_id.unique())), n_train + 100, replace=False)
    val = tr[tr.query_id.isin(pick[:100])]
    train = tr[tr.query_id.isin(pick[100:])].sample(frac=1, random_state=0)
    return (train.reset_index(drop=True), val.sort_values(["query_id", "example_id"]).reset_index(drop=True),
            test.sort_values(["query_id", "example_id"]).reset_index(drop=True))


def _clean(x, n=None):
    s = "" if x is None or str(x) in ("nan", "None") else str(x)
    s = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()
    return s[:n] if n else s


def pair_text(query, title, brand="", color="", details=""):
    """What the model reads: the query, then the product (details cut to 120 characters)."""
    lines = [f"Query: {_clean(query)}", f"Product: {_clean(title)}"]
    if _clean(brand):
        lines.append(f"Brand: {_clean(brand)}")
    if _clean(color):
        lines.append(f"Color: {_clean(color)}")
    if _clean(details, 120):
        lines.append(f"Details: {_clean(details, 120)}")
    return "\n".join(lines)


def row_text(r):
    return pair_text(r["query"], r["product_title"], r["product_brand"], r["product_color"], r["product_bullet_point"])


JUDGE_SYSTEM = (
    "You judge search results for an online shop. You get a shopper's search query and one product. "
    "Answer with exactly one word:\n"
    "Exact - the product is what the query asks for and matches all its details.\n"
    "Substitute - the product is somewhat relevant but misses some detail of the query; it could be bought instead.\n"
    "Complement - the product does not match the query but goes together with what the query asks for (for example an accessory).\n"
    "Irrelevant - the product is not relevant to the query."
)


def judge_messages(text):
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": text + "\n\nAnswer with one word: Exact, Substitute, Complement or Irrelevant."}]


def relevance(probs):
    """One 0-1 score from the four probabilities: Exact counts 1, Substitute 0.1, Complement 0.01, Irrelevant 0."""
    return float(sum(p * GAIN[l] for p, l in zip(probs, LABELS)))


def ndcg10(scores, labels):
    """How good the top-10 order is: 1.0 = the best products are on top, in the best order."""
    disc = 1 / np.log2(np.arange(2, 12))
    got = [GAIN[l] for _, l in sorted(zip(scores, labels), key=lambda x: -x[0])][:10]   # stable: ties keep list order
    best = sorted((GAIN[l] for l in labels), reverse=True)[:10]
    ideal = (np.array(best) * disc[:len(best)]).sum()
    return None if ideal == 0 else float((np.array(got) * disc[:len(got)]).sum() / ideal)


def evaluate(df, preds=None, scores=None):
    """Accuracy of the labels (if given) and NDCG@10 averaged over the queries (if scores are given)."""
    out = {}
    if preds is not None:
        out["accuracy"] = round(float(np.mean([p == g for p, g in zip(preds, df.esci_label)])), 3)
    if scores is not None:
        d = df.assign(s=list(scores))
        per_q = [ndcg10(g.s.tolist(), g.esci_label.tolist()) for _, g in d.groupby("query_id", sort=False)]
        out["ndcg@10"] = round(float(np.mean([v for v in per_q if v is not None])), 3)
    return out


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25(df, k1=1.5, b=0.75, eps=0.25):
    """Plain keyword search (BM25, as in the rank-bm25 package): how well each product's words match its query."""
    docs = [_words(row_text(r).split("\n", 1)[1]) for _, r in df.iterrows()]       # the product lines only
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    df_count = Counter(w for d in docs for w in set(d))
    idf = {w: math.log(n - c + 0.5) - math.log(c + 0.5) for w, c in df_count.items()}
    floor = eps * sum(idf.values()) / len(idf)
    idf = {w: (v if v >= 0 else floor) for w, v in idf.items()}
    out = []
    for q, d in zip(df["query"], docs):
        tf = Counter(d)
        out.append(sum(idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl)) for w in _words(q)))
    return out


def show_ranking(df, scores, k=5):
    """Top k products of one query by score, with the label Amazon's annotators gave."""
    d = df.assign(s=list(scores)).sort_values("s", ascending=False, kind="stable")
    print(f"NDCG@10 {ndcg10(list(scores), df.esci_label.tolist()):.2f}")
    for _, r in d.head(k).iterrows():
        print(f"  {r.s:.2f}  {r.esci_label:<11} {r.product_title[:70]}")


def scoreboard(base, tuned, bm25_ndcg, train_secs, gpu_gb, steps, batch):
    print("Search-relevance judge · OLMo 2 1B Instruct · MacBook Pro M5 Max 36 GB")
    print(f"Training: {steps} steps x {batch} pairs, {train_secs / 60:.1f} min, GPU memory {gpu_gb:.1f} GB")
    print(f"{'':22}{'accuracy':>10}{'NDCG@10':>10}")
    print(f"{'keyword search (BM25)':22}{'-':>10}{bm25_ndcg:>10.3f}")
    print(f"{'base model':22}{base['accuracy']:>10.1%}{base['ndcg@10']:>10.3f}")
    print(f"{'fine-tuned':22}{tuned['accuracy']:>10.1%}{tuned['ndcg@10']:>10.3f}")
