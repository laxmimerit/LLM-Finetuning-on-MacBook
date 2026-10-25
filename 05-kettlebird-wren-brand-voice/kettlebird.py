"""Lesson 5 helpers: Kettlebird (a made-up shop), its style guide, the on-brand checks, the data, and the judge.

Kettlebird is a fictional online shop for kitchen and home goods; its support assistant is called Wren.
Its phone number, links and email use reserved fictional values (555-01xx, the .example domain).
"""
import gc
import json
import random
import re

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx_lm import batch_generate
from tqdm.auto import tqdm

BASE = "openbmb/MiniCPM5-1B-MLX"           # the model we fine-tune (Apache-2.0, official 4-bit MLX build)
TEACHER = "mlx-community/Qwen3.8-27B-4bit"  # writes the on-brand rewrites, and judges blind
BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"   # CDLA-Sharing-1.0
BITEXT_CSV = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

SIGNOFF = "— Wren at Kettlebird"

# The short system prompt every model gets, before and after fine-tuning.
SYSTEM = "You are Wren, the customer support assistant for Kettlebird, an online shop for kitchen and home goods."

# Kettlebird's details. Bitext writes {{Customer Support Phone Number}} and so on; we fill in these instead.
SHOP = {
    "phone": "555-0142",
    "hours": "8am to 8pm",
    "website": "kettlebird.example",
    "account page": "kettlebird.example/account",
    "login page": "kettlebird.example/login",
    "reset page": "kettlebird.example/reset",
    "email": "help@kettlebird.example",
    "stores": "kettlebird.example/stores",
    "plans": "Basic, Plus, Premium",
    "delivery": "standard 3 to 5 business days, express next business day, order by 2pm",
}

STYLE_GUIDE = f"""Kettlebird support voice (style guide)
1. Answer first. The first sentence gives the answer or the first step. No warm-up lines such as
   "I understand", "I appreciate", "Thank you for reaching out", "Certainly!", "Absolutely!", "I'm here to help".
2. Short. At most 80 words before the sign-off. If there are steps, use at most 4 short numbered steps.
   No bold text, no headings, no bullet lists with labels.
3. Plain and warm, like a helpful person at the shop counter. Everyday words, contractions are fine.
   Say "we" for Kettlebird and "you" for the customer.
4. Never use these phrases: "I apologize for any inconvenience", "rest assured", "don't hesitate",
   "valued customer", "I'm here to assist", "please be advised", "at your earliest convenience",
   "it's our pleasure", "seamless", "I hope this helps".
5. If something went wrong for the customer, say sorry once, in plain words ("Sorry about that.").
6. Never invent policies, prices, times, phone numbers or links: use only Kettlebird's details below.
7. At most one exclamation mark.
8. End every reply with this sign-off on its own last line: {SIGNOFF}

Kettlebird's details: """ + "; ".join(f"{k}: {v}" for k, v in SHOP.items()) + "."

# Rule 4 + the warm-up openers of rule 1 (lower-case, matched anywhere in the reply).
BANNED = [
    "i apologize for any inconvenience", "apologize for the inconvenience", "rest assured", "don't hesitate",
    "do not hesitate", "valued customer", "i'm here to assist", "i am here to assist", "please be advised",
    "at your earliest convenience", "it's our pleasure", "it is our pleasure", "seamless", "i hope this helps",
    "i understand", "i appreciate", "thank you for reaching out", "i'm here to help", "i am here to help",
]


# ---------------------------------------------------------------- the on-brand checks (code, no model)

def _norm(s):
    return s.replace("’", "'").lower()


def body_of(answer):
    """The reply without its sign-off line."""
    lines = answer.strip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and "wren at kettlebird" in _norm(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines)


def check(answer):
    """The style guide's rules that code can check. Returns True/False per rule, "all", and the word count."""
    a = answer.strip()
    last = a.splitlines()[-1].strip() if a else ""
    words = len(re.findall(r"\S+", body_of(a)))
    n = _norm(a)
    r = {
        "signoff": _norm(last).lstrip("-—– ").startswith("wren at kettlebird"),
        "no_banned": not any(b in n for b in BANNED),
        "short": words <= 80,
        "no_markdown": "**" not in a and not re.search(r"^#+ ", a, re.M),
        "one_exclaim": a.count("!") <= 1,
    }
    r["all"] = all(r.values())
    r["words"] = words
    return r


def rule_report(answers):
    """How many answers pass each rule."""
    cs = [check(a) for a in answers]
    n = len(cs)
    for k, label in [("signoff", "sign-off last"), ("no_banned", "no banned phrase"), ("short", "80 words or fewer"),
                     ("no_markdown", "no markdown"), ("one_exclaim", "at most one '!'"), ("all", "ALL RULES")]:
        print(f"  {label:<18} {sum(c[k] for c in cs):>4} / {n}")
    print(f"  mean words         {sum(c['words'] for c in cs) / n:>6.0f}")


# ---------------------------------------------------------------- the data

def load_bitext():
    """Bitext's customer-support dataset: 26,872 rows of question, intent, category and a generic answer."""
    import pandas as pd
    return pd.read_csv(hf_hub_download(BITEXT, BITEXT_CSV, repo_type="dataset"))


def test_questions(d, per_intent=4):
    """The fixed test set: 4 questions per intent (27 intents = 108), shuffled with seed 0.
    Duplicates (same text after lower-case + spaces) are dropped first, so no test question is a training question."""
    d = d.assign(key=d.instruction.map(lambda s: re.sub(r"\s+", " ", s.strip().lower()))).drop_duplicates("key")
    d = d.assign(id=d.index)
    test = []
    for intent, g in d.groupby("intent"):
        for r in g.sample(frac=1, random_state=0).head(per_intent).to_dict("records"):
            test.append({"id": int(r["id"]), "intent": intent, "question": r["instruction"].strip(),
                         "bitext": r["response"].strip()})
    return test


def load_pairs(path="kettlebird_pairs.jsonl"):
    """The ready DPO pairs: question, rejected = Bitext's generic answer, chosen = the Kettlebird rewrite."""
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------- filling Bitext's {{placeholders}}

# Kettlebird's details, wherever Bitext writes a placeholder for the shop.
SHOP_FILL = {
    "Customer Support Phone Number": SHOP["phone"],
    "Customer Support Hours": SHOP["hours"], "Customer Assistance Hours": SHOP["hours"],
    "Website URL": SHOP["website"], "Company Name": "Kettlebird", "Company Account": "Kettlebird account",
    "Online Company Portal Info": SHOP["account page"], "Login Page URL": SHOP["login page"],
    "Account Recovery Page URL": SHOP["reset page"], "Account Recovery Page": SHOP["reset page"],
    "Password Reset Page": SHOP["reset page"],
    "Customer Support Email": SHOP["email"], "Customer Service Email": SHOP["email"],
    "Customer Service Email Address": SHOP["email"], "Store Location": "a Kettlebird store",
    "Date Range": "3 to 5", "Standard Shipping Time": "3 to 5 business days",
    "Express Shipping Time": "next business day", "Expedited Shipping Time": "next business day",
    "Shipping Cut-off Time": "2pm", "Same-Day Order Time": "2pm", "Store Pickup Time": "the same day",
    # buttons and pages of the shop's website
    "Settings": "Settings", "Profile": "Profile", "Profile Settings": "Profile", "Profile Type": "Plan",
    "Upgrade Account": "Upgrade", "Account Change": "Change plan", "Account Security": "Security",
    "PIN Settings": "Security", "Forgot Password": "Forgot password", "Forgot PIN": "Forgot password",
    "Forgot Key": "Forgot password", "Forgot Account Key": "Forgot password",
    "Profile PIN Recovery": "Forgot password", "Contact Method": "email", "Order Tracking": "Track order",
    "Track Order": "Track order", "Order Status": "Order status", "Purchase Details": "Order details",
    "Product Category": "kitchen goods", "Currency Symbol": "$",
}
ORDER_BUTTON = {"cancel_order": "Cancel order", "track_order": "Track order"}   # {{Online Order Interaction}}
FIRST = ["Sam", "Priya", "Jordan", "Mei", "Alex", "Maria", "Omar", "Lena"]
LAST = ["Patel", "Garcia", "Chen", "Okafor", "Smith", "Rossi", "Kim", "Novak"]
CITIES = ["Leeds", "Austin", "Toronto", "Dublin"]
COUNTRIES = ["Canada", "Ireland", "Germany", "Australia"]


def customer_details(row_id):
    """Made-up details of the customer who wrote this question (fixed per row)."""
    r = random.Random(row_id)
    first, last = r.choice(FIRST), r.choice(LAST)
    return {
        "Order Number": f"KB-{r.randint(10000, 99999)}", "Invoice Number": f"INV-{r.randint(10000, 99999)}",
        "Tracking Number": f"KB{r.randint(10**7, 10**8 - 1)}", "Rebate Tracking Number": f"RF-{r.randint(1000, 9999)}",
        "Refund Amount": f"{r.randint(12, 140)}.{r.choice(['00', '50', '99'])}",
        "Person Name": f"{first} {last}", "Client Name": f"{first} {last}", "Client First Name": first,
        "Client Last Name": last, "Salutation": r.choice(["Mr.", "Ms."]),
        "Account Type": r.choice(["Basic", "Plus", "Premium"]), "Account Category": r.choice(["Personal", "Business"]),
        "Delivery City": r.choice(CITIES), "Delivery Country": r.choice(COUNTRIES),
    }


PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}?")     # (a few rewrites dropped one closing brace)


def fill(row):
    """Replace every {{placeholder}} in a row's question and answers with a value.
    Customer details the question gives (an order number, a name) get one made-up value, the same in the question
    and in both answers. Details the customer did NOT give are written as words ("your order number"), so no answer
    quotes a number the customer never sent."""
    given = set(PLACEHOLDER.findall(row["question"]))
    if given & {"Person Name", "Client Name", "Client First Name", "Client Last Name"}:
        given |= {"Person Name", "Client Name", "Client First Name", "Client Last Name", "Salutation"}
    cust = customer_details(row["id"])
    plans = {"Account Type", "Account Category", "Delivery City", "Delivery Country"}   # always filled

    def one(text):
        text = re.sub(r"\bour \{\{Store Location\}\}", "our store", text)
        def sub(m):
            name = m.group(1)
            if name in cust and (name in given or name in plans):
                return cust[name]
            if name == "Online Order Interaction":
                return ORDER_BUTTON.get(row["intent"], "Your orders")
            if name in SHOP_FILL:
                return SHOP_FILL[name]
            if name == "Salutation":
                return ""
            if name in ("Person Name", "Client Name", "Client First Name", "Client Last Name"):
                return "the account holder"
            return "\0" + name.lower()          # not given: say it in words (fixed up below)
        t = PLACEHOLDER.sub(sub, text)
        # "order {{Order Number}}" → "your order"; "your {{Order Number}}" → "your order number"
        t = re.sub(r"\b(?:the |your |my )?(order|invoice)( number)? #?\0\1 number", r"your \1", t, flags=re.I)
        t = re.sub(r"\b(your|the|my|a|this) #?\0", r"\1 ", t, flags=re.I)
        t = t.replace("#\0", "your ").replace("\0", "your ")
        return re.sub(r"[ \t]{2,}", " ", t).replace(" .", ".").replace(" ,", ",")
    out = {**row, "question": one(row["question"])}
    for k in ("bitext", "rejected", "chosen"):
        if k in row:
            out[k] = one(row[k])
    return out


# ---------------------------------------------------------------- asking a model

def prompt_for(tokenizer, question, guide=False):
    """The chat prompt for one customer message, in MiniCPM5's non-thinking mode (an empty <think></think>).
    guide=True puts the whole style guide in the system prompt (the "just prompt it" try)."""
    system = SYSTEM + ("\n\n" + STYLE_GUIDE if guide else "")
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                         add_generation_prompt=True, enable_thinking=False, tokenize=False)


def answer_all(model, tokenizer, questions, guide=False, batch=36, max_tokens=300):
    """Answer many customer messages at once (greedy, batched)."""
    prompts = [tokenizer.encode(prompt_for(tokenizer, q, guide), add_special_tokens=False) for q in questions]
    texts = []
    for i in tqdm(range(0, len(prompts), batch), desc="answering", unit="batch"):
        texts += batch_generate(model, tokenizer, prompts[i:i + batch], max_tokens=max_tokens).texts
    return [t.strip() for t in texts]


def free(*_):
    """Give the GPU memory back (call after `del model`)."""
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------- the teacher: rewrite an answer in the brand voice

REWRITE = """You write customer support replies for Kettlebird, an online shop for kitchen and home goods.
Follow this style guide exactly:

{guide}

Customer message:
\"\"\"{q}\"\"\"

A generic reply that has the right content but the wrong voice:
\"\"\"{a}\"\"\"

Rewrite the reply in the Kettlebird voice. Keep the same facts and steps (shorten them; drop filler),
add nothing new. Reply with the rewritten reply only."""


def rewrite(model, tokenizer, rows, batch=8):
    """The teacher rewrites Bitext's generic answers in the Kettlebird voice."""
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": REWRITE.format(guide=STYLE_GUIDE, q=r["question"], a=r["bitext"])}],
        add_generation_prompt=True, enable_thinking=False) for r in rows]
    texts = []
    for i in range(0, len(prompts), batch):
        texts += batch_generate(model, tokenizer, prompts[i:i + batch], max_tokens=220).texts
    return [t.strip() for t in texts]


# ---------------------------------------------------------------- the blind judge

JUDGE = """You are reviewing customer support replies for Kettlebird, an online shop for kitchen and home goods.
The style guide:

{guide}

Customer message:
\"\"\"{q}\"\"\"

Reply to review:
\"\"\"{a}\"\"\"

Score the reply.
- brand: 1 to 5, how well the reply follows the style guide (1 = ignores it, 5 = follows every rule).
- helpful: true if the reply actually addresses the customer's message and does not invent facts, else false.
Answer with JSON only, like {{"brand": 3, "helpful": true}}."""


def judge(model, tokenizer, answers, questions, batch=8):
    """answers = {system name: [answer per question]}. Every answer is judged on its own, without the system's name,
    in one shuffled order (seed 0). Returns {system: [{"brand": 1-5, "helpful": bool}, ...]}."""
    todo = [(s, i) for s in answers for i in range(len(questions))]
    random.Random(0).shuffle(todo)
    out = {s: [None] * len(questions) for s in answers}
    for k in tqdm(range(0, len(todo), batch), desc="judging", unit="batch"):
        chunk = todo[k:k + batch]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": JUDGE.format(guide=STYLE_GUIDE, q=questions[i], a=answers[s][i])}],
            add_generation_prompt=True, enable_thinking=False) for s, i in chunk]
        for (s, i), text in zip(chunk, batch_generate(model, tokenizer, prompts, max_tokens=40).texts):
            b = re.search(r'"brand"\s*:\s*(\d)', text)
            h = re.search(r'"helpful"\s*:\s*(true|false)', text)
            out[s][i] = {"brand": int(b.group(1)) if b else 0, "helpful": bool(h and h.group(1) == "true")}
    return out


def scoreboard(answers, scores):
    """ON BRAND = every rule check passes AND the judge gives 4 or 5."""
    print(f"{'model':<12} {'rules ok':>9} {'judge 1-5':>10} {'helpful':>8} {'words':>6} {'ON BRAND':>9}")
    result = {}
    for s in answers:
        cs, js = [check(a) for a in answers[s]], scores[s]
        n = len(cs)
        on = sum(c["all"] and j["brand"] >= 4 for c, j in zip(cs, js))
        result[s] = 100 * on / n
        print(f"{s:<12} {100 * sum(c['all'] for c in cs) / n:>7.0f} % {sum(j['brand'] for j in js) / n:>10.2f} "
              f"{100 * sum(j['helpful'] for j in js) / n:>6.0f} % {sum(c['words'] for c in cs) / n:>6.0f} "
              f"{result[s]:>7.1f} %")
    return result
