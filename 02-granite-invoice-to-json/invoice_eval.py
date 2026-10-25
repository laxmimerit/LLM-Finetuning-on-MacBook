"""Test set + prompt + field-by-field scoring for invoice → JSON.

Test set: the 100 digital PDFs (text layer) of drew-ipp/invoice-extraction-benchmark (CC BY 4.0) in the categories
clean, currency, tax, credit, multipage, freight (its 13 "hard" PDFs are left out). Scoring follows the benchmark's rules for these 8 header fields:
amounts exact to the cent, dates as ISO dates, vendor names after normalising legal forms, invoice numbers after
normalising case and labels.
"""
import json
import re

import pypdf
from huggingface_hub import snapshot_download

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency", "subtotal", "tax", "total", "document_type"]
PROMPT = ("Extract this invoice as JSON with exactly these keys: vendor_name, invoice_number, "
          "invoice_date (YYYY-MM-DD), currency (ISO code), subtotal, tax, total (numbers), "
          "document_type (invoice or credit_note).\n\n")
CATEGORIES = ("clean", "currency", "tax", "credit", "multipage", "freight")


def load_test():
    root = snapshot_download("drew-ipp/invoice-extraction-benchmark", repo_type="dataset",
                             allow_patterns=["corpus/v1/metadata.jsonl", "corpus/v1/documents/*.pdf"]) + "/corpus/v1/"
    rows = [json.loads(l) for l in open(root + "metadata.jsonl")]
    test = []
    for r in rows:
        if r["category"] in CATEGORIES and r["file_name"].endswith(".pdf"):
            text = "\n".join(p.extract_text() for p in pypdf.PdfReader(root + r["file_name"]).pages)
            test.append({"id": r["doc_id"], "category": r["category"], "text": text,
                         "json": {k: r["expected"][k] for k in FIELDS}})
    return test


def messages(text, answer=None):
    m = [{"role": "user", "content": PROMPT + text}]
    if answer is not None:
        m.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return m


def parse(reply):
    """The first {...} in the reply, or None."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


LEGAL = r"\b(pty|ltd|limited|inc|llc|corp|corporation|co|company|plc|gmbh|kg|ag|bv|nv|sarl|sas|sa|cc|llp|pvt|private)\b"


def norm_vendor(s):
    s = re.sub(r"[^a-z0-9& ]", " ", str(s).lower().replace("b.v.", "bv").replace("n.v.", "nv"))
    return " ".join(re.sub(LEGAL, " ", s).split())


def norm_number(s):
    s = re.sub(r"^((TAX )?INVOICE|CREDIT NOTE)\s*(NO\b\.?|NUMBER|#)?\s*:?\s*", "", str(s).upper().strip())
    return re.sub(r"[\s#:]", "", s)


def to_num(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(re.sub(r"[^\d.\-]", "", str(x)))
    except ValueError:
        return None


def field_ok(k, got, want):
    if got is None:
        return want is None
    if k in ("subtotal", "tax", "total"):
        g = to_num(got)
        return g is not None and want is not None and abs(abs(g) - want) < 0.005
    if k == "vendor_name":
        return norm_vendor(got) == norm_vendor(want)
    if k == "invoice_number":
        return norm_number(got) == norm_number(want)
    return str(got).strip().upper() == str(want).strip().upper()


def score(pred, want):
    """Fields right (0–8) for one document; pred may be None (no JSON)."""
    if pred is None:
        return 0
    return sum(field_ok(k, pred.get(k), want[k]) for k in FIELDS)
