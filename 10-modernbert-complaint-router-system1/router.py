"""The 9 routes, a timed accuracy check on the test complaints, and the before/after scoreboard."""
import re
import subprocess
import time

# the 9 financial products a complaint can be about (the same routes as lesson 3)
ROUTES = ["credit_reporting", "debt_collection", "credit_card", "bank_account", "mortgage",
          "money_transfer", "vehicle_loan", "student_loan", "personal_loan"]

# lesson 3 on the same 826 test complaints and the same Mac
LAYA = 0.546            # Laya (ModernBERT-large + a decision head), zero-shot, KGP Talkie's Laya benchmark
MINI_CLEF = 0.731       # mini-clef (Qwen 3.5 0.8B + joint schema head), lesson 3
MINI_CLEF_MS = 346      # its milliseconds per complaint, 16 at a time


def score(router, rows, batch_size=16):
    """Accuracy of a text-classification pipeline on rows, and its milliseconds per complaint."""
    texts = [r["complaint"] for r in rows]
    router(texts[:16], batch_size=batch_size)                      # warm-up: the first call builds the GPU graphs
    start = time.time()
    out = router(texts, batch_size=batch_size, truncation=True, max_length=256)
    ms = (time.time() - start) / len(rows) * 1000
    preds = [o[0]["label"] if isinstance(o, list) else o["label"] for o in out]
    return sum(p == r["product"] for p, r in zip(preds, rows)) / len(rows), ms


def bar(x, width=30):
    full = round(width * x)
    return "█" * full + "░" * (width - full)


def scoreboard(before, after, ms, train_secs, gpu_gb, steps, batch, n_test=826):
    hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
    info = dict(re.findall(r"^\s*(Model Name|Chip|Memory): (.+)$", hw, re.M))
    print(f"Machine      {info['Model Name']} · {info['Chip']} · {info['Memory']} unified memory")
    print(f"GPU          {info['Chip']} through Metal (PyTorch MPS) · no NVIDIA GPU, no CUDA")
    print()
    print(f"Training     full fine-tune · {steps} steps × {batch} complaints")
    print(f"Train time   {int(train_secs // 60)} min {round(train_secs % 60)} s")
    print(f"GPU memory   {gpu_gb:.1f} GB")
    print()
    print(f"Test: {n_test} complaints, 9 products, right product" + " " * 24 + "ms per complaint")
    rows = [("ModernBERT-base, untrained head", before, ""),
            ("Laya (ModernBERT-large + head)", LAYA, ""),
            ("mini-clef (Qwen 3.5 0.8B), lesson 3", MINI_CLEF, f"{MINI_CLEF_MS:.0f}"),
            ("ModernBERT-base, fine-tuned", after, f"{ms:.1f}")]
    for name, acc, speed in rows:
        print(f"  {name:<36} {bar(acc)}  {acc * 100:5.1f} %  {speed:>7}")
