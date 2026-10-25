"""Before vs after fine-tuning: the machine, the training cost, and both models on the same 100 test invoices."""
import re
import subprocess


def bar(n, total, width=30):
    full = round(width * n / total)
    return "█" * full + "░" * (width - full)


def show(base, tuned, train_secs, gpu_gb, steps, examples, docs=100):
    hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
    info = dict(re.findall(r"^\s*(Model Name|Chip|Memory): (.+)$", hw, re.M))
    (b_docs, b_fields), (t_docs, t_fields) = base, tuned
    print(f"Machine       {info['Model Name']} · {info['Chip']} · {info['Memory']} unified memory")
    print(f"GPU           {info['Chip']} through Metal (PyTorch MPS) · no NVIDIA GPU, no CUDA")
    print()
    print(f"Fine-tuning   LoRA · {steps} steps · {examples:,} examples")
    print(f"Train time    {int(train_secs // 60)} min {round(train_secs % 60)} s")
    print(f"GPU memory    {gpu_gb:.1f} GB")
    print()
    print(f"Test: {docs} invoice PDFs neither model was trained on, all 8 fields right")
    print(f"  without fine-tuning  {bar(b_docs, docs)}  {b_docs:>3} / {docs}")
    print(f"  with fine-tuning     {bar(t_docs, docs)}  {t_docs:>3} / {docs}")
    print(f"  fields right         {b_fields} → {t_fields} of {docs * 8}")
