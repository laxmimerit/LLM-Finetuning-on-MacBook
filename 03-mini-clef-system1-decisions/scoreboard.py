"""Before vs after: the machine, the training cost, and three models on the same 826 test complaints."""
import re
import subprocess

# Laya (ModernBERT-large + a decision head), scored on the same 826 complaints in KGP Talkie's Laya benchmark
LAYA = 0.546


def bar(x, width=30):
    full = round(width * x)
    return "█" * full + "░" * (width - full)


def show(base, tuned, train_secs, gpu_gb, steps, batch, n_test=826):
    hw = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
    info = dict(re.findall(r"^\s*(Model Name|Chip|Memory): (.+)$", hw, re.M))
    print(f"Machine      {info['Model Name']} · {info['Chip']} · {info['Memory']} unified memory")
    print(f"GPU          {info['Chip']} through Metal (PyTorch MPS) · no NVIDIA GPU, no CUDA")
    print()
    print(f"Training     LoRA + joint schema head · {steps} steps × {batch} = {steps * batch:,} complaints")
    print(f"Train time   {int(train_secs // 60)} min {round(train_secs % 60)} s")
    print(f"GPU memory   {gpu_gb:.1f} GB")
    print()
    print(f"Test: {n_test} complaints, 9 products, right product")
    print(f"  Qwen 3.5 0.8B, chat + JSON      {bar(base)}  {base * 100:5.1f} %")
    print(f"  Laya (ModernBERT-large + head)  {bar(LAYA)}  {LAYA * 100:5.1f} %")
    print(f"  mini-clef (Qwen 3.5 0.8B)       {bar(tuned)}  {tuned * 100:5.1f} %")
