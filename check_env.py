"""
check_env.py - verifies everything the project needs, before you train.
Run inside your virtual environment:   python check_env.py
"""
import importlib
import shutil
import sys

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = []


def report(status, msg, fix=None):
    print(f"{status} {msg}")
    if status == BAD:
        problems.append(fix or msg)


print("=" * 60)
print("ATA classifier - environment check")
print("=" * 60)

# 1. Python
v = sys.version_info
if v >= (3, 9):
    report(OK, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    report(BAD, f"Python {v.major}.{v.minor} is too old", "Install Python 3.10-3.12")

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
report(OK if in_venv else WARN,
       "Running inside a virtual environment" if in_venv
       else "Not in a virtual environment (fine, but venv is recommended)")

# 2. Packages
pkgs = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib", "joblib": "joblib",
        "streamlit": "streamlit", "transformers": "transformers", "torch": "torch"}
for mod, pipname in pkgs.items():
    try:
        m = importlib.import_module(mod)
        report(OK, f"{pipname} {getattr(m, '__version__', '')}")
    except Exception as e:
        report(BAD, f"{pipname} missing ({type(e).__name__})", f"pip install {pipname}"
               if pipname != "torch" else "Install PyTorch (see README)")

# 3. GPU
try:
    import torch
    print(f"       torch built for CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        report(OK, f"GPU detected: {name} (compute {cap[0]}.{cap[1]}, {vram:.1f} GB VRAM)")
        arch = f"sm_{cap[0]}{cap[1]}"
        archs = torch.cuda.get_arch_list()
        if archs and arch not in archs:
            report(BAD, f"This PyTorch build does not support {arch} (supports {', '.join(archs)})",
                   "pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu128")
        else:
            # real test: run a small fp16 matmul on the GPU
            try:
                a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
                (a @ a).sum().item()
                report(OK, "GPU compute test passed (fp16)")
            except Exception as e:
                report(BAD, f"GPU compute test failed: {e}",
                       "Reinstall PyTorch with cu128 and update the NVIDIA driver")
    else:
        report(WARN, "No GPU visible to PyTorch - training will run on CPU (~30-60 min). "
               "If you expected a GPU: reinstall torch with the cu128 command in README, "
               "and update the NVIDIA driver.")
except ImportError:
    pass

# 4. Hugging Face model download (needed once, ~260 MB)
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("distilbert-base-uncased")
    report(OK, "DistilBERT tokenizer downloaded / cached (Hugging Face reachable)")
except ImportError:
    pass
except Exception as e:
    report(BAD, f"Could not download DistilBERT: {type(e).__name__}",
           "Check internet / proxy; try again on another network")

# 5. Disk space
free_gb = shutil.disk_usage(".").free / 1e9
report(OK if free_gb > 3 else BAD, f"Free disk space: {free_gb:.1f} GB",
       "Free up at least 3 GB")

# 6. Project files
from pathlib import Path
missing = [f for f in ["generate_data.py", "train_baseline.py", "train_bert.py",
                       "ner.py", "predict.py", "app.py", "common.py"] if not Path(f).exists()]
if missing:
    report(BAD, f"Missing project files: {missing}", "Run this from inside the project folder")
else:
    report(OK, "All project files present")

print("=" * 60)
if problems:
    print(f"{len(problems)} problem(s) to fix:")
    for p in problems:
        print("  ->", p)
else:
    print("All checks passed. Next:  python run_all.py")
print("=" * 60)
