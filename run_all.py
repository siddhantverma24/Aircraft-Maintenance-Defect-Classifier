"""
run_all.py - one command to rebuild everything on any laptop.
    python run_all.py            (data + baseline + DistilBERT)
    python run_all.py --skip-bert
"""
import subprocess
import sys

steps = [
    ["generate_data.py"],
    ["generate_data.py", "--n", "40", "--seed", "999", "--out", "data/sample_batch.csv"],
    ["train_baseline.py"],
]
if "--skip-bert" not in sys.argv:
    steps.append(["train_bert.py"])

for s in steps:
    print(f"\n>>> python {' '.join(s)}", flush=True)
    subprocess.run([sys.executable, *s], check=True)

print("\nAll done. Start the demo with:  streamlit run app.py")
