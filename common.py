"""Shared helpers: data splits, metrics and plots, so both models are evaluated identically."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

DATA = Path("data/defects.csv")
REPORTS = Path("reports")
SEED = 42


def load_splits():
    """70 / 15 / 15 stratified split. Same seed -> same split for every model."""
    df = pd.read_csv(DATA, dtype={"ata": str})
    train, temp = train_test_split(df, test_size=0.30, stratify=df["ata"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["ata"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(name, y_true, y_pred, labels, train_time_s, infer_ms_per_item):
    REPORTS.mkdir(exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
    print(f"\n=== {name} ===\nAccuracy: {acc:.3f}   Macro-F1: {f1:.3f}\n{report}")

    (REPORTS / f"{name}_classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=[f"ATA {l}" for l in labels], rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels=[f"ATA {l}" for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {name}  (acc {acc:.3f}, macro-F1 {f1:.3f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "train_time_s": round(train_time_s, 1), "infer_ms_per_item": round(infer_ms_per_item, 3)}
    (REPORTS / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_all_metrics():
    rows = [json.loads(p.read_text()) for p in sorted(REPORTS.glob("*_metrics.json"))]
    return pd.DataFrame(rows)
