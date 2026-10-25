"""
train_bert.py
-------------
Fine-tunes DistilBERT (distilbert-base-uncased) for ATA chapter classification.

- Uses the GPU automatically if available (RTX 5050 / GTX 1650), else CPU.
- Mixed precision (fp16) on GPU for speed.
- Batch size picked from available VRAM (4 GB cards get a smaller batch).
- Keeps the checkpoint with the best validation accuracy.

Usage:  python train_bert.py               (default 4 epochs)
        python train_bert.py --epochs 3 --cpu
Output: models/distilbert/ , reports/distilbert_*.{txt,png,json}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import evaluate, load_splits

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = Path("models/distilbert")
MAX_LEN = 64  # write-ups are short


class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length")
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def pick_device(force_cpu):
    if not force_cpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch = 16 if vram_gb < 6 else 32
        print(f"Using GPU: {name} ({vram_gb:.1f} GB VRAM) -> batch size {batch}")
        return dev, batch, True
    print("Using CPU (training will be slower, ~30-60 min)")
    return torch.device("cpu"), 16, False


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    torch.manual_seed(42)
    device, batch_size, amp = pick_device(args.cpu)

    train, val, test = load_splits()
    labels = sorted(train["ata"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id).to(device)

    def loader(df, shuffle):
        ds = TextDS(df["text"], df["ata"].map(label2id), tok)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_dl, val_dl, test_dl = loader(train, True), loader(val, False), loader(test, False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_acc, t_start = -1.0, time.time()
    val_true = val["ata"].map(label2id).values
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {total / step:.4f}", flush=True)

        val_acc = (predict(model, val_dl, device, amp) == val_true).mean()
        print(f"Epoch {epoch}: train loss {total / len(train_dl):.4f}  val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            print(f"  saved best model (val acc {val_acc:.3f})")
    train_time = time.time() - t_start

    # Evaluate the best checkpoint on the held-out test set
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    t0 = time.time()
    test_pred = predict(model, test_dl, device, amp)
    infer_ms = (time.time() - t0) * 1000 / len(test)
    preds = [id2label[i] for i in test_pred]
    evaluate("distilbert", test["ata"].values, preds, labels, train_time, infer_ms)
    (OUT_DIR / "train_info.json").write_text(json.dumps(
        {"device": str(device), "epochs": args.epochs, "batch_size": batch_size,
         "best_val_acc": round(float(best_acc), 4)}, indent=2))


if __name__ == "__main__":
    main()
