"""
predict.py
----------
Loads the trained models and exposes one interface for the app:
    model.predict_proba(list_of_texts) -> (n, n_classes) array, model.classes

Also implements word-importance explanations by OCCLUSION (leave-one-word-out):
each word is removed in turn and the drop in confidence for the predicted
chapter is measured. Model-agnostic, so it works for both models.
"""
from pathlib import Path

import joblib
import numpy as np

from generate_data import ATA_NAMES

BASELINE_PATH = Path("models/baseline.joblib")
BERT_PATH = Path("models/distilbert")


class BaselineModel:
    name = "TF-IDF + SVM"

    def __init__(self):
        self.pipe = joblib.load(BASELINE_PATH)
        self.classes = list(self.pipe.classes_)

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))


class BertModel:
    name = "DistilBERT"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(BERT_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH).to(self.device).eval()
        id2label = self.model.config.id2label
        self.classes = [id2label[i] for i in range(len(id2label))]

    def predict_proba(self, texts, batch_size=64):
        out = []
        texts = list(texts)
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=64,
                               padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, -1).cpu().numpy())
        return np.vstack(out)


def available_models():
    models = {}
    if BASELINE_PATH.exists():
        models["TF-IDF + SVM"] = BaselineModel
    if BERT_PATH.exists() and (BERT_PATH / "config.json").exists():
        models["DistilBERT"] = BertModel
    return models


def top_k(model, text, k=3):
    p = model.predict_proba([text])[0]
    idx = np.argsort(p)[::-1][:k]
    return [{"ata": model.classes[i], "name": ATA_NAMES.get(model.classes[i], "?"),
             "prob": float(p[i])} for i in idx]


def explain(model, text):
    """Leave-one-word-out importance for the predicted class.
    Returns (predicted_ata, [(word, importance), ...]) where importance > 0
    means the word pushed TOWARDS the predicted chapter."""
    words = text.split()
    base = model.predict_proba([text])[0]
    cls = int(np.argmax(base))
    if len(words) < 2:
        return model.classes[cls], [(w, 0.0) for w in words]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = model.predict_proba(variants)[:, cls]
    return model.classes[cls], [(w, float(base[cls] - p)) for w, p in zip(words, probs)]


def explanation_html(word_scores):
    """Green = pushed towards the prediction, red = pushed away."""
    import html
    m = max((abs(s) for _, s in word_scores), default=0) or 1.0
    spans = []
    for w, s in word_scores:
        a = min(abs(s) / m, 1.0) * 0.85
        rgb = "46,160,67" if s > 0 else "218,54,51"
        spans.append(f'<span title="{s:+.3f}" style="background:rgba({rgb},{a:.2f});'
                     f'padding:2px 4px;border-radius:4px;margin:1px;line-height:2">{html.escape(w)}</span>')
    return '<div style="font-size:1.05em">' + " ".join(spans) + "</div>"


if __name__ == "__main__":
    m = BaselineModel()
    t = "#2 EGT exceeded limits during climb, EGT thermocouple replaced IAW AMM"
    print(top_k(m, t))
    print(explain(m, t))
    print(top_k(m, "valve leaking in wheel well"))
