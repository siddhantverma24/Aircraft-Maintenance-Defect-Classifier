"""Helpers for the Streamlit UI: chapter icons, result cards, report parsing."""
import html
import json
import re
from pathlib import Path

import pandas as pd

from generate_data import ATA_NAMES

ICONS = {
    "21": "❄️", "24": "⚡", "25": "💺", "26": "🔥", "27": "🕹️", "28": "⛽",
    "29": "💧", "30": "🌧️", "32": "🛞", "33": "💡", "34": "🧭", "36": "🌬️",
    "38": "🚰", "49": "⚙️", "52": "🚪", "72": "✈️",
}

REVIEW_THRESHOLD = 0.6

NAVY = "#0b2545"
SKY = "#1f6feb"


def conf_style(p):
    """(border colour, background, label) for a confidence value."""
    if p >= 0.8:
        return "#1a7f37", "#e9f7ee", "High confidence"
    if p >= REVIEW_THRESHOLD:
        return "#b7791f", "#fff6e0", "Medium confidence"
    return "#cf222e", "#fdecec", "Low confidence: manual review"


def result_card(model_name, ata, prob):
    border, bg, label = conf_style(prob)
    name = html.escape(ATA_NAMES.get(ata, "?"))
    return f"""
<div style="border:2px solid {border};background:{bg};border-radius:14px;padding:16px 18px;margin-bottom:8px">
  <div style="font-size:0.8em;font-weight:700;letter-spacing:.05em;color:#57606a;text-transform:uppercase">{html.escape(model_name)}</div>
  <div style="display:flex;align-items:center;gap:14px;margin-top:6px">
    <div style="font-size:2.6em;line-height:1">{ICONS.get(ata, "🔧")}</div>
    <div>
      <div style="font-size:1.7em;font-weight:800;color:{NAVY};line-height:1.1">ATA {ata}</div>
      <div style="font-size:1.05em;color:#24292f">{name}</div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="font-size:1.6em;font-weight:800;color:{border}">{prob:.0%}</div>
      <div style="font-size:0.75em;color:{border};font-weight:600">{label}</div>
    </div>
  </div>
</div>"""


def agreement_badge(preds_by_model):
    """preds_by_model: {model_name: ata}. Returns HTML badge, or '' for a single model."""
    if len(preds_by_model) < 2:
        return ""
    if len(set(preds_by_model.values())) == 1:
        color, bg, text = "#1a7f37", "#e9f7ee", "✅ Both models agree"
    else:
        color, bg, text = "#b7791f", "#fff6e0", "⚠️ Models disagree: worth a manual check"
    return (f'<div style="display:inline-block;border:1px solid {color};background:{bg};color:{color};'
            f'font-weight:700;padding:6px 14px;border-radius:999px;margin:4px 0 12px">{text}</div>')


def stat_strip(items):
    """items: list of (value, label)."""
    cells = "".join(
        f'<div style="flex:1;min-width:120px;background:rgba(255,255,255,0.12);border-radius:10px;padding:10px 14px">'
        f'<div style="font-size:1.5em;font-weight:800">{html.escape(str(v))}</div>'
        f'<div style="font-size:0.8em;opacity:0.85">{html.escape(l)}</div></div>'
        for v, l in items)
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px">{cells}</div>'


def header_html(stats):
    return f"""
<div style="background:linear-gradient(120deg,{NAVY} 0%,#13315c 55%,{SKY} 100%);color:white;
            border-radius:16px;padding:22px 26px;margin-bottom:18px">
  <div style="font-size:2em;font-weight:800">✈️ Aircraft Maintenance Defect Classifier</div>
  <div style="opacity:0.9;margin-top:4px">NLP that reads technical-log defect write-ups, assigns the ATA chapter,
  extracts key entities and flags ambiguous entries for engineer review.</div>
  {stat_strip(stats)}
</div>"""


def load_metrics(reports=Path("reports")):
    rows = [json.loads(p.read_text()) for p in sorted(reports.glob("*_metrics.json"))]
    return pd.DataFrame(rows)


def per_class_f1(reports=Path("reports")):
    """Parse sklearn classification_report text files -> long DataFrame (model, ata, f1, support)."""
    rows = []
    for p in sorted(reports.glob("*_classification_report.txt")):
        model = p.name.replace("_classification_report.txt", "")
        for line in p.read_text().splitlines():
            m = re.match(r"^\s*(\d{2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
            if m:
                rows.append({"model": model, "ata": m.group(1),
                             "chapter": f"ATA {m.group(1)} {ATA_NAMES.get(m.group(1), '')}",
                             "f1": float(m.group(4)), "support": int(m.group(5))})
    return pd.DataFrame(rows)


PRETTY = {"baseline": "TF-IDF + SVM", "distilbert": "DistilBERT"}


def pipeline_html():
    steps = [
        ("📝", "Defect write-up", "Free text from the tech log"),
        ("🧹", "Pre-processing", "Lower-casing, tokenisation, word + char n-grams"),
        ("🤖", "Classifiers", "TF-IDF + SVM  ·  fine-tuned DistilBERT"),
        ("🏷️", "Entity extraction", "Lexicon + regex: parts, S/N, positions, values"),
        ("🔍", "Explanation", "Leave-one-word-out importance"),
        ("📊", "Output", "ATA chapter, confidence, review flag"),
    ]
    boxes = []
    for i, (ic, t, d) in enumerate(steps):
        boxes.append(
            f'<div style="flex:1;min-width:140px;background:#ffffff;border:1px solid #d0d7e2;border-radius:12px;'
            f'padding:12px;text-align:center"><div style="font-size:1.8em">{ic}</div>'
            f'<div style="font-weight:700;color:{NAVY}">{t}</div>'
            f'<div style="font-size:0.8em;color:#57606a">{d}</div></div>')
        if i < len(steps) - 1:
            boxes.append(f'<div style="align-self:center;font-size:1.4em;color:{SKY}">➜</div>')
    return '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch">' + "".join(boxes) + "</div>"


if __name__ == "__main__":
    print(per_class_f1().head())
    print(load_metrics())
    print(result_card("TF-IDF + SVM", "72", 0.89)[:120])
    print(agreement_badge({"a": "72", "b": "49"})[:80])
