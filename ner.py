"""
ner.py
------
Rule-based Named Entity Recognition for maintenance write-ups
(domain lexicon + regular expressions, longest-match conflict resolution).
Pure Python, no model download needed.

Entities:
  COMPONENT     aircraft components (from an ATA component lexicon)
  PART_NO       part numbers        e.g. 572756-1, 2043M41P01, P/N S56521-01
  SERIAL_NO     serial numbers      e.g. S/N Y777425
  AIRCRAFT_REG  registrations       e.g. VT-ABC
  POSITION      L/H, R/H, #2, fwd, inbd ...
  MEASUREMENT   2945 psi, 640 degC, 3 mm ...
  MANUAL_REF    AMM 32-41-11, CMM, FIM, TSM ...

Usage:
    from ner import extract_entities, render_html
"""
import html
import re

from generate_data import CHAPTERS, GENERIC_COMP

LABEL_COLORS = {
    "COMPONENT": "#a7d8ff",
    "PART_NO": "#ffd59e",
    "SERIAL_NO": "#ffe8a3",
    "AIRCRAFT_REG": "#c9b8ff",
    "POSITION": "#b8f2c9",
    "MEASUREMENT": "#ffb8c6",
    "MANUAL_REF": "#d9d9d9",
}

_components = sorted({c for ch in CHAPTERS.values() for c in ch["comp"]} | set(GENERIC_COMP),
                     key=len, reverse=True)
_comp_re = r"(?<!\w)(?:" + "|".join(re.escape(c) for c in _components) + r")s?(?!\w)"

PATTERNS = [
    # (label, regex, capture group to keep)
    ("SERIAL_NO", r"\bS/?N\s?[A-Z]?\d{5,7}\b", 0),
    ("PART_NO", r"P/N\s?([A-Z0-9][A-Z0-9-]{3,}[A-Z0-9])", 1),
    ("PART_NO", r"\b\d{6}-\d{1,2}\b", 0),
    ("PART_NO", r"\b\d{4}M\d{2}P\d{2}\b", 0),
    ("AIRCRAFT_REG", r"\bVT-[A-Z]{3}\b", 0),
    ("MANUAL_REF", r"\b(?:AMM|CMM|FIM|TSM|SRM|IPC)(?:\s\d{2}-\d{2}-\d{2})?\b", 0),
    ("MEASUREMENT", r"\b\d+(?:\.\d+)?\s?(?:psi|degC|°C|mm|qts|kg|units|%)(?!\w)", 0),
    ("POSITION", r"(?<!\w)(?:L/H|R/H|LH|RH|#[1-4]|No\.\s?[1-4]|left|right|fwd|aft|inbd|outbd|upper|lower)(?!\w)", 0),
    ("COMPONENT", _comp_re, 0),
]


def extract_entities(text):
    """Return non-overlapping entities: list of {text, label, start, end}, in text order."""
    cands = []
    for label, pat, grp in PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            s, e = m.span(grp)
            cands.append((s, e, label))
    # longest span wins; ties keep pattern priority (earlier in PATTERNS)
    cands.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    taken, out = [], []
    for s, e, label in cands:
        if all(e <= ts or s >= te for ts, te in taken):
            taken.append((s, e))
            out.append({"text": text[s:e], "label": label, "start": s, "end": e})
    return sorted(out, key=lambda d: d["start"])


def render_html(text):
    """Colour-highlighted HTML for Streamlit (st.markdown(..., unsafe_allow_html=True))."""
    parts, pos = [], 0
    for ent in extract_entities(text):
        parts.append(html.escape(text[pos:ent["start"]]))
        color = LABEL_COLORS[ent["label"]]
        parts.append(
            f'<span style="background:{color};color:#111;padding:2px 6px;margin:0 2px;'
            f'border-radius:6px;line-height:2.2">{html.escape(ent["text"])}'
            f'<span style="font-size:0.65em;font-weight:700;margin-left:6px;opacity:0.7">'
            f'{ent["label"]}</span></span>')
        pos = ent["end"]
    parts.append(html.escape(text[pos:]))
    return '<div style="font-size:1.05em">' + "".join(parts) + "</div>"


if __name__ == "__main__":
    samples = [
        "L/H main gear tire cut found during walkaround. P/N 572756-1 replaced IAW AMM 32-45-11. ops check satisfactory",
        "#2 EGT exceeded limits during climb on VT-ABC. EGT thermocouple reading 980 degC, replaced P/N 2043M41P01 S/N K123456",
        "CREW REPORTED HYD PUMP LOW PRESSURE LIGHT. PTU READING 1570 PSI. REPLACED PER TSM",
        "valve leaking in wheel well",
    ]
    for s in samples:
        print(s)
        for e in extract_entities(s):
            print(f"   {e['label']:13s} {e['text']}")
        print()
