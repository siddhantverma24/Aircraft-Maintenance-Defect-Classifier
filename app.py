"""
app.py - Streamlit demo
Run:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from generate_data import ATA_NAMES
from ner import LABEL_COLORS, extract_entities, render_html
from predict import available_models, explain, explanation_html
from ui_helpers import (ICONS, PRETTY, REVIEW_THRESHOLD, SKY, agreement_badge,
                        header_html, load_metrics, per_class_f1, pipeline_html,
                        result_card)

st.set_page_config(page_title="ATA Defect Classifier", page_icon="✈️", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1250px;}
  div[data-testid="stTabs"] button p {font-size: 1.02rem; font-weight: 600;}
  div[data-testid="stMetricValue"] {color: #0b2545;}
</style>""", unsafe_allow_html=True)

EXAMPLES = [
    "#2 EGT exceeded limits during climb on VT-ABC. EGT thermocouple replaced P/N 2043M41P01 S/N K123456",
    "L/H main gear tire cut found during walkaround. P/N 572756-1 replaced IAW AMM 32-45-11. ops check satisfactory",
    "Crew reported HYD PUMP LOW PRESSURE light. PTU reading 1570 psi. replaced per TSM",
    "BLEED TRIP OFF light during climb. R/H PRSOV replaced IAW AMM. leak check ok",
    "valve leaking in wheel well",
    "APU not starting on arrival, APU igniter changed",
]


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    return available_models()[name]()


@st.cache_data
def dataset_size():
    p = Path("data/defects.csv")
    return len(pd.read_csv(p)) if p.exists() else 0


models_avail = available_models()
if not models_avail:
    st.error("No trained models found. Run `python run_all.py` first.")
    st.stop()
models = {name: load_model(name) for name in models_avail}
metrics = load_metrics()

# ------------------------------------------------------------------ header
best_acc = f"{metrics['accuracy'].max():.1%}" if not metrics.empty else "-"
st.markdown(header_html([
    (len(ATA_NAMES), "ATA chapters"),
    (f"{dataset_size():,}", "synthetic write-ups"),
    (len(models), "models"),
    (best_acc, "best test accuracy"),
]), unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    explain_model = st.selectbox("Explain predictions with", list(models), index=0)
    show_explain = st.toggle("Show word importance", value=True)
    st.divider()
    st.markdown("**Entity colours**")
    st.markdown(" ".join(
        f'<span style="background:{c};color:#111;padding:2px 8px;border-radius:6px;'
        f'display:inline-block;margin:2px;font-size:0.85em">{lab}</span>'
        for lab, c in LABEL_COLORS.items()), unsafe_allow_html=True)
    st.divider()
    st.markdown("**Confidence**")
    st.markdown("🟢 ≥ 80% high  \n🟡 60–80% medium  \n🔴 < 60% → manual review")
    if "DistilBERT" not in models:
        st.info("DistilBERT not trained yet: run `python train_bert.py` to enable it.")


def top_chart(model, text, k=5):
    p = model.predict_proba([text])[0]
    df = pd.DataFrame({"ata": model.classes, "prob": p}).sort_values("prob", ascending=False).head(k)
    df["label"] = df["ata"].map(lambda a: f"{ICONS.get(a, '')} ATA {a} {ATA_NAMES.get(a, '')}")
    df["rank"] = ["top"] + ["other"] * (len(df) - 1)
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("prob:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=18).encode(
        color=alt.Color("rank:N", scale=alt.Scale(domain=["top", "other"], range=[SKY, "#b6c6d8"]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Chapter"), alt.Tooltip("prob:Q", format=".1%", title="Probability")],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("prob:Q", format=".0%"))
    return (bars + labels).properties(height=34 * len(df), width="container"), df.iloc[0]


tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📄 Batch", "📊 Model comparison", "ℹ️ About"])

# ------------------------------------------------------------------ tab 1: classify
with tab1:
    ex = st.selectbox("Try an example, or type your own below", ["-"] + EXAMPLES)
    text = st.text_area("Defect write-up", value="" if ex == "-" else ex, height=90,
                        placeholder="e.g. #1 engine oil leak found during transit check, oil filter replaced")
    go = st.button("Classify", type="primary")

    if go and text.strip():
        preds = {}
        cols = st.columns(len(models))
        charts = []
        for col, (name, m) in zip(cols, models.items()):
            chart, best = top_chart(m, text)
            preds[name] = best["ata"]
            with col:
                st.markdown(result_card(name, best["ata"], float(best["prob"])), unsafe_allow_html=True)
                st.altair_chart(chart)
        badge = agreement_badge(preds)
        if badge:
            st.markdown(badge, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### 🏷️ Extracted entities")
            st.markdown(render_html(text), unsafe_allow_html=True)
            ents = extract_entities(text)
            if ents:
                counts = pd.Series([e["label"] for e in ents]).value_counts()
                st.caption(" · ".join(f"{lab}: {n}" for lab, n in counts.items()))
            else:
                st.caption("No entities found.")

        if show_explain:
            with st.container(border=True):
                m = models[explain_model]
                cls, scores = explain(m, text)
                st.markdown(f"#### 🔍 Why {explain_model} chose ATA {cls}")
                st.caption("Each word is removed in turn and the model re-predicts. "
                           "🟩 green words pushed it towards this chapter, 🟥 red words pushed it away.")
                st.markdown(explanation_html(scores), unsafe_allow_html=True)
    elif go:
        st.warning("Type a write-up first.")

# ------------------------------------------------------------------ tab 2: batch
with tab2:
    st.markdown("Upload a CSV with a **`text`** column. If it also has an **`ata`** column, accuracy is shown.")
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if c2.button("Use sample file"):
        st.session_state["use_sample"] = True
    df = None
    if up is not None:
        df = pd.read_csv(up, dtype={"ata": str})
    elif st.session_state.get("use_sample") and Path("data/sample_batch.csv").exists():
        df = pd.read_csv("data/sample_batch.csv", dtype={"ata": str})

    if df is not None:
        if "text" not in df.columns:
            st.error("CSV needs a 'text' column.")
        else:
            primary = st.radio("Model for results", list(models), horizontal=True, index=len(models) - 1)
            texts = df["text"].astype(str)
            for name, m in models.items():
                probs = m.predict_proba(texts)
                df[f"{name} ATA"] = [m.classes[i] for i in probs.argmax(1)]
                df[f"{name} conf"] = probs.max(1).round(3)
            df["chapter"] = df[f"{primary} ATA"].map(lambda a: f"{ICONS.get(a, '')} {a} {ATA_NAMES.get(a, '')}")
            df["needs_review"] = df[f"{primary} conf"] < REVIEW_THRESHOLD
            if len(models) > 1:
                ata_cols = [f"{n} ATA" for n in models]
                df["models_agree"] = df[ata_cols].nunique(axis=1) == 1

            k = st.columns(4)
            k[0].metric("Write-ups", len(df))
            k[1].metric("Chapters found", df[f"{primary} ATA"].nunique())
            k[2].metric("Flagged for review", int(df["needs_review"].sum()))
            if "ata" in df.columns:
                k[3].metric("Accuracy on this file", f"{(df['ata'] == df[f'{primary} ATA']).mean():.1%}")
            elif "models_agree" in df:
                k[3].metric("Models agree", f"{df['models_agree'].mean():.0%}")

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Defects by ATA chapter**")
                cnt = df["chapter"].value_counts().reset_index()
                cnt.columns = ["chapter", "count"]
                st.altair_chart(alt.Chart(cnt).mark_bar(cornerRadiusEnd=4, color=SKY).encode(
                    y=alt.Y("chapter:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                    x=alt.X("count:Q", title="Write-ups"), tooltip=["chapter", "count"],
                ).properties(height=max(200, 24 * len(cnt)), width="container"))
            with g2:
                st.markdown("**Confidence distribution**")
                hist = alt.Chart(df).mark_bar(color="#7a9cc6").encode(
                    x=alt.X(f"{primary} conf:Q", bin=alt.Bin(maxbins=20), title="Confidence"),
                    y=alt.Y("count():Q", title="Write-ups"))
                rule = alt.Chart(pd.DataFrame({"x": [REVIEW_THRESHOLD]})).mark_rule(
                    color="#cf222e", strokeDash=[6, 4], size=2).encode(x="x:Q")
                st.altair_chart((hist + rule).properties(height=max(200, 24 * len(cnt)), width="container"))
                st.caption("Red dashed line = review threshold (60%).")

            show = ["text", "chapter", f"{primary} conf", "needs_review"] + \
                   (["models_agree"] if "models_agree" in df else []) + (["ata"] if "ata" in df else [])
            styled = df[show].style.apply(
                lambda r: ["background-color:#fdecec" if r["needs_review"] else "" for _ in r], axis=1)
            st.dataframe(styled, hide_index=True)
            st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(),
                               "classified_defects.csv", "text/csv")

# ------------------------------------------------------------------ tab 3: comparison
with tab3:
    if metrics.empty:
        st.info("No metrics yet - train the models first.")
    else:
        metrics["Model"] = metrics["model"].map(PRETTY).fillna(metrics["model"])
        cols = st.columns(len(metrics))
        best_acc_v, best_f1_v = metrics["accuracy"].max(), metrics["macro_f1"].max()
        for col, (_, r) in zip(cols, metrics.iterrows()):
            with col, st.container(border=True):
                st.markdown(f"### {r['Model']}")
                a, b = st.columns(2)
                a.metric("Test accuracy", f"{r['accuracy']:.1%}",
                         None if r["accuracy"] == best_acc_v else f"{r['accuracy'] - best_acc_v:+.1%}")
                b.metric("Macro-F1", f"{r['macro_f1']:.3f}",
                         None if r["macro_f1"] == best_f1_v else f"{r['macro_f1'] - best_f1_v:+.3f}")
                c, d = st.columns(2)
                c.metric("Training time", f"{r['train_time_s']:.0f} s")
                d.metric("Inference", f"{r['infer_ms_per_item']:.2f} ms / item")

        pcf = per_class_f1()
        if not pcf.empty:
            st.markdown("#### Per-chapter F1 score")
            st.caption("Both models struggle on the same small chapters with overlapping vocabulary.")
            pcf["Model"] = pcf["model"].map(PRETTY).fillna(pcf["model"])
            st.altair_chart(alt.Chart(pcf).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("chapter:N", title=None, sort=alt.EncodingSortField("f1", op="min", order="ascending"),
                        axis=alt.Axis(labelLimit=230)),
                yOffset="Model:N",
                x=alt.X("f1:Q", title="F1 score", scale=alt.Scale(domain=[0.5, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(range=[SKY, "#f59e0b"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Model", "chapter", alt.Tooltip("f1:Q", format=".3f"), "support"],
            ).properties(height=560, width="container"))

        st.markdown("#### Confusion matrices")
        ccols = st.columns(len(metrics))
        for col, (_, r) in zip(ccols, metrics.iterrows()):
            img = Path(f"reports/{r['model']}_confusion_matrix.png")
            if img.exists():
                col.image(str(img), caption=r["Model"])

# ------------------------------------------------------------------ tab 4: about
with tab4:
    st.markdown("### How it works")
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown("#### 🎯 Problem")
        st.markdown("Every aircraft defect is recorded as free text and must be coded to an ATA chapter "
                    "for reliability analysis, repeat-defect tracking and regulatory reporting. Manual "
                    "coding is slow and inconsistent. This tool automates it and flags unclear entries "
                    "for an engineer to review.")
        st.markdown("#### 📚 Dataset")
        st.markdown(f"**{dataset_size():,} synthetic write-ups** across **{len(ATA_NAMES)} ATA chapters**, "
                    "generated from technical-log style templates with abbreviations, part numbers, "
                    "typos, ALL-CAPS entries and ~18% deliberately vague entries. Split 70/15/15, "
                    "stratified by chapter.")
    with b, st.container(border=True):
        st.markdown("#### 🤖 Models")
        st.markdown("- **TF-IDF + Linear SVM**: word 1-2 grams and character 2-5 grams\n"
                    "- **DistilBERT**: fine-tuned transformer, GPU mixed precision\n"
                    "- **Entity extraction**: ATA component lexicon + regular expressions\n"
                    "- **Explainability**: leave-one-word-out occlusion")
        st.markdown("#### 🚀 Future scope")
        st.markdown("Real FAA SDR data · aviation-domain language model · sub-chapter codes "
                    "(e.g. 72-30) · repeat-defect detection · MRO system integration")
    st.markdown("#### ATA chapters covered")
    st.markdown(" ".join(
        f'<span style="display:inline-block;background:#e8eef7;border-radius:8px;padding:4px 10px;'
        f'margin:3px;font-size:0.9em">{ICONS.get(a, "")} <b>{a}</b> {n}</span>'
        for a, n in ATA_NAMES.items()), unsafe_allow_html=True)
