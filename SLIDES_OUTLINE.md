# Presentation outline (8 slides, ~7 minutes)

**1. Title**
Aircraft Maintenance Defect Classifier: auto-coding technical-log write-ups to ATA chapters with NLP.
Your name, course, date.

**2. Problem**
- Every defect is written as free text in the tech log.
- It must be coded to an ATA chapter for reliability programmes, repeat-defect tracking and regulator reporting.
- Manual coding is slow, inconsistent between engineers, and error-prone.
- Goal: read the write-up → predict ATA chapter + extract key details + flag ambiguous ones.

**3. Dataset**
- 3,000 synthetic write-ups across 16 ATA chapters, generated from tech-log style templates.
- Realism: abbreviations, part/serial numbers, typos, ALL CAPS, and ~18% vague entries.
- Show 3 example rows. Show class distribution (imbalanced like a real fleet).
- Say clearly: synthetic; real FAA SDR data is the next step.

**4. Architecture** (diagram)
Write-up → [TF-IDF + SVM]  /  [DistilBERT] → ATA chapter + confidence
       → [Rule-based NER] → entities
       → [Occlusion] → word importance
       → Streamlit app (single, batch, comparison)

**5. Models**
- Baseline: TF-IDF word + char n-grams, Linear SVM. Why char n-grams: abbreviations and part numbers.
- DistilBERT: 66M parameters, fine-tuned 4 epochs on GPU, mixed precision.

**6. Results**
- Comparison table: accuracy, macro-F1, train time, inference ms (from the app's comparison tab).
- One confusion matrix. Point out the confusions that make sense:
  fire detection (26) vs APU (49), ice & rain (30) vs navigation probes (34).

**7. Live demo** (2 minutes)
1. Engine example → ATA 72, entities highlighted, word importance.
2. "valve leaking in wheel well" → low confidence, split between 32 and 29, "manual review" flag. This shows the model knows when it's unsure.
3. Batch tab → sample file → chapter bar chart → download.

**8. Limitations & future scope**
- Synthetic data; rule-based NER; chapter-level only.
- Next: real SDR data, aviation-domain language model, sub-chapter codes, repeat-defect detection, MRO system integration.

---

## Likely viva questions (prepare these)
- **Why macro-F1 and not just accuracy?** Classes are imbalanced; macro-F1 weighs small chapters equally.
- **Why character n-grams?** They handle abbreviations, typos and part numbers that word features miss.
- **Why DistilBERT over BERT?** 40% smaller, ~60% faster, keeps ~97% of BERT's performance; fits a 4 GB GPU.
- **How do you prevent data leakage?** Fixed stratified split; test set never seen in training or model selection (best epoch chosen on validation).
- **Why is the data synthetic?** Real tech-log data is proprietary; FAA SDR is public and is the planned next step.
- **How does the explanation work?** Remove each word, re-predict, measure the confidence drop.
- **What does the confidence threshold do?** Below 60% the entry is flagged for an engineer to review; human in the loop.
