"""
train_baseline.py
-----------------
Baseline: TF-IDF (word 1-2 grams + character 2-5 grams) + Linear SVM.
Character n-grams handle abbreviations (hyd, eng) and part numbers well.

Usage:  python train_baseline.py
Output: models/baseline.joblib, reports/baseline_*.{txt,png,json}
"""
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common import evaluate, load_splits


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)),
    ])
    # Calibration gives probabilities, which the app shows as confidence bars.
    clf = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)
    return Pipeline([("tfidf", features), ("svm", clf)])


def main():
    train, val, test = load_splits()
    labels = sorted(train["ata"].unique())

    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(train["text"], train["ata"])
    train_time = time.time() - t0

    val_acc = (pipe.predict(val["text"]) == val["ata"]).mean()
    print(f"Validation accuracy: {val_acc:.3f}")

    t0 = time.time()
    preds = pipe.predict(test["text"])
    infer_ms = (time.time() - t0) * 1000 / len(test)

    evaluate("baseline", test["ata"].values, preds, labels, train_time, infer_ms)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/baseline.joblib")
    print("Saved models/baseline.joblib")


if __name__ == "__main__":
    main()
