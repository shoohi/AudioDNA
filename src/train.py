"""Model training and evaluation.

Trains a logistic-regression baseline and a grid-searched random forest
on the Phase 3 feature vectors, evaluates on a held-out test set, writes
a confusion matrix figure + metrics report to reports/, and persists the
fitted scaler + best model to models/.

Usage:
    python -m src.train
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # render figures to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CATEGORIES, MODELS_DIR, REPORTS_DIR
from src.db import get_connection

log = logging.getLogger("train")

MODEL_PATH = MODELS_DIR / "audiodna_model.joblib"

# Reused everywhere randomness appears so results are reproducible run-to-run.
RANDOM_STATE = 42

# Why these acoustic explanations live in code: the trained model's top
# confusions are summarized in plain language at the end of the run, and
# these pairs are the ones EDA (Phase 4) predicted would collide.
CONFUSION_EXPLANATIONS = {
    frozenset({"explosion", "impact"}): (
        "both are single percussive low-frequency bursts; a distant boom and "
        "a heavy thud have very similar spectral envelopes"
    ),
    frozenset({"explosion", "weapon"}): (
        "gunshots ARE small explosions acoustically — sharp broadband burst "
        "with a low-frequency tail"
    ),
    frozenset({"impact", "weapon"}): (
        "melee weapon sounds (sword hits, gun handling) are literally impact "
        "sounds recorded under a different label"
    ),
    frozenset({"ambience", "footsteps"}): (
        "walking-loop footstep clips are long, steady, broadband — exactly "
        "the statistical profile of ambience"
    ),
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Feature matrix X, labels y, and the feature column order.

    The column order matters: the saved model expects features in exactly
    this order forever after, so we persist it alongside the model.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT s.category, f.feature_json FROM sounds s "
            "JOIN features f ON f.sound_id = s.id ORDER BY s.id"
        ).fetchall()
    X = pd.json_normalize([json.loads(r["feature_json"]) for r in rows])
    y = pd.Series([r["category"] for r in rows], name="category")
    return X, y, list(X.columns)


def evaluate(name: str, model, X_test, y_test) -> tuple[float, np.ndarray]:
    """Print accuracy + per-class report; return accuracy and the
    confusion matrix (rows = true class, columns = predicted class)."""
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    log.info("%s — test accuracy: %.3f", name, acc)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, labels=CATEGORIES, digits=3))
    return acc, confusion_matrix(y_test, pred, labels=CATEGORIES)


def save_confusion_figure(cm: np.ndarray, accuracy: float) -> None:
    """Heatmap of the confusion matrix, normalized by row (true class),
    so each cell reads 'what fraction of true-X was predicted as Y'."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, cbar_kws={"label": "recall"},
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Random forest confusion matrix (test accuracy {accuracy:.1%})")
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("confusion matrix figure -> %s", out)


def summarize_confusions(cm: np.ndarray, top_n: int = 3) -> None:
    """Plain-language summary of the largest off-diagonal cells."""
    print("\n=== Where the model gets confused ===")
    pairs = []
    for i, true_cat in enumerate(CATEGORIES):
        for j, pred_cat in enumerate(CATEGORIES):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], true_cat, pred_cat))
    pairs.sort(reverse=True)
    for count, true_cat, pred_cat in pairs[:top_n]:
        line = f"- {count} '{true_cat}' sounds were predicted as '{pred_cat}'"
        why = CONFUSION_EXPLANATIONS.get(frozenset({true_cat, pred_cat}))
        if why:
            line += f" — plausible because {why}."
        print(line)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y, feature_cols = load_dataset()
    log.info("dataset: %d sounds x %d features", *X.shape)

    # Stratified split: each category keeps the same proportion in train
    # and test, so test metrics aren't skewed by an unlucky draw.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Fit the scaler on TRAIN ONLY. Fitting on all data would leak test-set
    # statistics (its mean/std) into training — a subtle form of cheating
    # that inflates reported accuracy.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    # (Tree models don't need scaling — they only compare thresholds within
    # one feature — but it's harmless, and using scaled features everywhere
    # means a single preprocessing path for both models and the app.)

    # --- Baseline: logistic regression -----------------------------------
    # A linear model: cheap, hard to overfit, and an honest yardstick.
    # If the fancy model can't beat this, the fancy model isn't earning
    # its complexity.
    logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    logreg.fit(X_train_s, y_train)
    logreg_acc, _ = evaluate("Logistic regression (baseline)", logreg, X_test_s, y_test)

    # --- Main model: random forest with a small grid search --------------
    # 5-fold CV on the training set picks hyperparameters without ever
    # touching the test set. The grid is deliberately small: with ~720
    # training samples, fine-tuning beyond this mostly fits noise.
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)
    log.info("best RF params: %s (CV accuracy %.3f)", grid.best_params_, grid.best_score_)
    rf = grid.best_estimator_
    rf_acc, cm = evaluate("Random forest (best)", rf, X_test_s, y_test)

    save_confusion_figure(cm, rf_acc)
    summarize_confusions(cm)

    # --- Persist artifacts ------------------------------------------------
    # One bundle file instead of separate scaler/model files: the app can
    # never accidentally load a scaler from one training run with a model
    # from another. feature_cols pins the expected column order.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "model": rf,
            "feature_cols": feature_cols,
            "categories": CATEGORIES,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_accuracy": rf_acc,
        },
        MODEL_PATH,
    )
    log.info("model bundle -> %s", MODEL_PATH)

    # Text metrics report for the README / reports folder.
    report = classification_report(
        y_test, rf.predict(X_test_s), labels=CATEGORIES, digits=3
    )
    metrics_path = REPORTS_DIR / "metrics.txt"
    metrics_path.write_text(
        f"AudioDNA — random forest evaluation ({datetime.now(timezone.utc):%Y-%m-%d})\n"
        f"Baseline logistic regression accuracy: {logreg_acc:.3f}\n"
        f"Random forest params: {grid.best_params_}\n"
        f"Random forest test accuracy: {rf_acc:.3f}\n\n{report}",
        encoding="utf-8",
    )
    log.info("metrics report -> %s", metrics_path)


if __name__ == "__main__":
    main()
