"""
Step 4 — Random Forest Baseline
CICIDS2017 PortScan / Adaptive MTD project

Reads (read-only, never modified):
  step3_train_windows.csv
  step3_test_windows.csv
  step3_episode_summary.csv

Writes (new files only):
  step4_rf_metrics.json
  step4_rf_confusion_matrix.csv
  step4_rf_pr_curve.csv
  step4_rf_feature_importance.csv
  step4_rf_per_episode_detection.csv
  step4_rf_predictions.csv
  step4_rf_pr_curve.png (only if matplotlib is available; skipped otherwise)

Architecture note: this RF baseline is the bar that CNN must beat before
CNN is considered load-bearing (per project design). LSTM must separately
beat EWMA before being treated as primary. Neither is built in this step.

Evaluation note: with only 27 positive windows total (8 in test), accuracy
is not used as the primary metric. See printed caveats.

Feature note: RF is scale-invariant, so this script trains on the RAW
(unscaled) step3_*_windows.csv files, not the *_scaled.csv versions. The
scaler is left untouched for models that actually need it (e.g. LSTM).

Tested against: Python 3.14.5, pandas 3.0.5, numpy 2.5.1
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
)

RANDOM_STATE = 42
N_ESTIMATORS = 300


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
def load_windows():
    log("=== Loading Step 3 outputs (read-only) ===")
    train_df = pd.read_csv("step3_train_windows.csv")
    test_df = pd.read_csv("step3_test_windows.csv")
    episode_summary = pd.read_csv("step3_episode_summary.csv",
                                   parse_dates=["start_minute", "end_minute"])

    meta_cols = ["source_ip", "target_minute", "label", "split"]
    feat_cols = [c for c in train_df.columns if c not in meta_cols]

    log(f"Train windows: {len(train_df)} | positives: {train_df['label'].sum()}")
    log(f"Test windows : {len(test_df)} | positives: {test_df['label'].sum()}")
    log(f"Feature columns: {len(feat_cols)} (expected 45 = 5 lags x 9 features)")

    if test_df["label"].sum() != 8:
        log(f"NOTE: expected 8 positive test windows per prior audit, "
            f"found {test_df['label'].sum()} — re-check before trusting results.")

    return train_df, test_df, episode_summary, feat_cols


# ----------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------
def train_rf(train_df, feat_cols):
    log("\n=== Training Random Forest (class_weight='balanced_subsample') ===")
    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=int)

    log(f"Train class balance: {y_train.sum()} positive / {len(y_train)} total "
        f"({100 * y_train.sum() / len(y_train):.3f}%)")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    log("RF training complete.")
    return clf


# ----------------------------------------------------------------------
# EVALUATE
# ----------------------------------------------------------------------
def evaluate(clf, test_df, feat_cols):
    log("\n=== Evaluating on test set ===")
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=int)
    n_pos = int(y_test.sum())

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)  # printed but flagged non-diagnostic

    log("\n--- CONFUSION MATRIX (raw counts) ---")
    log(f"                 Predicted Benign   Predicted PortScan")
    log(f"Actual Benign    {tn:>16}   {fp:>18}")
    log(f"Actual PortScan  {fn:>16}   {tp:>18}")

    log("\n--- METRICS ---")
    log(f"Precision       : {precision:.4f}")
    log(f"Recall          : {recall:.4f}  ({tp}/{n_pos} positive windows detected)")
    log(f"F1              : {f1:.4f}")
    log(f"PR-AUC          : {pr_auc:.4f}")
    log(f"Balanced Acc.   : {bal_acc:.4f}")
    log(f"Accuracy        : {acc:.4f}  <-- NOT DIAGNOSTIC, see caveat below")

    log("\n--- SMALL-SAMPLE CAVEAT (mandatory reading before comparing models) ---")
    log(f"Test set has only {n_pos} positive windows. Each single test error moves")
    log(f"recall by {100/n_pos:.1f} percentage points. A model predicting ALL-NEGATIVE")
    log(f"on this test set would score {100*(len(y_test)-n_pos)/len(y_test):.2f}% accuracy")
    log(f"while detecting 0 attacks — this is why accuracy is not used as the primary metric.")
    log(f"Do not treat recall/F1 differences smaller than ~{100/n_pos:.1f}pp between model")
    log(f"variants as meaningful; that is within single-sample noise at this test size.")

    metrics = {
        "n_train": len(clf.estimators_) and None,  # placeholder, filled below
        "n_test_windows": len(y_test),
        "n_test_positive": n_pos,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(bal_acc),
        "accuracy_NOT_DIAGNOSTIC": float(acc),
        "caveat": (
            f"Test set has only {n_pos} positive windows; each single error moves "
            f"recall by {100/n_pos:.1f} percentage points. Do not compare model "
            f"variants on differences smaller than this."
        ),
    }
    del metrics["n_train"]

    return y_pred, y_proba, metrics, cm


# ----------------------------------------------------------------------
# PR CURVE
# ----------------------------------------------------------------------
def save_pr_curve(test_df, y_proba):
    y_test = test_df["label"].to_numpy(dtype=int)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    pr_df = pd.DataFrame({
        "precision": precisions,
        "recall": recalls,
        "threshold": np.append(thresholds, np.nan),
    })
    pr_df.to_csv("step4_rf_pr_curve.csv", index=False)
    log(f"\nSaved: step4_rf_pr_curve.csv ({len(pr_df)} points — expect a coarse, "
        f"step-like curve given only 8 positive test samples)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.step(recalls, precisions, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("RF Baseline — Precision-Recall Curve (n_pos_test=8)")
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("step4_rf_pr_curve.png", dpi=150)
        plt.close()
        log("Saved: step4_rf_pr_curve.png")
    except ImportError:
        log("matplotlib not available — skipped PNG plot, CSV still saved.")


# ----------------------------------------------------------------------
# PER-EPISODE DETECTION (test episodes 7, 8, 9)
# ----------------------------------------------------------------------
def per_episode_detection(test_df, y_pred, episode_summary):
    log("\n=== Per-episode detection (test episodes 7, 8, 9) ===")

    test_df = test_df.copy()
    test_df["pred"] = y_pred
    test_df["target_minute"] = pd.to_datetime(test_df["target_minute"])

    test_episodes = episode_summary[episode_summary["split"] == "test"]
    rows = []

    for _, ep in test_episodes.iterrows():
        ep_id = ep["episode_id"]
        start, end = ep["start_minute"], ep["end_minute"]
        mask = (
            (test_df["target_minute"] >= start)
            & (test_df["target_minute"] <= end)
            & (test_df["label"] == 1)
        )
        ep_windows = test_df[mask]
        n_positive = len(ep_windows)
        n_detected = int(ep_windows["pred"].sum())

        if n_positive == 0:
            status = "NO POSITIVE WINDOWS FOUND (check episode/window alignment)"
        elif n_detected == n_positive:
            status = "FULLY DETECTED"
        elif n_detected > 0:
            status = "PARTIALLY DETECTED"
        else:
            status = "MISSED ENTIRELY"

        missed_minutes = ep_windows[ep_windows["pred"] == 0]["target_minute"].tolist()

        log(f"Episode {ep_id} [{start} - {end}]: "
            f"{n_detected}/{n_positive} minutes detected — {status}")
        if missed_minutes:
            log(f"  Missed minutes: {missed_minutes}")

        rows.append({
            "episode_id": ep_id,
            "start_minute": start,
            "end_minute": end,
            "n_positive_minutes": n_positive,
            "n_detected": n_detected,
            "status": status,
        })

    pd.DataFrame(rows).to_csv("step4_rf_per_episode_detection.csv", index=False)
    log("\nSaved: step4_rf_per_episode_detection.csv")


# ----------------------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------------------
def save_feature_importance(clf, feat_cols):
    imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv("step4_rf_feature_importance.csv", index=False)

    log("\n=== Top 10 features by RF importance ===")
    for _, row in imp.head(10).iterrows():
        log(f"  {row['feature']:<30} {row['importance']:.4f}")
    log("\nSaved: step4_rf_feature_importance.csv")


# ----------------------------------------------------------------------
# SAVE PREDICTIONS + METRICS
# ----------------------------------------------------------------------
def save_predictions(test_df, y_pred, y_proba):
    out = test_df[["source_ip", "target_minute", "label"]].copy()
    out["predicted_label"] = y_pred
    out["predicted_proba"] = y_proba
    out.to_csv("step4_rf_predictions.csv", index=False)
    log("Saved: step4_rf_predictions.csv")


def save_metrics(metrics):
    with open("step4_rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved: step4_rf_metrics.json")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual Benign", "Actual PortScan"],
        columns=["Predicted Benign", "Predicted PortScan"],
    )
    cm_df.to_csv("step4_rf_confusion_matrix.csv")
    log("Saved: step4_rf_confusion_matrix.csv")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    train_df, test_df, episode_summary, feat_cols = load_windows()
    clf = train_rf(train_df, feat_cols)
    y_pred, y_proba, metrics, cm = evaluate(clf, test_df, feat_cols)

    save_metrics(metrics)
    save_predictions(test_df, y_pred, y_proba)
    save_pr_curve(test_df, y_proba)
    save_feature_importance(clf, feat_cols)
    per_episode_detection(test_df, y_pred, episode_summary)

    log("\n=== STEP 4 (RF BASELINE) COMPLETE ===")
    log("This RF result is the bar CNN must beat before CNN is treated as load-bearing.")
    log("Review metrics, confusion matrix, PR curve, and per-episode detection before")
    log("proceeding to CNN. Do not proceed to CNN/LSTM until this is reviewed.")


if __name__ == "__main__":
    main()
