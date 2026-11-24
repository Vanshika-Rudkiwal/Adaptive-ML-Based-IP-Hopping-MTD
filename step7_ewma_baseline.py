"""
Step 7 — EWMA Statistical Trend Baseline
CICIDS2017 PortScan / Adaptive MTD project

Reads (read-only, never modified):
  step3_train_windows_scaled.csv
  step3_test_windows_scaled.csv
  step3_episode_summary.csv
  step4_rf_metrics.json    (comparison only)
  step5_cnn_metrics.json   (comparison only)
  step6_lstm_metrics.json  (comparison only)

Writes (new files only):
  step7_ewma_metrics.json
  step7_ewma_confusion_matrix.csv
  step7_ewma_pr_curve.csv
  step7_ewma_per_episode_detection.csv
  step7_ewma_predictions.csv
  step7_all_models_comparison.csv

Why this step exists: per the locked architecture, LSTM is secondary/
experimental (only 9 episodes, 1 attacker IP) and EWMA is the PRIMARY
escalation baseline. The rule was: "LSTM must beat EWMA before being
included as a primary component." RF and CNN were never LSTM's real
qualifying bar. This script builds that missing baseline.

Design: for each window, compute an exponentially-weighted moving
average per feature across the 5 lag slots (t-5...t-1), most recent
(t-1) weighted highest. Combine into one composite escalation score
(equal-weight mean across the 9 already-scaled features — no learned
coefficients, this stays a statistical baseline, not a trained model).
Classify by a threshold.

Hyperparameters (alpha, threshold) are selected using TRAIN DATA ONLY:
  - alpha chosen via 5-fold stratified CV on train (mean val PR-AUC)
  - threshold chosen by maximizing F1 on the full train set's PR curve
  - both are then applied to test EXACTLY ONCE, never re-tuned on test

Tested against: Python 3.12
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
)

RANDOM_STATE = 42
N_LAGS = 5
N_FEATS = 9
N_FOLDS = 5
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

np.random.seed(RANDOM_STATE)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# LOAD (same scaled files, same split as CNN/LSTM)
# ----------------------------------------------------------------------
def load_windows():
    log("=== Loading Step 3 SCALED outputs (read-only) ===")
    train_df = pd.read_csv("step3_train_windows_scaled.csv")
    test_df = pd.read_csv("step3_test_windows_scaled.csv")
    episode_summary = pd.read_csv("step3_episode_summary.csv",
                                   parse_dates=["start_minute", "end_minute"])

    base_feats = [
        "flow_count", "distinct_dst_ports", "distinct_dst_ips", "mean_flow_duration",
        "total_fwd_bytes", "total_bwd_bytes", "psh_rate", "mean_init_win_fwd", "frac_tcp",
    ]
    feat_cols = [f"t-{lag}_{f}" for lag in range(N_LAGS, 0, -1) for f in base_feats]

    missing = [c for c in feat_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing: {missing}")

    log(f"Train windows: {len(train_df)} | positives: {train_df['label'].sum()}")
    log(f"Test windows : {len(test_df)} | positives: {test_df['label'].sum()}")

    nan_check = train_df[feat_cols].isna().sum().sum() + test_df[feat_cols].isna().sum().sum()
    if nan_check > 0:
        raise AssertionError("NaNs present in Step 3 outputs — do not proceed.")

    return train_df, test_df, episode_summary, feat_cols, base_feats


# ----------------------------------------------------------------------
# EWMA COMPOSITE SCORE
# ----------------------------------------------------------------------
def compute_ewma_scores(df, feat_cols, base_feats, alpha):
    """
    For each window, per feature: weighted avg across lags t-5..t-1,
    weight w_i = alpha * (1-alpha)^i, i=0 for most recent (t-1) ... i=4 for t-5.
    Composite score = equal-weight mean across the 9 per-feature EWMA values.
    """
    N = len(df)
    weights = np.array([alpha * (1 - alpha) ** i for i in range(N_LAGS)])  # i=0..4
    weights = weights / weights.sum()  # normalize (finite window, not infinite series)

    # lag order in columns is t-5, t-4, t-3, t-2, t-1 (i.e. i=4,3,2,1,0)
    # reverse to align weights[0] (most recent, i=0) with t-1
    lag_order_ivals = [4, 3, 2, 1, 0]  # column order -> i value

    composite = np.zeros(N, dtype=np.float64)
    for f in base_feats:
        feat_matrix = np.zeros((N, N_LAGS), dtype=np.float64)
        for col_idx, lag in enumerate(range(N_LAGS, 0, -1)):  # 5,4,3,2,1
            col = f"t-{lag}_{f}"
            i_val = lag_order_ivals[col_idx]
            feat_matrix[:, i_val] = df[col].to_numpy(dtype=np.float64)
        # feat_matrix columns now ordered i=0..4 matching `weights`
        ewma_feat = feat_matrix @ weights
        composite += ewma_feat

    composite /= len(base_feats)  # equal-weight mean across 9 features
    return composite


# ----------------------------------------------------------------------
# ALPHA SELECTION — 5-fold CV on TRAIN only
# ----------------------------------------------------------------------
def select_alpha_via_cv(train_df, feat_cols, base_feats):
    log(f"\n=== {N_FOLDS}-fold stratified CV on TRAIN ONLY (alpha selection) ===")
    log("Test set is NOT used anywhere in this step.")

    y_train = train_df["label"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    alpha_scores = {}
    for alpha in ALPHA_GRID:
        fold_pr_aucs = []
        scores_full = compute_ewma_scores(train_df, feat_cols, base_feats, alpha)
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train), start=1):
            val_scores = scores_full[val_idx]
            y_val = y_train[val_idx]
            if y_val.sum() > 0:
                pr_auc = average_precision_score(y_val, val_scores)
            else:
                pr_auc = np.nan
            fold_pr_aucs.append(pr_auc)
        mean_pr_auc = np.nanmean(fold_pr_aucs)
        alpha_scores[alpha] = mean_pr_auc
        log(f"alpha={alpha}: mean CV val PR-AUC={mean_pr_auc:.4f}")

    best_alpha = max(alpha_scores, key=alpha_scores.get)
    log(f"\nSelected alpha={best_alpha} (mean CV val PR-AUC={alpha_scores[best_alpha]:.4f})")
    log("NOTE: ~3-4 positive samples per validation fold — treat as a rough anchor.")
    return best_alpha


# ----------------------------------------------------------------------
# THRESHOLD SELECTION — maximize F1 on full TRAIN set
# ----------------------------------------------------------------------
def select_threshold_on_train(train_df, feat_cols, base_feats, alpha):
    y_train = train_df["label"].to_numpy(dtype=int)
    scores = compute_ewma_scores(train_df, feat_cols, base_feats, alpha)

    precisions, recalls, thresholds = precision_recall_curve(y_train, scores)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0,
    )
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.0
    log(f"\nSelected threshold={best_threshold:.4f} "
        f"(train F1={f1s[best_idx]:.4f} at this threshold, train-only selection)")
    return best_threshold


# ----------------------------------------------------------------------
# FINAL EVAL — test touched exactly once
# ----------------------------------------------------------------------
def final_eval(test_df, feat_cols, base_feats, alpha, threshold):
    log(f"\n=== Final evaluation on test set (alpha={alpha}, threshold={threshold:.4f}) ===")
    y_test = test_df["label"].to_numpy(dtype=int)
    scores = compute_ewma_scores(test_df, feat_cols, base_feats, alpha)
    y_pred = (scores >= threshold).astype(int)

    n_pos = int(y_test.sum())
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, scores)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

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
    log(f"Accuracy        : {acc:.4f}  <-- NOT DIAGNOSTIC")

    log("\n--- SMALL-SAMPLE CAVEAT ---")
    log(f"Test set has only {n_pos} positive windows. Each single error moves recall by "
        f"{100/n_pos:.1f} percentage points.")

    metrics = {
        "alpha": alpha,
        "threshold": float(threshold),
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
            f"recall by {100/n_pos:.1f} percentage points."
        ),
    }
    return y_pred, scores, metrics


# ----------------------------------------------------------------------
# PR CURVE
# ----------------------------------------------------------------------
def save_pr_curve(y_test, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    pr_df = pd.DataFrame({
        "precision": precisions,
        "recall": recalls,
        "threshold": np.append(thresholds, np.nan),
    })
    pr_df.to_csv("step7_ewma_pr_curve.csv", index=False)
    log(f"\nSaved: step7_ewma_pr_curve.csv ({len(pr_df)} points)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.step(recalls, precisions, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("EWMA — Precision-Recall Curve (n_pos_test=8)")
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("step7_ewma_pr_curve.png", dpi=150)
        plt.close()
        log("Saved: step7_ewma_pr_curve.png")
    except ImportError:
        log("matplotlib not available — skipped PNG plot, CSV still saved.")


# ----------------------------------------------------------------------
# PER-EPISODE DETECTION
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
            status = "NO POSITIVE WINDOWS FOUND"
        elif n_detected == n_positive:
            status = "FULLY DETECTED"
        elif n_detected > 0:
            status = "PARTIALLY DETECTED"
        else:
            status = "MISSED ENTIRELY"

        missed = ep_windows[ep_windows["pred"] == 0]["target_minute"].tolist()
        log(f"Episode {ep_id} [{start} - {end}]: {n_detected}/{n_positive} detected — {status}")
        if missed:
            log(f"  Missed minutes: {missed}")

        rows.append({
            "episode_id": ep_id, "start_minute": start, "end_minute": end,
            "n_positive_minutes": n_positive, "n_detected": n_detected, "status": status,
        })

    pd.DataFrame(rows).to_csv("step7_ewma_per_episode_detection.csv", index=False)
    log("\nSaved: step7_ewma_per_episode_detection.csv")


# ----------------------------------------------------------------------
# FULL COMPARISON — RF, CNN, LSTM, EWMA + explicit LSTM-vs-EWMA verdict
# ----------------------------------------------------------------------
def compare_all_models(ewma_metrics):
    log("\n=== RF vs CNN vs LSTM vs EWMA comparison ===")
    files = {
        "rf": "step4_rf_metrics.json",
        "cnn": "step5_cnn_metrics.json",
        "lstm": "step6_lstm_metrics.json",
    }
    loaded = {}
    for name, path in files.items():
        try:
            with open(path) as f:
                loaded[name] = json.load(f)
        except FileNotFoundError:
            log(f"{path} not found — skipping from comparison.")

    loaded["ewma"] = ewma_metrics

    rows = []
    for key in ["precision", "recall", "f1", "pr_auc", "balanced_accuracy"]:
        row = {"metric": key}
        for name in ["rf", "cnn", "lstm", "ewma"]:
            if name in loaded:
                row[name] = round(loaded[name][key], 4)
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    log(comp_df.to_string(index=False))
    comp_df.to_csv("step7_all_models_comparison.csv", index=False)
    log("\nSaved: step7_all_models_comparison.csv")

    # --- THE ACTUAL GATING RULE: LSTM must beat EWMA ---
    log("\n--- LSTM vs EWMA (the locked architecture's real gating rule) ---")
    if "lstm" not in loaded:
        log("step6_lstm_metrics.json not found — cannot evaluate this rule.")
        return

    lstm_m = loaded["lstm"]
    n_pos = ewma_metrics["n_test_positive"]
    noise_floor = 100 / n_pos

    for key, label in [("pr_auc", "PR-AUC"), ("recall", "Recall"),
                        ("precision", "Precision"), ("f1", "F1"),
                        ("balanced_accuracy", "Balanced Acc.")]:
        ewma_v, lstm_v = ewma_metrics[key], lstm_m[key]
        direction = "up" if lstm_v > ewma_v else ("down" if lstm_v < ewma_v else "flat")
        log(f"{label:14s}: EWMA={ewma_v:.4f}  LSTM={lstm_v:.4f}  (LSTM vs EWMA: {direction})")

    log(f"\nSample noise floor at n_pos_test={n_pos}: ~{noise_floor:.1f}pp per single error.")
    log("Read every line above before concluding. Per the locked rule, LSTM becomes a")
    log("primary component only if it clearly beats EWMA — not RF, not CNN. If LSTM's")
    log("advantage here is smaller than the noise floor, or trades away precision/recall")
    log("the way CNN traded against RF, state that explicitly rather than declaring a")
    log("single winner.")


def save_predictions(test_df, y_pred, scores):
    out = test_df[["source_ip", "target_minute", "label"]].copy()
    out["predicted_label"] = y_pred
    out["ewma_score"] = scores
    out.to_csv("step7_ewma_predictions.csv", index=False)
    log("Saved: step7_ewma_predictions.csv")


def save_metrics(metrics):
    with open("step7_ewma_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved: step7_ewma_metrics.json")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual Benign", "Actual PortScan"],
        columns=["Predicted Benign", "Predicted PortScan"],
    )
    cm_df.to_csv("step7_ewma_confusion_matrix.csv")
    log("Saved: step7_ewma_confusion_matrix.csv")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    train_df, test_df, episode_summary, feat_cols, base_feats = load_windows()

    best_alpha = select_alpha_via_cv(train_df, feat_cols, base_feats)
    best_threshold = select_threshold_on_train(train_df, feat_cols, base_feats, best_alpha)

    y_test = test_df["label"].to_numpy(dtype=int)
    y_pred, scores, metrics = final_eval(test_df, feat_cols, base_feats, best_alpha, best_threshold)

    save_metrics(metrics)
    save_predictions(test_df, y_pred, scores)
    save_pr_curve(y_test, scores)
    per_episode_detection(test_df, y_pred, episode_summary)
    compare_all_models(metrics)

    log("\n=== STEP 7 (EWMA BASELINE) COMPLETE ===")
    log("Test set was touched exactly once, at final evaluation. Alpha and threshold")
    log("were both selected using TRAIN data only.")
    log("This is the baseline LSTM must beat to be considered a primary component —")
    log("review the LSTM vs EWMA section above before deciding on fusion.")


if __name__ == "__main__":
    main()
