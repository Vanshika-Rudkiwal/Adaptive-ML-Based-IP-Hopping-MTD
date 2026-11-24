"""
Step 6 — LSTM Baseline
CICIDS2017 PortScan / Adaptive MTD project

Reads (read-only, never modified):
  step3_train_windows_scaled.csv
  step3_test_windows_scaled.csv
  step3_episode_summary.csv
  step4_rf_metrics.json   (comparison only)
  step5_cnn_metrics.json  (comparison only)

Writes (new files only):
  step6_lstm_metrics.json
  step6_lstm_confusion_matrix.csv
  step6_lstm_pr_curve.csv
  step6_lstm_per_episode_detection.csv
  step6_lstm_predictions.csv
  step6_vs_rf_cnn_comparison.csv

Hard rules (same as Step 5):
  - Test set touched EXACTLY ONCE, at final evaluation.
  - Epoch selection via 5-fold stratified CV on TRAIN only.
  - Decision threshold fixed at 0.5, never tuned on test.
  - Same features/split/windows as RF and CNN — scaled CSVs used (LSTM,
    like CNN, needs scaled inputs; RF did not).

Note on architecture context: per the locked project design, LSTM's
real bar is EWMA (statistical trend baseline), not RF/CNN — LSTM must
beat EWMA to be treated as a primary component. EWMA has not been
built yet in this step sequence. This script still reports RF/CNN
comparison as requested, but the "primary component" question stays
open until EWMA exists.

Verdict-logic note: Step 5's RF-vs-CNN verdict block had a messaging
bug — a binary AND-gate (PR-AUC gain AND F1 >= RF) collapsed a
precision/recall trade-off into a false "no improvement" statement
when PR-AUC had, in fact, improved substantially. This script reports
each metric's delta explicitly instead of folding them into one
pass/fail sentence, so a genuine trade-off is visible rather than
hidden by whichever metric the AND-gate happened to fail on.

Requires: pip install torch scikit-learn pandas numpy --break-system-packages
Tested against: Python 3.12, pandas/numpy per your environment
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
MAX_EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_SIZE = 32

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# LOAD (same scaled files as CNN, same split)
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
    log(f"Feature columns confirmed: {len(feat_cols)} (5 lags x 9 features, ordered)")

    nan_check = train_df[feat_cols].isna().sum().sum() + test_df[feat_cols].isna().sum().sum()
    log(f"NaN check on loaded Step 3 outputs: {int(nan_check)} (expect 0, per Step 3 fix)")
    if nan_check > 0:
        raise AssertionError("NaNs present in Step 3 outputs — do not proceed, re-check Step 3.")

    return train_df, test_df, episode_summary, feat_cols


def to_lstm_tensor(df, feat_cols):
    # (N, 45) -> (N, seq_len=5, input_size=9) — LSTM wants (batch, seq, features),
    # no channel-first transpose needed (unlike the CNN's Conv1d layout).
    X = df[feat_cols].to_numpy(dtype=np.float32).reshape(-1, N_LAGS, N_FEATS)
    return X


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
class TinyLSTM(nn.Module):
    def __init__(self, input_size=N_FEATS, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch, seq_len=5, input_size=9)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (batch, hidden_size) — final hidden state
        z = self.relu(self.fc1(self.dropout(h_last)))
        return self.fc2(z).squeeze(-1)  # logits


# ----------------------------------------------------------------------
# TRAIN LOOP (shared by CV and final fit)
# ----------------------------------------------------------------------
def train_one_model(X_train, y_train, X_val, y_val, n_epochs, verbose=False):
    device = torch.device("cpu")
    model = TinyLSTM().to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_pr_auc_history = []
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32))
                val_proba = torch.sigmoid(val_logits).numpy()
            pr_auc = average_precision_score(y_val, val_proba) if y_val.sum() > 0 else np.nan
            val_pr_auc_history.append(pr_auc)
            if verbose and (epoch + 1) % 10 == 0:
                log(f"    epoch {epoch+1}/{n_epochs}  val_PR-AUC={pr_auc:.4f}")

    return model, val_pr_auc_history


# ----------------------------------------------------------------------
# CV — epoch selection on TRAIN only
# ----------------------------------------------------------------------
def select_epochs_via_cv(X_train, y_train):
    log(f"\n=== {N_FOLDS}-fold stratified CV on TRAIN ONLY (epoch selection) ===")
    log("Test set is NOT used anywhere in this step.")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_curves = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        log(f"Fold {fold_i}: train_pos={y_tr.sum()}, val_pos={y_val.sum()}")

        _, val_curve = train_one_model(X_tr, y_tr, X_val, y_val, MAX_EPOCHS)
        fold_curves.append(val_curve)

    fold_curves = np.array(fold_curves)
    mean_curve = np.nanmean(fold_curves, axis=0)
    best_epoch = int(np.nanargmax(mean_curve)) + 1
    log(f"\nBest epoch count by mean CV PR-AUC: {best_epoch} "
        f"(mean val PR-AUC={mean_curve[best_epoch-1]:.4f})")
    log("NOTE: ~3-4 positive samples per validation fold — treat as a rough anchor, not a precise optimum.")

    return best_epoch


# ----------------------------------------------------------------------
# FINAL FIT — train on all of train, evaluate on test exactly once
# ----------------------------------------------------------------------
def final_fit_and_eval(X_train, y_train, X_test, y_test, n_epochs):
    log(f"\n=== Final fit on full train set ({n_epochs} epochs), single test evaluation ===")
    model, _ = train_one_model(X_train, y_train, None, None, n_epochs, verbose=True)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_proba = torch.sigmoid(logits).numpy()
    y_pred = (y_proba >= 0.5).astype(int)

    n_pos = int(y_test.sum())
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)
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
        "n_epochs_used": n_epochs,
        "n_test_windows": len(y_test),
        "n_test_positive": n_pos,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(bal_acc),
        "accuracy_NOT_DIAGNOSTIC": float(acc),
        "decision_threshold": 0.5,
        "caveat": (
            f"Test set has only {n_pos} positive windows; each single error moves "
            f"recall by {100/n_pos:.1f} percentage points."
        ),
    }
    return model, y_pred, y_proba, metrics


# ----------------------------------------------------------------------
# PR CURVE
# ----------------------------------------------------------------------
def save_pr_curve(y_test, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    pr_df = pd.DataFrame({
        "precision": precisions,
        "recall": recalls,
        "threshold": np.append(thresholds, np.nan),
    })
    pr_df.to_csv("step6_lstm_pr_curve.csv", index=False)
    log(f"\nSaved: step6_lstm_pr_curve.csv ({len(pr_df)} points)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.step(recalls, precisions, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("LSTM — Precision-Recall Curve (n_pos_test=8)")
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("step6_lstm_pr_curve.png", dpi=150)
        plt.close()
        log("Saved: step6_lstm_pr_curve.png")
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

    pd.DataFrame(rows).to_csv("step6_lstm_per_episode_detection.csv", index=False)
    log("\nSaved: step6_lstm_per_episode_detection.csv")


# ----------------------------------------------------------------------
# RF / CNN / LSTM COMPARISON — corrected verdict logic (no collapsing AND-gate)
# ----------------------------------------------------------------------
def compare_to_rf_and_cnn(lstm_metrics):
    log("\n=== RF vs CNN vs LSTM comparison ===")
    try:
        with open("step4_rf_metrics.json") as f:
            rf_metrics = json.load(f)
        with open("step5_cnn_metrics.json") as f:
            cnn_metrics = json.load(f)
    except FileNotFoundError as e:
        log(f"Could not load prior metrics ({e}) — skipping comparison.")
        return

    rows = []
    for key in ["precision", "recall", "f1", "pr_auc", "balanced_accuracy"]:
        rows.append({
            "metric": key,
            "rf": round(rf_metrics[key], 4),
            "cnn": round(cnn_metrics[key], 4),
            "lstm": round(lstm_metrics[key], 4),
            "delta_lstm_minus_rf": round(lstm_metrics[key] - rf_metrics[key], 4),
            "delta_lstm_minus_cnn": round(lstm_metrics[key] - cnn_metrics[key], 4),
        })
    comp_df = pd.DataFrame(rows)
    log(comp_df.to_string(index=False))
    comp_df.to_csv("step6_vs_rf_cnn_comparison.csv", index=False)
    log("\nSaved: step6_vs_rf_cnn_comparison.csv")

    n_pos = lstm_metrics["n_test_positive"]
    noise_floor = 100 / n_pos
    log(f"\nSample noise floor at n_pos_test={n_pos}: ~{noise_floor:.1f}pp per single error.")

    # --- corrected verdict: report each metric's movement explicitly instead
    #     of collapsing multiple metrics into one AND-gated true/false sentence,
    #     which is the exact bug found in Step 5's comparison block. ---
    log("\n--- VERDICT (explicit per-metric, not collapsed) ---")
    for key, label in [("pr_auc", "PR-AUC"), ("recall", "Recall"),
                        ("precision", "Precision"), ("f1", "F1")]:
        rf_v, cnn_v, lstm_v = rf_metrics[key], cnn_metrics[key], lstm_metrics[key]
        vs_rf = "up" if lstm_v > rf_v else ("down" if lstm_v < rf_v else "flat")
        vs_cnn = "up" if lstm_v > cnn_v else ("down" if lstm_v < cnn_v else "flat")
        log(f"{label:10s}: RF={rf_v:.4f}  CNN={cnn_v:.4f}  LSTM={lstm_v:.4f}  "
            f"(LSTM vs RF: {vs_rf}, vs CNN: {vs_cnn})")

    log("\nInterpretation guidance (read all lines above before concluding — do not")
    log("reduce this to a single pass/fail flag): if LSTM improves PR-AUC but at the")
    log("cost of precision (same pattern seen with CNN vs RF), that is a genuine")
    log("trade-off, not 'no improvement.' State the trade-off explicitly rather than")
    log("picking one metric to declare a winner.")

    log("\nREMINDER: per the locked architecture, LSTM's real qualifying bar is EWMA")
    log("(statistical trend baseline), which has not been built yet in this sequence.")
    log("This RF/CNN comparison is informative but does NOT by itself determine")
    log("whether LSTM becomes a primary component — that still requires the EWMA run.")


def save_predictions(test_df, y_pred, y_proba):
    out = test_df[["source_ip", "target_minute", "label"]].copy()
    out["predicted_label"] = y_pred
    out["predicted_proba"] = y_proba
    out.to_csv("step6_lstm_predictions.csv", index=False)
    log("Saved: step6_lstm_predictions.csv")


def save_metrics(metrics):
    with open("step6_lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved: step6_lstm_metrics.json")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual Benign", "Actual PortScan"],
        columns=["Predicted Benign", "Predicted PortScan"],
    )
    cm_df.to_csv("step6_lstm_confusion_matrix.csv")
    log("Saved: step6_lstm_confusion_matrix.csv")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    train_df, test_df, episode_summary, feat_cols = load_windows()

    X_train = to_lstm_tensor(train_df, feat_cols)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = to_lstm_tensor(test_df, feat_cols)
    y_test = test_df["label"].to_numpy(dtype=int)

    log(f"\nLSTM input shape: train={X_train.shape}, test={X_test.shape} "
        f"(N, seq_len=5, input_size=9)")

    best_epochs = select_epochs_via_cv(X_train, y_train)
    model, y_pred, y_proba, metrics = final_fit_and_eval(
        X_train, y_train, X_test, y_test, best_epochs
    )

    save_metrics(metrics)
    save_predictions(test_df, y_pred, y_proba)
    save_pr_curve(y_test, y_proba)
    per_episode_detection(test_df, y_pred, episode_summary)
    compare_to_rf_and_cnn(metrics)

    log("\n=== STEP 6 (LSTM BASELINE) COMPLETE ===")
    log("Test set was touched exactly once, at final evaluation. No threshold or")
    log("hyperparameter was tuned using test performance.")
    log("EWMA baseline still outstanding — required before LSTM can be called primary.")


if __name__ == "__main__":
    main()
