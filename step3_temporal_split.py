"""
Step 3 — Temporal Windowing + Leakage-Safe Episode Split (CORRECTED)
CICIDS2017 PortScan / Adaptive MTD project

Input : cicids2017_portscan_step2_buckets.csv
Output: step3_train_windows.csv, step3_test_windows.csv,
        step3_train_windows_scaled.csv, step3_test_windows_scaled.csv,
        step3_scaler.pkl, step3_episode_summary.csv

FIX (this version): Step 2 leaves `mean_init_win_fwd` as NaN for the 262
buckets where every Init_Win_bytes_forward value was -1 (excluded) and no
valid value remained. That NaN was flowing straight through the per-IP
minute lookup into every downstream window/CSV. Fixed at the single
source point (build_ip_minute_lookup) so ALL FOUR output files inherit
the fix — not patched later in scaling, saving, or Step 5.

Locked design (unchanged from prior approval):
  - Episode def: positive minutes exactly 1 min apart = same episode; gap > 1 min = new episode
  - W = 5 previous COMPLETED minutes [t-5 ... t-1], target minute t excluded from its own input
  - Zero-fill missing historical MINUTES (whole minute absent) per source IP
  - NaN-fill (this fix) for individual features present in a real minute but
    undefined at Step 2 (mean_init_win_fwd only, so far) -> converted to 0.0
  - Attack episodes 1-6 -> train, episodes 7-9 -> test (chronological)
  - Benign source IPs: whole-IP split, stratified by per-IP activity level
  - Attacker IP allowed in both splits (chronological design, not a leak)
  - Scaler fit on train only

Tested against: Python 3.14.5, pandas 3.0.5, numpy 2.5.1
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_FILE = "cicids2017_portscan_step2_buckets.csv"
OUT_DIR = "."

W = 5
ATTACKER_IP_HINT = "172.16.0.1"

LOCKED_FEATURES = [
    "flow_count",
    "distinct_dst_ports",
    "distinct_dst_ips",
    "mean_flow_duration",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "psh_rate",
    "mean_init_win_fwd",
    "frac_tcp",
]

RANDOM_STATE = 42
BENIGN_TEST_FRACTION = 1 / 3

KNOWN_EPISODES = [
    ("01:05", "01:06"), ("01:52", "01:52"), ("01:55", "02:00"),
    ("02:14", "02:16"), ("02:51", "02:56"), ("03:03", "03:03"),
    ("03:08", "03:10"), ("03:12", "03:13"), ("03:21", "03:23"),
]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# STEP 0 — Load + detect columns
# ----------------------------------------------------------------------
def detect_column(df, candidates, purpose):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not detect column for [{purpose}]. Tried {candidates}. "
        f"Actual columns present: {list(df.columns)}"
    )


def load_data():
    log(f"\n=== STEP 0: Loading {INPUT_FILE} ===")
    df = pd.read_csv(f"{OUT_DIR}/{INPUT_FILE}")
    log(f"Loaded shape: {df.shape}")

    source_col = detect_column(df, ["Source IP", "Src IP", "source_ip", "SourceIP", "src_ip"], "source IP")
    minute_col = detect_column(df, ["minute", "Minute", "minute_timestamp", "Bucket_Minute", "timestamp"], "minute timestamp")
    label_col = detect_column(df, ["PortScan", "is_portscan", "Label", "label", "portscan_flag", "bucket_label"], "label")

    missing_feats = [f for f in LOCKED_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing locked feature columns: {missing_feats}")

    log(f"Detected source IP column : '{source_col}'")
    log(f"Detected minute column    : '{minute_col}'")
    log(f"Detected label column     : '{label_col}'")

    # --- diagnostic: confirm NaN root cause before fixing ---
    nan_report = df[LOCKED_FEATURES].isna().sum()
    nan_report = nan_report[nan_report > 0]
    if len(nan_report):
        log(f"\nNaN check on raw Step 2 input (pre-fix):")
        log(nan_report.to_string())
        log(f"Total NaN cells in Step 2 input: {int(df[LOCKED_FEATURES].isna().sum().sum())}")
    else:
        log("No NaNs found in Step 2 input feature columns.")

    df[minute_col] = pd.to_datetime(df[minute_col])
    df = df.sort_values([source_col, minute_col]).reset_index(drop=True)

    if df[label_col].dtype == object:
        df["_label_bin"] = df[label_col].astype(str).str.upper().str.contains("PORTSCAN").astype(int)
    else:
        df["_label_bin"] = (df[label_col] != 0).astype(int)

    return df, source_col, minute_col, label_col


# ----------------------------------------------------------------------
# STEP 1 — Episodes
# ----------------------------------------------------------------------
def identify_episodes(df, source_col, minute_col):
    log("\n=== STEP 1: Episode identification ===")
    attacker_ips = df.loc[df["_label_bin"] == 1, source_col].unique()
    if len(attacker_ips) != 1:
        log(f"WARNING: expected exactly 1 attacker IP, found {len(attacker_ips)}: {attacker_ips}")
    attacker_ip = attacker_ips[0] if len(attacker_ips) else ATTACKER_IP_HINT
    log(f"Attacker source IP: {attacker_ip}")

    pos = df[(df[source_col] == attacker_ip) & (df["_label_bin"] == 1)].sort_values(minute_col)
    minutes = pos[minute_col].tolist()
    if len(minutes) != 27:
        log(f"WARNING: expected 27 positive buckets, found {len(minutes)}")

    episodes = []
    current_ep = [minutes[0]]
    for prev, cur in zip(minutes[:-1], minutes[1:]):
        gap = (cur - prev).total_seconds() / 60.0
        if gap <= 1.0:
            current_ep.append(cur)
        else:
            episodes.append(current_ep)
            current_ep = [cur]
    episodes.append(current_ep)

    log(f"Episodes found: {len(episodes)} (expected 9)")
    for i, ep in enumerate(episodes, start=1):
        start_str, end_str = ep[0].strftime("%H:%M"), ep[-1].strftime("%H:%M")
        expected = KNOWN_EPISODES[i - 1] if i - 1 < len(KNOWN_EPISODES) else ("?", "?")
        flag = "OK" if (start_str, end_str) == expected else "CHECK MANUALLY"
        log(f"  Episode {i}: {start_str}-{end_str} (n={len(ep)}) expected {expected[0]}-{expected[1]} [{flag}]")

    if len(episodes) != 9:
        raise AssertionError(f"Episode count mismatch: got {len(episodes)}, expected 9.")

    total_pos = sum(len(e) for e in episodes)
    if total_pos != 27:
        log(f"WARNING: total positive minutes = {total_pos}, expected 27")

    return attacker_ip, episodes


def attacker_split_cutoff(episodes):
    cutoff = episodes[6][0]  # start of episode 7
    log(f"\n=== STEP 2: Attacker split cutoff = {cutoff} (start of episode 7) ===")
    log("Attacker buckets < cutoff -> train, >= cutoff -> test (chronological).")
    return cutoff


# ----------------------------------------------------------------------
# STEP 3 — Benign split
# ----------------------------------------------------------------------
def split_benign_ips(df, source_col, attacker_ip):
    log("\n=== STEP 3: Benign source IP split (stratified by activity level) ===")
    benign = df[df[source_col] != attacker_ip]
    activity = benign.groupby(source_col).size().rename("bucket_count").reset_index()

    try:
        activity["activity_bin"] = pd.qcut(activity["bucket_count"], q=3, labels=["low", "med", "high"], duplicates="drop")
    except ValueError:
        activity["activity_bin"] = "all"

    log(f"Total benign source IPs: {len(activity)}")

    train_ips, test_ips = train_test_split(
        activity, test_size=BENIGN_TEST_FRACTION,
        stratify=activity["activity_bin"], random_state=RANDOM_STATE,
    )
    train_ip_set, test_ip_set = set(train_ips[source_col]), set(test_ips[source_col])

    overlap = train_ip_set & test_ip_set
    if overlap:
        raise AssertionError(f"Benign IP leak detected — overlap: {overlap}")

    log(f"Benign IPs -> train: {len(train_ip_set)}, test: {len(test_ip_set)} (no overlap confirmed)")
    return train_ip_set, test_ip_set


# ----------------------------------------------------------------------
# STEP 4 — Per-IP minute lookup  ***FIX APPLIED HERE***
# ----------------------------------------------------------------------
def build_ip_minute_lookup(df, source_col, minute_col):
    log("\n=== STEP 4: Building per-IP minute lookup (NaN->0 fix applied here) ===")
    lookup = {}
    nan_replaced_count = 0

    for ip, grp in df.groupby(source_col):
        d = {}
        for _, row in grp.iterrows():
            vec = row[LOCKED_FEATURES].to_numpy(dtype=np.float32)
            n_nan = int(np.isnan(vec).sum())
            if n_nan > 0:
                nan_replaced_count += n_nan
                vec = np.nan_to_num(vec, nan=0.0)  # <-- FIX: NaN -> 0 at source
            d[row[minute_col]] = vec
        lookup[ip] = d

    log(f"Total NaN feature values replaced with 0.0 at source: {nan_replaced_count}")
    log("This lookup is the single point every downstream window/CSV is built from, "
        "so the fix propagates to all four output files.")
    return lookup


# ----------------------------------------------------------------------
# STEP 5 — Build windows
# ----------------------------------------------------------------------
def build_windows(df, source_col, minute_col, ip_lookup, attacker_ip, cutoff, benign_train_ips):
    log("\n=== STEP 5: Building windows (W=5, zero-filled missing minutes) ===")
    n_feat = len(LOCKED_FEATURES)
    zero_vec = np.zeros(n_feat, dtype=np.float32)

    rows = []
    zero_fill_count = 0
    total_cells = 0

    for _, row in df.iterrows():
        ip = row[source_col]
        t = row[minute_col]
        label = int(row["_label_bin"])

        if ip == attacker_ip:
            split = "train" if t < cutoff else "test"
        else:
            split = "train" if ip in benign_train_ips else "test"

        feats = []
        for lag in range(W, 0, -1):
            t_lag = t - pd.Timedelta(minutes=lag)
            vec = ip_lookup.get(ip, {}).get(t_lag, None)
            total_cells += 1
            if vec is None:
                vec = zero_vec
                zero_fill_count += 1
            feats.append(vec)

        flat = np.concatenate(feats)
        rows.append({
            "source_ip": ip, "target_minute": t, "label": label, "split": split,
            **{f"t-{lag}_{f}": v for lag in range(W, 0, -1)
               for f, v in zip(LOCKED_FEATURES, feats[W - lag])},
        })

    out = pd.DataFrame(rows)
    zero_fill_rate = 100.0 * zero_fill_count / total_cells
    log(f"Total windows built: {len(out)}")
    log(f"History cells zero-filled (missing minute): {zero_fill_count}/{total_cells} ({zero_fill_rate:.1f}%)")

    # explicit post-build NaN check (should already be 0 due to Step 4 fix)
    remaining_nans = int(out.isna().sum().sum())
    log(f"NaN check immediately after window build: {remaining_nans}")
    if remaining_nans > 0:
        nan_cols = out.isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        raise AssertionError(f"Unexpected NaNs remain after fix:\n{nan_cols.to_string()}")

    return out


# ----------------------------------------------------------------------
# STEP 6 — Split, scale, save (with mandatory NaN==0 verification per file)
# ----------------------------------------------------------------------
def verify_no_nan(df, filename):
    total_nan = int(df.isna().sum().sum())
    check_passed = (total_nan == 0)
    log(f"  Verify {filename}: df.isna().sum().sum() == 0  ->  {check_passed} (total NaN = {total_nan})")
    if not check_passed:
        raise AssertionError(f"{filename} still contains {total_nan} NaN values — fix did not fully propagate.")


def finalize_and_save(windows_df, attacker_ip):
    log("\n=== STEP 6: Final split integrity check + scaling + NaN verification ===")

    train_df = windows_df[windows_df["split"] == "train"].reset_index(drop=True)
    test_df = windows_df[windows_df["split"] == "test"].reset_index(drop=True)

    train_ips, test_ips = set(train_df["source_ip"]), set(test_df["source_ip"])
    overlap = (train_ips & test_ips) - {attacker_ip}
    if overlap:
        raise AssertionError(f"LEAK: benign source IP(s) in both splits: {list(overlap)[:5]}...")
    log(f"Confirmed: 0 benign source IPs overlap between train and test.")
    log(f"Attacker IP present in both splits (expected, chronological design): "
        f"{attacker_ip in train_ips and attacker_ip in test_ips}")
    log(f"Train windows: {len(train_df)} (positives={train_df['label'].sum()})")
    log(f"Test windows : {len(test_df)} (positives={test_df['label'].sum()})")

    feat_cols = [c for c in windows_df.columns if c not in ("source_ip", "target_minute", "label", "split")]

    train_df.to_csv(f"{OUT_DIR}/step3_train_windows.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/step3_test_windows.csv", index=False)
    log("Saved: step3_train_windows.csv, step3_test_windows.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    X_test_scaled = scaler.transform(test_df[feat_cols].to_numpy(dtype=np.float32))

    train_scaled_df = train_df.copy()
    train_scaled_df[feat_cols] = X_train_scaled
    test_scaled_df = test_df.copy()
    test_scaled_df[feat_cols] = X_test_scaled

    train_scaled_df.to_csv(f"{OUT_DIR}/step3_train_windows_scaled.csv", index=False)
    test_scaled_df.to_csv(f"{OUT_DIR}/step3_test_windows_scaled.csv", index=False)
    log("Saved: step3_train_windows_scaled.csv, step3_test_windows_scaled.csv")

    joblib.dump(scaler, f"{OUT_DIR}/step3_scaler.pkl")
    log("Saved: step3_scaler.pkl (fit on train only)")

    # --- mandatory verification: df.isna().sum().sum() == 0 for all four files ---
    log("\n--- Final NaN verification (all four output files) ---")
    verify_no_nan(train_df, "step3_train_windows.csv")
    verify_no_nan(test_df, "step3_test_windows.csv")
    verify_no_nan(train_scaled_df, "step3_train_windows_scaled.csv")
    verify_no_nan(test_scaled_df, "step3_test_windows_scaled.csv")
    log("All four files verified: 0 NaN values.")

    return train_df, test_df


def save_episode_summary(episodes, cutoff):
    rows = []
    for i, ep in enumerate(episodes, start=1):
        split = "train" if i <= 6 else "test"
        rows.append({
            "episode_id": i, "start_minute": ep[0], "end_minute": ep[-1],
            "n_positive_minutes": len(ep), "split": split,
        })
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/step3_episode_summary.csv", index=False)
    log(f"\nSaved: step3_episode_summary.csv")
    log(f"Attacker chronological cutoff for train/test: {cutoff}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    df, source_col, minute_col, label_col = load_data()
    attacker_ip, episodes = identify_episodes(df, source_col, minute_col)
    cutoff = attacker_split_cutoff(episodes)
    benign_train_ips, benign_test_ips = split_benign_ips(df, source_col, attacker_ip)

    ip_lookup = build_ip_minute_lookup(df, source_col, minute_col)

    windows_df = build_windows(df, source_col, minute_col, ip_lookup, attacker_ip, cutoff, benign_train_ips)

    train_df, test_df = finalize_and_save(windows_df, attacker_ip)
    save_episode_summary(episodes, cutoff)

    log("\n=== STEP 3 COMPLETE (NaN fix applied at source) ===")
    log(f"Train: {len(train_df)} windows | Test: {len(test_df)} windows")
    log("Rerun Step 4 (RF) and Step 5 (CNN) against these regenerated files.")


if __name__ == "__main__":
    main()
