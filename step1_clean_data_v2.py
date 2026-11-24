"""
Step 1 (CORRECTED) -- CICIDS2017 Friday PortScan: data loading and cleaning ONLY.

Fixes the v1 bug where zero-duration rows were silently dropped by a
row-wise Inf removal step despite the stated intent to keep them.

Policy in this version:
  - Exact duplicates: dropped.
  - Negative Flow Duration: dropped (physically invalid).
  - Zero Flow Duration: KEPT. Their Inf/NaN rate-derived cells
    (Flow Bytes/s, Flow Packets/s, etc.) are repaired to 0, and an
    explicit 'Flow_Duration_Zero_Flag' column is added.
  - Any row with NaN/Inf that is NOT explained by Flow Duration == 0
    is treated as a true anomaly and dropped.
  - Init_Win_bytes_forward == -1 is a legitimate CICFlowMeter sentinel,
    left untouched.

Usage (PowerShell):
    python step1_clean_data_v2.py --input "C:\\path\\to\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --output "C:\\path\\to\\cicids2017_portscan_step1_clean_v2.csv"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[step1] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean CICIDS2017 Friday PortScan CSV (v2: keeps zero-duration rows).")
    parser.add_argument("--input", required=True, help="Path to the ORIGINAL CSV (read-only).")
    parser.add_argument("--output", required=True, help="Path to write the cleaned CSV to.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        log(f"ERROR: input file not found: {in_path}")
        sys.exit(1)
    if out_path.exists():
        log(f"ERROR: refusing to overwrite existing output file: {out_path}")
        sys.exit(1)

    log(f"Reading (read-only): {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    n0 = len(df)
    log(f"Raw shape: {n0} rows x {df.shape[1]} cols")

    if "Flow Duration" not in df.columns:
        log("ERROR: 'Flow Duration' column not found after stripping headers.")
        sys.exit(1)

    # --- 1. Exact duplicates ----------------------------------------------------
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()
    log(f"Exact duplicate rows dropped: {n_dup}")

    # --- 2. Negative Flow Duration (physically invalid) --------------------------
    neg_mask = df["Flow Duration"] < 0
    n_neg = int(neg_mask.sum())
    df = df.loc[~neg_mask].copy()
    log(f"Negative Flow Duration rows dropped: {n_neg}")

    # --- 3. Identify zero-duration rows -- these are KEPT -------------------------
    zero_mask = (df["Flow Duration"] == 0).to_numpy()
    n_zero = int(zero_mask.sum())
    log(f"Zero-duration rows identified (policy: KEEP): {n_zero}")

    if "Label" in df.columns:
        zero_label_counts = df.loc[df["Flow Duration"] == 0, "Label"].value_counts().to_dict()
        log(f"Zero-duration row label breakdown: {zero_label_counts}")

    # --- 4. Locate Inf/NaN cells and separate explained vs unexplained -----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cells = np.isinf(df[numeric_cols].to_numpy())
    nan_cells = df[numeric_cols].isna().to_numpy()

    bad_row_any = inf_cells.any(axis=1) | nan_cells.any(axis=1)
    explainable_mask = bad_row_any & zero_mask
    unexplained_mask = bad_row_any & (~zero_mask)

    n_explainable = int(explainable_mask.sum())
    n_unexplained = int(unexplained_mask.sum())
    log(f"Inf/NaN rows explained by Flow Duration == 0: {n_explainable} (repairing, not dropping)")
    log(f"Inf/NaN rows NOT explained by zero duration (true anomalies): {n_unexplained} (dropping)")

    # --- 5. Repair explainable rows: Inf -> NaN -> 0 in rate-derived cells only --
    df["Flow_Duration_Zero_Flag"] = zero_mask.astype(int)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    zero_idx = df.index[df["Flow Duration"] == 0]
    df.loc[zero_idx, numeric_cols] = df.loc[zero_idx, numeric_cols].fillna(0)

    # --- 6. Drop rows where NaN/Inf remains (true, unexplained anomalies) --------
    remaining_bad_mask = df[numeric_cols].isna().any(axis=1)
    n_dropped_unexplained = int(remaining_bad_mask.sum())
    df = df.loc[~remaining_bad_mask].copy()
    log(f"Dropped {n_dropped_unexplained} row(s) with unexplained NaN/Inf (duration > 0).")

    # --- 7. Init_Win_bytes_forward/backward == -1: sentinel, untouched -----------
    for col in ("Init_Win_bytes_forward", "Init_Win_bytes_backward"):
        if col in df.columns:
            n_sentinel = int((df[col] == -1).sum())
            log(f"Sentinel -1 in '{col}': {n_sentinel} row(s) -- left as -1, not modified.")

    # --- Final checks --------------------------------------------------------------
    n_final = len(df)
    remaining_zero = int((df["Flow Duration"] == 0).sum())
    log(f"Final shape: {n_final} rows x {df.shape[1]} cols")
    log(f"Zero-duration rows present in FINAL cleaned data: {remaining_zero} (should equal {n_zero})")

    still_bad = int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()
                     + df.select_dtypes(include=[np.number]).isna().to_numpy().sum())
    log(f"Post-clean check -> remaining Inf/NaN numeric cells: {still_bad} (should be 0)")

    if "Label" in df.columns:
        log(f"Final label counts:\n{df['Label'].value_counts()}")

    df.to_csv(out_path, index=False)
    log(f"Wrote cleaned CSV to: {out_path}")
    log("Done. Original input file was not modified.")


if __name__ == "__main__":
    main()
