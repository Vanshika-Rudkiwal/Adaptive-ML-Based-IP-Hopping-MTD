"""
Step 2 -- Build per-(Source IP, minute) 9-feature buckets from the
Step 1 (v2) cleaned flow-level CSV.

Does NOT do windowing, train/test splitting, or any leakage-sensitive
work -- this is a deterministic aggregation of data that already fully
exists, computed independently per bucket.

Usage (PowerShell):
    python step2_build_buckets.py --input "cicids2017_portscan_clean_v2.csv" --output "cicids2017_portscan_step2_buckets.csv"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_BUCKETS = 12_041
EXPECTED_POSITIVE_BUCKETS = 27

REQUIRED_COLS = [
    "Source IP", "Destination IP", "Destination Port", "Protocol",
    "Timestamp", "Flow Duration", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "PSH Flag Count",
    "Init_Win_bytes_forward", "Label",
]


def log(msg: str) -> None:
    print(f"[step2] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-source-IP, per-minute 9-feature buckets.")
    parser.add_argument("--input", required=True, help="Cleaned CSV from Step 1 (v2).")
    parser.add_argument("--output", required=True, help="Output bucket-level CSV path.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        log(f"ERROR: input not found: {in_path}")
        sys.exit(1)
    if out_path.exists():
        log(f"ERROR: refusing to overwrite existing output: {out_path}")
        sys.exit(1)

    log(f"Reading cleaned data: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log(f"ERROR: required columns missing from input: {missing}")
        sys.exit(1)

    log(f"Loaded {len(df)} rows x {df.shape[1]} cols")

    # --- Parse timestamp, floor to minute --------------------------------------
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    n_bad_ts = int(df["_parsed_ts"].isna().sum())
    if n_bad_ts > 0:
        log(f"WARNING: {n_bad_ts} row(s) have an unparseable Timestamp -- dropping before bucketing.")
        df = df.loc[df["_parsed_ts"].notna()].copy()

    df["_minute"] = df["_parsed_ts"].dt.floor("min")

    # --- Helper indicator columns -----------------------------------------------
    df["_psh_flag"] = (df["PSH Flag Count"] > 0).astype(int)
    df["_is_tcp"] = (df["Protocol"] == 6).astype(int)
    df["_init_win_valid"] = df["Init_Win_bytes_forward"].where(df["Init_Win_bytes_forward"] != -1)

    group_cols = ["Source IP", "_minute"]
    log(f"Grouping by {group_cols} ...")
    grouped = df.groupby(group_cols, sort=True)

    agg = grouped.agg(
        flow_count=("Source IP", "size"),
        distinct_dst_ports=("Destination Port", "nunique"),
        distinct_dst_ips=("Destination IP", "nunique"),
        mean_flow_duration=("Flow Duration", "mean"),
        total_fwd_bytes=("Total Length of Fwd Packets", "sum"),
        total_bwd_bytes=("Total Length of Bwd Packets", "sum"),
        psh_rate=("_psh_flag", "mean"),
        frac_tcp=("_is_tcp", "mean"),
        n_valid_init_win=("_init_win_valid", "count"),
        mean_init_win_fwd=("_init_win_valid", "mean"),
    ).reset_index()

    # Explicit handling for buckets where every Init_Win_bytes_forward was -1:
    # do NOT let -1 leak into the mean (it's already excluded via .where above),
    # and explicitly flag rather than silently leaving a NaN unexplained.
    agg["mean_init_win_fwd_missing"] = (agg["n_valid_init_win"] == 0).astype(int)
    agg.loc[agg["n_valid_init_win"] == 0, "mean_init_win_fwd"] = np.nan
    agg = agg.drop(columns=["n_valid_init_win"])

    # --- Bucket label: positive if ANY flow in the bucket is PortScan ----------
    label_agg = (
        grouped["Label"]
        .apply(lambda s: int((s == "PortScan").any()))
        .reset_index(name="bucket_label")
    )
    agg = agg.merge(label_agg, on=group_cols, how="left")

    agg = agg.rename(columns={"Source IP": "source_ip", "_minute": "minute"})

    n_buckets = len(agg)
    log(f"Built {n_buckets} (source_ip, minute) buckets (expected {EXPECTED_BUCKETS})")

    n_missing_init_win = int(agg["mean_init_win_fwd_missing"].sum())
    log(f"Buckets with no valid Init_Win_bytes_forward samples (all -1): {n_missing_init_win}")

    n_positive = int(agg["bucket_label"].sum())
    log(f"PortScan-positive buckets: {n_positive} (expected {EXPECTED_POSITIVE_BUCKETS})")

    log(f"Writing bucket-level CSV to: {out_path}")
    agg.to_csv(out_path, index=False)
    log("Done.")


if __name__ == "__main__":
    main()
