"""
Step 11 — MTD Effectiveness Evaluation
CICIDS2017 PortScan / Adaptive MTD project

Reads (read-only):
  step10_experiment_static.csv
  step10_experiment_adaptive.csv

Writes:
  step11_mtd_comparison.csv
  step11_mtd_comparison.png (if matplotlib available)

This is the missing evidence flagged in the examiner review: does hopping
actually degrade the attacker, and at what cost to legitimate traffic.

Step 10 now runs --trials repetitions per mode (default 5, different RNG
each time) and two attacker strategies per trial (rescan_last_known,
pool_sweep). This script aggregates across trials — reporting mean +/-
std per metric instead of a single number — and reports both attacker
strategies separately rather than only the weaker one.

Metrics computed, static vs adaptive, per attacker strategy:
  - Attacker hit rate: fraction of scan attempts that found the victim
  - Attacker ports-found rate: average ports discovered per attempt
  - Mean time-to-first-hit (attacker): how long until first successful scan
  - Legitimate client success rate: fraction of probes that reached the
    current VIP successfully (continuity cost of hopping)

Even with repeated trials, this remains a single-topology, single-VIP-pool-
size pilot study. Interpretation guidance is printed alongside the numbers
for exactly that reason — read it before writing any claim into the paper.
"""

import pandas as pd
import numpy as np


def log(msg):
    print(msg, flush=True)


def load_run(mode):
    path = f"step10_experiment_{mode}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        log(f"WARNING: {path} not found — run Step 10 in '{mode}' mode first.")
        return None
    return df


def _one_trial_attacker_metrics(atk_trial):
    """Compute the four attacker metrics for a single trial's rows
    (already filtered to one mode + one attacker_strategy + one trial)."""
    hit_rate = atk_trial["hit"].astype(str).str.lower().eq("true").mean()
    mean_ports = pd.to_numeric(atk_trial["ports_found"], errors="coerce").mean()
    mean_scan_duration = pd.to_numeric(atk_trial["scan_duration_s"], errors="coerce").mean()

    hits = atk_trial[atk_trial["hit"].astype(str).str.lower() == "true"]
    if len(hits):
        first_hit_time = hits["timestamp"].astype(float).min()
        run_start_time = atk_trial["timestamp"].astype(float).min()
        time_to_first_hit = first_hit_time - run_start_time
    else:
        time_to_first_hit = np.nan  # attacker never landed a hit this trial

    return hit_rate, mean_ports, mean_scan_duration, time_to_first_hit


def summarize_attacker(df, mode):
    """
    must-fix #5: aggregates PER TRIAL, then reports mean +/- std across
    trials, instead of pooling every attempt from every trial into one
    number (which would understate variance and overstate confidence).
    must-fix #3: reported separately per attacker_strategy, so the
    stronger pool_sweep adversary's number is never hidden behind the
    weaker rescan_last_known number.
    """
    atk = df[df["role"] == "attacker"].copy()
    if len(atk) == 0:
        return None
    if "attacker_strategy" not in atk.columns:
        atk["attacker_strategy"] = "rescan_last_known"  # back-compat with old single-strategy CSVs

    summaries = {}
    for strategy, strat_df in atk.groupby("attacker_strategy"):
        per_trial_rows = []
        for trial_num, trial_df in strat_df.groupby("trial"):
            hit_rate, mean_ports, mean_scan_duration, ttfh = _one_trial_attacker_metrics(trial_df)
            per_trial_rows.append({
                "trial": trial_num, "hit_rate": hit_rate, "mean_ports": mean_ports,
                "mean_scan_duration_s": mean_scan_duration, "time_to_first_hit_s": ttfh,
            })
        per_trial_df = pd.DataFrame(per_trial_rows)
        n_trials = len(per_trial_df)

        summary = {
            "mode": mode, "attacker_strategy": strategy,
            "n_trials": n_trials, "attacker_attempts": len(strat_df),
            "attacker_hit_rate_mean": per_trial_df["hit_rate"].mean(),
            "attacker_hit_rate_std": per_trial_df["hit_rate"].std(ddof=0) if n_trials > 1 else 0.0,
            "attacker_mean_ports_per_attempt_mean": per_trial_df["mean_ports"].mean(),
            "attacker_mean_ports_per_attempt_std": per_trial_df["mean_ports"].std(ddof=0) if n_trials > 1 else 0.0,
            "attacker_mean_scan_duration_s_mean": per_trial_df["mean_scan_duration_s"].mean(),
            "attacker_time_to_first_hit_s_mean": per_trial_df["time_to_first_hit_s"].mean(),
            "attacker_time_to_first_hit_s_std": per_trial_df["time_to_first_hit_s"].std(ddof=0) if n_trials > 1 else 0.0,
        }

        log(f"\n[{mode.upper()} / {strategy}] Attacker summary ({n_trials} trials, "
            f"{len(strat_df)} total attempts):")
        log(f"  Hit rate           : {summary['attacker_hit_rate_mean']:.2%} "
            f"(+/- {summary['attacker_hit_rate_std']:.2%})")
        log(f"  Mean ports/attempt : {summary['attacker_mean_ports_per_attempt_mean']:.2f} "
            f"(+/- {summary['attacker_mean_ports_per_attempt_std']:.2f})")
        ttfh_mean = summary["attacker_time_to_first_hit_s_mean"]
        ttfh_std = summary["attacker_time_to_first_hit_s_std"]
        if pd.isna(ttfh_mean):
            log("  Time to first hit  : N/A (never hit in any trial)")
        else:
            log(f"  Time to first hit  : {ttfh_mean:.1f}s (+/- {ttfh_std:.1f}s)")

        summaries[strategy] = summary

    return summaries


def summarize_legit_client(df, mode):
    legit = df[df["role"] == "legit_client"].copy()
    if len(legit) == 0:
        return None

    if "trial" not in legit.columns:
        legit["trial"] = 1  # back-compat with old single-trial CSVs

    per_trial = legit.groupby("trial")["success"].apply(
        lambda s: s.astype(str).str.lower().eq("true").mean()
    )
    n_trials = len(per_trial)
    success_rate_mean = per_trial.mean()
    success_rate_std = per_trial.std(ddof=0) if n_trials > 1 else 0.0

    log(f"\n[{mode.upper()}] Legitimate client summary ({n_trials} trials, "
        f"{len(legit)} total probes):")
    log(f"  Success rate : {success_rate_mean:.2%} (+/- {success_rate_std:.2%})")

    return {
        "mode": mode, "n_trials": n_trials, "legit_attempts": len(legit),
        "legit_success_rate_mean": success_rate_mean,
        "legit_success_rate_std": success_rate_std,
    }


def compare(static_atk_summaries, adaptive_atk_summaries, static_legit, adaptive_legit):
    log("\n" + "=" * 60)
    log("STATIC vs ADAPTIVE — MTD EFFECTIVENESS COMPARISON (mean +/- std over trials)")
    log("=" * 60)

    if not static_atk_summaries or not adaptive_atk_summaries:
        log("Cannot compare — one or both attacker summaries missing. "
            "Run Step 10 in both modes first.")
        return None

    strategies = sorted(set(static_atk_summaries) & set(adaptive_atk_summaries))
    if not strategies:
        log("No attacker_strategy in common between static and adaptive runs — "
            "check that both were generated with the current Step 10.")

    rows = []
    for strategy in strategies:
        s = static_atk_summaries[strategy]
        a = adaptive_atk_summaries[strategy]
        for key, label in [
            ("attacker_hit_rate", "Attacker hit rate"),
            ("attacker_mean_ports_per_attempt", "Attacker mean ports/attempt"),
            ("attacker_time_to_first_hit_s", "Attacker time-to-first-hit (s)"),
        ]:
            rows.append({
                "attacker_strategy": strategy, "metric": label,
                "static_mean": s.get(f"{key}_mean"), "static_std": s.get(f"{key}_std"),
                "adaptive_mean": a.get(f"{key}_mean"), "adaptive_std": a.get(f"{key}_std"),
            })
            log(f"[{strategy}] {label:32s}: "
                f"static={s.get(f'{key}_mean')} (+/- {s.get(f'{key}_std')})  "
                f"adaptive={a.get(f'{key}_mean')} (+/- {a.get(f'{key}_std')})")

    if static_legit and adaptive_legit:
        rows.append({
            "attacker_strategy": "n/a", "metric": "Legit client success rate",
            "static_mean": static_legit["legit_success_rate_mean"],
            "static_std": static_legit["legit_success_rate_std"],
            "adaptive_mean": adaptive_legit["legit_success_rate_mean"],
            "adaptive_std": adaptive_legit["legit_success_rate_std"],
        })
        log(f"{'Legit client success rate':40s}: "
            f"static={static_legit['legit_success_rate_mean']:.2%} "
            f"(+/- {static_legit['legit_success_rate_std']:.2%})  "
            f"adaptive={adaptive_legit['legit_success_rate_mean']:.2%} "
            f"(+/- {adaptive_legit['legit_success_rate_std']:.2%})")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("step11_mtd_comparison.csv", index=False)
    log("\nSaved: step11_mtd_comparison.csv")

    log("\n--- INTERPRETATION (read before claiming anything in a paper) ---")
    for strategy in strategies:
        s = static_atk_summaries[strategy]
        a = adaptive_atk_summaries[strategy]
        if a["attacker_hit_rate_mean"] < s["attacker_hit_rate_mean"]:
            log(f"[{strategy}] Adaptive hit rate lower than static — directionally "
                f"consistent with the MTD hypothesis against this adversary.")
        else:
            log(f"[{strategy}] Adaptive hit rate did NOT improve on static. For "
                f"pool_sweep specifically, this may be expected (a sweeping attacker "
                f"that revisits the whole pool every interval is a much harder adversary "
                f"to defeat with address hopping alone) — do not silently drop this "
                f"result, report and discuss it honestly.")
    log("\nEven with repeated trials, this is still a single topology and a single VIP "
        "pool size. Before generalizing beyond this specific configuration:")
    log("  - Vary attacker scan interval and VIP pool size as controlled variables")
    log("  - Consider an adversary that also attempts cross-VIP fingerprinting "
        "(explicitly out of scope — see Step 9 module docstring)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = comp_df[comp_df["metric"].isin(["Attacker hit rate", "Legit client success rate"])]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = (plot_df["attacker_strategy"] + " / " + plot_df["metric"]).tolist()
        x = np.arange(len(plot_df))
        width = 0.35
        ax.bar(x - width/2, plot_df["static_mean"], width, yerr=plot_df["static_std"],
               label="static", capsize=3)
        ax.bar(x + width/2, plot_df["adaptive_mean"], width, yerr=plot_df["adaptive_std"],
               label="adaptive", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Rate")
        ax.set_title("MTD Effectiveness: Static vs Adaptive (mean +/- std over trials)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("step11_mtd_comparison.png", dpi=150)
        plt.close()
        log("Saved: step11_mtd_comparison.png")
    except ImportError:
        log("matplotlib not available — skipped PNG plot, CSV still saved.")

    return comp_df


def main():
    static_df = load_run("static")
    adaptive_df = load_run("adaptive")

    # summarize_attacker now returns {strategy: summary_dict, ...} or None
    static_atk = summarize_attacker(static_df, "static") if static_df is not None else None
    adaptive_atk = summarize_attacker(adaptive_df, "adaptive") if adaptive_df is not None else None
    static_legit = summarize_legit_client(static_df, "static") if static_df is not None else None
    adaptive_legit = summarize_legit_client(adaptive_df, "adaptive") if adaptive_df is not None else None

    compare(static_atk, adaptive_atk, static_legit, adaptive_legit)

    log("\n=== STEP 11 (MTD EVALUATION) COMPLETE ===")


if __name__ == "__main__":
    main()
