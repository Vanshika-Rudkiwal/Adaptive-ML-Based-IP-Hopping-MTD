"""
Step 8 — Adaptive IP-Hopping Decision Engine
CICIDS2017 PortScan / Adaptive MTD project

Reads (read-only, never modified):
  step6_lstm_predictions.csv   (LSTM threat scores, already computed in Step 6)

Writes (new files only):
  step8_hopping_schedule.csv
  step8_hopping_schedule.png   (only if matplotlib available)
  step8_engine_config.json

IMPORTANT — no models are trained, loaded, or modified here:
  Step 6 (step6_lstm_baseline.py) did not persist trained LSTM weights,
  only metrics and per-window predicted_proba scores. This engine is
  therefore built MODEL-AGNOSTIC: it consumes a threat score in [0,1]
  from whatever upstream source provides one. Today that source is the
  already-computed step6_lstm_predictions.csv (LSTM, finalized primary
  detector per project decision). Live deployment later only needs a
  thin adapter feeding real-time LSTM inference into the same
  engine.update(score, timestamp) call — this file does not change.

Design:
  - EWMA smoothing on the incoming raw score (reduces reaction to
    single-minute detector noise, independent of Step 7's EWMA baseline)
  - Continuous score -> hop-interval mapping (exponential decay from
    T_max at score=0 to T_min at score=1)
  - Discrete tiers (NORMAL/ELEVATED/HIGH/CRITICAL) for simple SDN
    controller state and logging
  - Hysteresis on tier downgrades (prevents flapping near a boundary)
  - Rate limiting on interval changes (prevents abrupt network
    disruption from a single noisy score)
  - Minimum dwell time before another decision can take effect

No Ryu/Mininet imports — this module is meant to be unit-testable and
reusable standalone; the integration point is documented at the bottom.

Tested against: Python 3.12
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
@dataclass
class HoppingConfig:
    t_min: float = 5.0        # seconds — fastest hop, CRITICAL threat
    t_max: float = 300.0      # seconds — slowest hop, NORMAL/idle
    tiers: dict = field(default_factory=lambda: {
        "NORMAL":   (0.00, 0.25),
        "ELEVATED": (0.25, 0.50),
        "HIGH":     (0.50, 0.75),
        "CRITICAL": (0.75, 1.01),
    })
    ewma_alpha: float = 0.4            # smoothing on raw incoming score
    hysteresis_band: float = 0.05      # required margin below boundary to downgrade tier
    max_step_change_frac: float = 0.5  # max fractional interval change per decision
    min_dwell_seconds: float = 5.0     # min time before interval can change again
    decay_steepness: float = 4.0       # k in exponential score->interval mapping


@dataclass
class HopDecision:
    timestamp: pd.Timestamp
    raw_score: float
    smoothed_score: float
    tier: str
    interval_seconds: float
    hop_now: bool


# ----------------------------------------------------------------------
# ENGINE
# ----------------------------------------------------------------------
class AdaptiveHoppingEngine:
    """
    Model-agnostic. Call update(score, timestamp) once per polling interval
    with a threat score in [0, 1] from ANY upstream detector.
    """

    def __init__(self, config: Optional[HoppingConfig] = None):
        self.cfg = config or HoppingConfig()
        self._tier_order = list(self.cfg.tiers.keys())
        self._smoothed_score = None
        self._current_tier = "NORMAL"
        self._current_interval = self.cfg.t_max
        self._last_decision_time = None
        self.history = []

    def _map_score_to_interval(self, score: float) -> float:
        score = float(np.clip(score, 0.0, 1.0))
        k = self.cfg.decay_steepness
        interval = self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * np.exp(-k * score)
        return float(np.clip(interval, self.cfg.t_min, self.cfg.t_max))

    def _classify_tier(self, score: float) -> str:
        for tier, (lo, hi) in self.cfg.tiers.items():
            if lo <= score < hi:
                return tier
        return self._tier_order[-1]  # CRITICAL fallback

    def _apply_hysteresis(self, new_tier: str) -> str:
        cur_idx = self._tier_order.index(self._current_tier)
        new_idx = self._tier_order.index(new_tier)
        if new_idx < cur_idx:  # downgrade requested (less urgent)
            lo, _ = self.cfg.tiers[self._current_tier]
            if self._smoothed_score is not None and self._smoothed_score > (lo - self.cfg.hysteresis_band):
                return self._current_tier  # not enough margin yet, hold
        return new_tier

    def _apply_rate_limit(self, target_interval: float) -> float:
        max_delta = self._current_interval * self.cfg.max_step_change_frac
        if target_interval < self._current_interval:
            return max(target_interval, self._current_interval - max_delta)
        return min(target_interval, self._current_interval + max_delta)

    def update(self, raw_score: float, timestamp: pd.Timestamp) -> HopDecision:
        if self._smoothed_score is None:
            self._smoothed_score = raw_score
        else:
            a = self.cfg.ewma_alpha
            self._smoothed_score = a * raw_score + (1 - a) * self._smoothed_score

        raw_tier = self._classify_tier(self._smoothed_score)
        stable_tier = self._apply_hysteresis(raw_tier)

        target_interval = self._map_score_to_interval(self._smoothed_score)
        rate_limited_interval = self._apply_rate_limit(target_interval)

        hop_now = False
        if self._last_decision_time is None:
            hop_now = True
        else:
            elapsed = (timestamp - self._last_decision_time).total_seconds()
            hop_now = elapsed >= self.cfg.min_dwell_seconds

        if hop_now:
            self._current_interval = rate_limited_interval
            self._current_tier = stable_tier
            self._last_decision_time = timestamp

        decision = HopDecision(
            timestamp=timestamp,
            raw_score=float(raw_score),
            smoothed_score=float(self._smoothed_score),
            tier=self._current_tier,
            interval_seconds=float(self._current_interval),
            hop_now=hop_now,
        )
        self.history.append(decision)
        return decision

    def get_next_hop_time(self) -> Optional[pd.Timestamp]:
        if self._last_decision_time is None:
            return None
        return self._last_decision_time + pd.Timedelta(seconds=self._current_interval)

    def reset(self):
        self.__init__(self.cfg)


# ----------------------------------------------------------------------
# DEMO / REPLAY — validates the engine against Step 6's real test scores
# ----------------------------------------------------------------------
def load_network_threat_timeline():
    """
    Aggregate per-minute network-wide threat = max LSTM score across all
    source IPs active that minute (a defender doesn't know a priori which
    source is the attacker — any single high-scoring source should raise
    the whole network's posture).
    """
    log("=== Loading step6_lstm_predictions.csv (read-only) ===")
    df = pd.read_csv("step6_lstm_predictions.csv", parse_dates=["target_minute"])

    agg = (
        df.groupby("target_minute")
        .agg(threat_score=("predicted_proba", "max"),
             any_true_positive=("label", "max"))
        .reset_index()
        .sort_values("target_minute")
    )
    log(f"Loaded {len(df)} predictions -> {len(agg)} distinct minutes in test period.")
    log(f"Minutes with a true PortScan label present: {int(agg['any_true_positive'].sum())}")
    return agg


def run_replay(agg_df, config: HoppingConfig):
    log("\n=== Running adaptive hopping engine over test-period timeline ===")
    engine = AdaptiveHoppingEngine(config)
    rows = []
    for _, row in agg_df.iterrows():
        d = engine.update(row["threat_score"], row["target_minute"])
        rows.append({
            "timestamp": d.timestamp,
            "raw_lstm_score": d.raw_score,
            "smoothed_score": d.smoothed_score,
            "tier": d.tier,
            "interval_seconds": d.interval_seconds,
            "hop_decision_applied": d.hop_now,
            "true_positive_minute": bool(row["any_true_positive"]),
        })
    return pd.DataFrame(rows)


def sanity_checks(schedule_df, config: HoppingConfig):
    log("\n=== Sanity checks ===")
    lo, hi = schedule_df["interval_seconds"].min(), schedule_df["interval_seconds"].max()
    bounds_ok = (lo >= config.t_min - 1e-6) and (hi <= config.t_max + 1e-6)
    log(f"Interval bounds respected [{config.t_min}, {config.t_max}]: "
        f"observed [{lo:.2f}, {hi:.2f}] -> {bounds_ok}")

    tp_rows = schedule_df[schedule_df["true_positive_minute"]]
    fp_rows = schedule_df[~schedule_df["true_positive_minute"]]
    if len(tp_rows) and len(fp_rows):
        log(f"Mean hop interval during TRUE-POSITIVE minutes: {tp_rows['interval_seconds'].mean():.2f}s")
        log(f"Mean hop interval during benign minutes       : {fp_rows['interval_seconds'].mean():.2f}s")
        if tp_rows["interval_seconds"].mean() < fp_rows["interval_seconds"].mean():
            log("Confirmed: engine hops faster during true attack minutes than benign minutes.")
        else:
            log("WARNING: engine did NOT hop faster during attack minutes — review LSTM score "
                "quality or config thresholds before trusting this schedule.")
    else:
        log("Not enough true-positive/benign minutes in this slice to compare — check input.")

    tier_counts = schedule_df["tier"].value_counts()
    log(f"\nTier distribution across replay:\n{tier_counts.to_string()}")

    n_transitions = (schedule_df["tier"] != schedule_df["tier"].shift()).sum() - 1
    log(f"Total tier transitions during replay: {n_transitions}")


def save_outputs(schedule_df, config: HoppingConfig):
    schedule_df.to_csv("step8_hopping_schedule.csv", index=False)
    log("\nSaved: step8_hopping_schedule.csv")

    with open("step8_engine_config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    log("Saved: step8_engine_config.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(schedule_df["timestamp"], schedule_df["smoothed_score"], label="smoothed LSTM score")
        ax1.scatter(
            schedule_df.loc[schedule_df["true_positive_minute"], "timestamp"],
            schedule_df.loc[schedule_df["true_positive_minute"], "smoothed_score"],
            color="red", s=15, label="true PortScan minute", zorder=5,
        )
        ax1.set_ylabel("Threat score")
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(schedule_df["timestamp"], schedule_df["interval_seconds"], color="darkorange")
        ax2.set_ylabel("Hop interval (s)")
        ax2.set_xlabel("Time")
        plt.tight_layout()
        plt.savefig("step8_hopping_schedule.png", dpi=150)
        plt.close()
        log("Saved: step8_hopping_schedule.png")
    except ImportError:
        log("matplotlib not available — skipped PNG plot, CSV still saved.")


# ----------------------------------------------------------------------
# SDN INTEGRATION HOOK (Mininet/Ryu) — no Ryu/Mininet dependency here
# ----------------------------------------------------------------------
# In a live deployment, a Ryu controller app would, once per polling cycle
# (e.g. every minute, matching this project's bucket granularity):
#
#   1. Obtain a threat score in [0,1] from the live detector
#      (adapter around the trained LSTM — not yet built, since Step 6
#      did not persist model weights; this engine does not require that
#      adapter to exist yet).
#
#   2. decision = engine.update(score, current_timestamp)
#
#   3. If decision.hop_now:
#        - Select the host's next IP from the address pool
#        - Push a flow-mod / OVSDB update via Ryu's REST API
#          (ryu.app.wsgi + ofp_event) to remap the active address
#        - Schedule the next poll using decision.interval_seconds
#
#   4. Log decision.tier for operator visibility / alerting thresholds.
#
# This module has zero SDN imports so it can be unit-tested (as the demo
# below does, against real Step 6 output) independently of a running
# Mininet/Ryu environment.
# ----------------------------------------------------------------------


def main():
    config = HoppingConfig()
    log(f"Engine config: {config}")

    agg_df = load_network_threat_timeline()
    schedule_df = run_replay(agg_df, config)
    sanity_checks(schedule_df, config)
    save_outputs(schedule_df, config)

    log("\n=== STEP 8 (ADAPTIVE DECISION ENGINE) COMPLETE ===")
    log("No models were trained, loaded, or modified. Engine consumed the")
    log("already-computed LSTM scores from step6_lstm_predictions.csv.")
    log("For live SDN deployment, wire a real-time LSTM inference adapter")
    log("into AdaptiveHoppingEngine.update() — see integration hook above.")


if __name__ == "__main__":
    main()
