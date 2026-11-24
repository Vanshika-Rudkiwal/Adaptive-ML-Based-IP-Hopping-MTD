# Adaptive Machine Learning-Based IP Hopping Using Network Traffic Analysis for Moving Target Defense

An adaptive Moving Target Defense (MTD) framework that dynamically changes a
service's virtual IP address based on real-time network traffic analysis,
instead of hopping at a fixed interval. The system uses the CICIDS2017
dataset to build temporal traffic windows and compares four independent
threat-scoring approaches — Random Forest, CNN, LSTM, and an EWMA statistical
baseline. The selected threat score drives a model-agnostic adaptive decision
engine that determines IP-hopping frequency based on the observed threat
level. A mutation engine executes the virtual IP changes using Open vSwitch,
Flask, and Scapy inside a Mininet testbed, and the system is evaluated
against a static no-hopping baseline using both a legitimate
client and two attacker scenarios.

## Repository Structure

The project files are organized directly in the repository root.

```text
mtd/
├── README.md
├── INSTALLATION.md
├── USAGE.md
├── LICENSE
├── requirements-ml.txt
├── requirements-mtd.txt
├── step1_clean_data_v2.py
├── step2_build_buckets.py
├── step3_temporal_split.py
├── step4_rf_baseline.py
├── step5_cnn_baseline.py
├── step6_lstm_baseline.py
├── step7_ewma_baseline.py
├── step8_adaptive_decision_engine.py
├── step8_engine_config.json
├── step9_mutation_engine.py
├── step10_mininet_topology.py
├── step11_mtd_evaluation.py
└── results_final/
    ├── step10_experiment_static.csv
    ├── step10_experiment_adaptive.csv
    ├── step11_mtd_comparison.csv
    └── step11_mtd_comparison.png
```

## Dataset

This project uses the CICIDS2017 dataset (specifically the Friday
`PortScan` capture) from the Canadian Institute for Cybersecurity (CIC),
University of New Brunswick. Raw dataset files are **not** included in this
repository due to their size.

Download the dataset from the official source only:
https://www.unb.ca/cic/datasets/ids-2017.html

After downloading, place the relevant CSV file(s) in the project root (or
update the input path in `step1_clean_data_v2.py` to match your local layout) before running Step 1.

## Environments

The ML pipeline (Steps 1-8) and the MTD testbed (Steps 9-11) were developed
and run in two separate environments, and this repository keeps that
separation explicit rather than claiming a single unified environment. Both
require a compatible Python 3 environment; no specific Python version is
pinned here beyond that.

- **ML pipeline environment** (Steps 1-8): dependencies in
  `requirements-ml.txt` (pandas, numpy, scikit-learn, torch, matplotlib,
  joblib — verified against each script's actual imports). This environment
  trains and evaluates the four threat-scoring models and produces the
  adaptive hopping schedule.
- **MTD testbed environment** (Steps 9-11): a Linux/WSL2 environment with
  Mininet and Open vSwitch, with Python dependencies in
  `requirements-mtd.txt`. Mininet and Open vSwitch are system-level
  dependencies and are **not** installable via pip — see
  [INSTALLATION.md](INSTALLATION.md).

Full setup steps for both environments are in [INSTALLATION.md](INSTALLATION.md).
Full run instructions for every step are in [USAGE.md](USAGE.md).

## Architecture Summary

```
CICIDS2017 (raw) -> per-source/per-minute traffic buckets -> temporal windows
   -> threat-scoring model (RF / CNN / LSTM / EWMA, compared independently)
   -> adaptive decision engine (threat score -> hopping frequency)
   -> mutation engine (Open vSwitch + Flask + Scapy, VIP hopping)
   -> Mininet testbed (attacker, victim, legitimate client)
   -> evaluation vs. static no-hopping baseline
```

LSTM is the primary runtime detector driving the adaptive testbed. Per the
project's locked architecture rule (see USAGE.md, Step 7), LSTM must
outperform the EWMA baseline on the cross-model comparison to be treated as
primary. EWMA, Random Forest, and CNN are retained as comparison baselines
in the ML pipeline, not as the runtime score source for Steps 9-11.

## Scope and Known Limitations

These are carried directly from the implementation and evaluation code, not
added after the fact:

- The MTD evaluation (Step 11) is a **single-topology, single-VIP-pool-size
  pilot study**. Results should not be generalized beyond this specific
  configuration without varying attacker scan interval and VIP pool size as
  controlled variables.
- The VIP pool size (50 addresses) and cooldown size (10 addresses) used by
  the mutation engine were chosen with documented reasoning but were not
  swept or tuned for optimality; effectiveness at other pool sizes is
  untested.
- The end-to-end system's real responsiveness is bounded by the score-feed
  poll interval (60 seconds, matching the per-minute traffic bucket
  granularity established in Step 1), not by the adaptive decision engine's
  faster internal tick rate.
- Two attacker strategies are evaluated (`rescan_last_known` and
  `pool_sweep`); an adversary that also attempts cross-VIP fingerprinting is
  explicitly out of scope.

## Authenticated Endpoints

The mutation engine's REST API (`step9_mutation_engine.py`) gates its
state-exposing endpoints (`/mtd/current_vip`, `/mtd/reinit`, `/mtd/hop_log`)
behind a configured shared-secret API key. This key must be set consistently
between `step9_mutation_engine.py` and `step10_mininet_topology.py` before
running any experiment. See [USAGE.md](USAGE.md) for where this is
configured.

**Security note:** Do not commit a real API key value to version control.
The shipped key is a placeholder for local development and must be replaced
before any deployment.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Acknowledgements

This project uses the CICIDS2017 dataset developed by the Canadian
Institute for Cybersecurity (CIC), University of New Brunswick.
