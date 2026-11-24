# Usage Guide

## 1. Project Overview

This project implements Adaptive Machine Learning-Based IP Hopping using Network Traffic Analysis for Moving Target Defense.

The workflow consists of data preprocessing, machine learning models, adaptive decision-making, network mutation, and Mininet-based evaluation.

## 2. Machine Learning Pipeline (Steps 1–8)

Run the following commands from the project root directory after activating the ML environment.

### Step 1: Clean Dataset

```bash
python3 step1_clean_data_v2.py --input PATH_TO_ORIGINAL_CSV --output cleaned.csv
```

### Step 2: Build Traffic Buckets

```bash
python3 step2_build_buckets.py --input cleaned.csv --output cicids2017_portscan_step2_buckets.csv
```

### Step 3: Temporal Train-Test Split

```bash
python3 step3_temporal_split.py
```

### Step 4: Random Forest Model

```bash
python3 step4_rf_baseline.py
```

### Step 5: CNN Model

```bash
python3 step5_cnn_baseline.py
```

### Step 6: LSTM Model

```bash
python3 step6_lstm_baseline.py
```

### Step 7: EWMA Baseline

```bash
python3 step7_ewma_baseline.py
```

### Step 8: Adaptive Decision Engine

```bash
python3 step8_adaptive_decision_engine.py
```

## 3. Moving Target Defense Testbed (Steps 9–10)

The MTD testbed requires a properly configured Mininet and Open vSwitch environment.

Step 9 provides the mutation engine used by the network topology.

### Static Experiment

```bash
sudo -E python3 step10_mininet_topology.py --mode static --trials 5
```

### Adaptive Experiment

```bash
sudo -E python3 step10_mininet_topology.py --mode adaptive --trials 5
```

The experiments generate:

- step10_experiment_static.csv
- step10_experiment_adaptive.csv

## 4. Evaluation (Step 11)

Run this step in the ML environment, where pandas, numpy, and matplotlib are installed.

After both experiments have completed, run:

```bash
python3 step11_mtd_evaluation.py
```

The evaluation generates:

- step11_mtd_comparison.csv
- step11_mtd_comparison.png

## 5. Important Notes

- Run commands from the project root directory.
- Keep the ML and MTD environments separate.
- The original CICIDS2017 dataset is not included in this repository.
- Replace PATH_TO_ORIGINAL_CSV with the actual dataset path.
- Steps 9 and 10 require the configured MTD and Mininet environment.
- Static mode in the current implementation means no IP hopping.
- Five repeated trials are repeated runs, not five statistically independent datasets.
- Interpret experimental results within the scope and limitations described in README.md.
