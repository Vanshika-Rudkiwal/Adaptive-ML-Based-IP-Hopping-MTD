# Installation

This project requires two separate environments, matching how it was
actually developed and run: one for the ML pipeline (Steps 1-8), one for the
MTD testbed (Steps 9-11). Do not attempt to install both dependency sets
into a single environment and expect Steps 9-11 to work — Mininet and Open
vSwitch require a Linux (or WSL2) kernel and are not needed by, or
compatible with, the ML pipeline environment.

## 1. Clone the repository

```
git clone https://github.com/<username>/<repository-name>
cd <repository-name>
```

(Replace `<username>/<repository-name>` with this repository's actual GitHub
path.)

## 2. ML pipeline environment (Steps 1-8)

Requires a compatible Python 3 environment (no specific version is pinned).

```
python3 -m venv venv-ml
source venv-ml/bin/activate
pip install -r requirements-ml.txt
```

No additional system-level dependencies are required for this environment.

## 3. MTD testbed environment (Steps 9-11)

This environment requires a Linux system or WSL2 with a Linux kernel that
supports network namespaces (required by Mininet), and a compatible Python 3
environment. It was developed and tested on WSL2.

### 3a. Install Mininet and Open vSwitch (system-level, not pip)

These are installed via the official Mininet-documented package method, not
via pip:

```
sudo apt-get update
sudo apt-get install mininet
sudo apt-get install openvswitch-switch
sudo service openvswitch-switch start
```

Verify the install:

```
sudo mn --switch ovsbr --test pingall
```

If you prefer the official native-from-source installation instead of the
packaged one (e.g. to get a newer Mininet/OVS version), follow the
"Native Installation from Source" instructions at
http://mininet.org/download/, which uses:

```
git clone https://github.com/mininet/mininet
mininet/util/install.sh -nfv
```

### 3b. Install nmap (system-level, not pip)

The Mininet testbed script (`step10_mininet_topology.py`) invokes `nmap`
directly for the attacker simulation and requires it to be present on the
system PATH:

```
sudo apt-get install nmap
```

### 3c. Install the Python dependencies for this environment

```
python3 -m venv venv-mtd
source venv-mtd/bin/activate
pip install -r requirements-mtd.txt
```

Step 10 must be run with elevated privileges because Mininet, Open vSwitch
flow management, and Scapy's raw-socket ARP handling require them. Step 10
manages the Step 9 mutation-engine process automatically. If you use a
virtual environment, use `sudo -E` to preserve its PATH when launching Step 10.
Step 11 is an evaluation script and can be run as a normal user.

## 4. Dataset

Download the CICIDS2017 dataset from the official source and place it as
described in [README.md](README.md#dataset) before running Step 1:

https://www.unb.ca/cic/datasets/ids-2017.html

## Next steps

Once both environments are set up, see [USAGE.md](USAGE.md) for how to run
each step.
