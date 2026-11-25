"""
Step 10 — Mininet MTD Experiment
CICIDS2017 PortScan / Adaptive MTD project
(FLASK + ovs-ofctl EDITION — no Ryu, no RemoteController)

Topology: attacker(h1) --- switch(s1, standalone OVS) --- victim(h2)
                                    |
                              legit_client(h3)

WHAT CHANGED FROM THE RYU VERSION (read this before running)
--------------------------------------------------------------
- No RemoteController / OpenFlow controller session at all. The switch
  is created with failMode='standalone', so OVS does normal MAC learning
  on its own — Step 9 layers VIP<->real-IP NAT flows and the deny-real-IP
  flow on top of that via `ovs-ofctl`, same as before.
- Step 9 (step9_mutation_engine.py) is now launched BY THIS SCRIPT as a
  subprocess, once, the first time a topology is built — not started
  separately beforehand. That's because, unlike a Ryu controller that
  can start before any switch exists and just wait for a connection,
  ovs-ofctl needs an actual bridge to push flows to.
- Because this script tears down and rebuilds the Mininet topology once
  per trial (net.stop() -> build_topology()), and a fresh OVS bridge
  named "s1" has no memory of flows pushed to the previous bridge
  instance, build_topology() calls POST /mtd/reinit on every trial (not
  just the first) so Step 9 re-pushes its base flows + current VIP's NAT
  flow onto the new bridge. This replaces what Ryu's
  switch_features_handler used to do automatically on every reconnect.
- CONTROLLER_OF_PORT is gone (no OpenFlow session to connect to).
  CONTROLLER_REST_PORT (8080) is unchanged — Step 9's REST API is
  otherwise identical.

Two run modes (unchanged):
  --mode static    : Step 9's controller with hopping disabled (one fixed
                     VIP for the whole run) — this is the BASELINE.
  --mode adaptive   : Step 9's controller with hopping active, driven by
                     a score-feed process (see feed_scores() below, which
                     replays step6_lstm_predictions.csv on a real clock
                     to simulate a live detector).

What this measures (feeds directly into Step 11) — unchanged:
  - Attacker: repeated targeted scans against the last-known VIP, logging
    hit/miss and ports found per attempt.
  - Legitimate client: periodic requests against whatever VIP the
    /mtd/current_vip registry currently reports, logging success/fail —
    this is the continuity cost of hopping.

Requires: Mininet, Open vSwitch, nmap, Flask, scapy, pandas, requests.
          (Ryu is no longer required at all.)
Run as root: sudo python3 step10_mininet_topology.py --mode adaptive
Step 9 (step9_mutation_engine.py) must be in the same directory — this
script starts it for you; you do not need a separate terminal for it.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import threading
import requests
from datetime import datetime

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

CONTROLLER_IP = "127.0.0.1"
CONTROLLER_REST_PORT = 8080
MTD_CONTROLLER_SCRIPT = "step9_mutation_engine.py"  # launched as a subprocess

VICTIM_REAL_IP = "10.0.0.2"
# Port the victim actually listens on, so the (unchanged) attacker scans of
# 22/80/443 have a real open port to find. 80 is already one of the ports
# both attacker functions scan, so no change to attacker logic is needed.
# The whole script already requires root for Mininet itself, so binding a
# privileged port (<1024) on h2 needs no extra handling.
VICTIM_HTTP_PORT = 80
EXPERIMENT_DURATION_SECONDS = 300  # 5 minutes per run, adjust as needed
ATTACKER_SCAN_INTERVAL = 10        # seconds between re-scan-last-known-VIP attempts
LEGIT_CLIENT_INTERVAL = 5          # seconds between legit client probes

# must-fix #3: second, stronger attacker strategy. A full-pool sweep is a
# more capable adversary than "keep hitting the last known VIP" — any
# reviewer who has seen an MTD paper before will ask for this number.
# Interval chosen so a sweep completes well within one hop interval at
# the smallest tier boundary configured in Step 8, so it's a genuine test
# of whether hopping outruns a sweeping attacker, not a strawman.
ATTACKER_SWEEP_INTERVAL = 15       # seconds between full-pool sweep attempts

# must-fix #1: must match MTD_API_KEY in step9_mutation_engine.py exactly.
# The legit client (and now also this script's own reinit calls) are the
# only callers that are SUPPOSED to know this — the attacker functions
# below deliberately never send it.
MTD_API_KEY = "CHANGE-ME-lab-shared-secret"

# must-fix #4 (cadence, resolved by rescoping rather than by changing
# Step 8): Step 8's engine supports faster reaction, but this prototype's
# ML process only produces one score per minute — matching the per-minute
# bucket granularity established all the way back in Step 1. That is the
# real, honest responsiveness bound of the end-to-end system. Named here
# explicitly instead of a bare "60" so it's clear this is a deliberate,
# documented scope decision, not an oversight.
SCORE_POLL_INTERVAL_SECONDS = 60

_mtd_process = None  # the Step 9 subprocess handle, set once by ensure_mtd_controller()


def log(msg):
    print(f"[EXPERIMENT {datetime.now().isoformat()}] {msg}", flush=True)


def _resolve_nmap_path():
    """
    Mininet's Host.cmd() runs each command through a per-host shell that
    does NOT simply inherit this process's PATH — so a bare "nmap" can
    fail with "bash: nmap: command not found" inside h1 even though
    `nmap` works fine in the shell you launched Step 10 from. Resolve the
    absolute path once here, using THIS process's (correct) PATH, and use
    that absolute path for every nmap invocation inside Mininet hosts
    instead of depending on the host shell's PATH to find it.
    """
    path = shutil.which("nmap")
    if path is None:
        raise RuntimeError(
            "nmap not found on this machine's PATH. Install it first, "
            "e.g.: sudo apt install nmap"
        )
    return path


NMAP = _resolve_nmap_path()  # absolute path, e.g. "/usr/bin/nmap"


# ----------------------------------------------------------------------
# STEP 9 PROCESS MANAGEMENT (new — replaces the old "start ryu-manager
# in a separate terminal before running this script" instruction)
# ----------------------------------------------------------------------
def _resolve_mtd_interpreter():
    """
    Pick the Python interpreter to launch Step 9 with.

    Why not just use sys.executable unconditionally: if this script was
    started as `sudo python3 step10_mininet_topology.py` (no `-E`), sudo
    resets PATH to its secure default and drops the invoking shell's
    environment BEFORE this process even starts — so sys.executable may
    already have resolved to the system python3, not the venv one Flask
    and Scapy are installed in, and there's no correct PATH/VIRTUAL_ENV
    left to inherit even if we forwarded the environment as-is. Running
    `sudo -E python3 step9_mutation_engine.py` manually works precisely
    because `-E` preserves your shell's PATH (venv first) — that's the
    thing this function reconstructs explicitly instead of depending on.

    Preference order:
      1. VIRTUAL_ENV, if that environment variable survived (e.g. you
         DID use `sudo -E`, or exported it via sudoers env_keep) — use
         that venv's own python3 directly, bypassing PATH entirely.
      2. sys.executable — correct if you're not using a venv, or if the
         venv's PATH did make it through.
    """
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        venv_python = os.path.join(virtual_env, "bin", "python3")
        if os.path.exists(venv_python):
            return venv_python
    return sys.executable


def _preflight_check_dependencies(interpreter, env):
    """
    Try to import Step 9's dependencies with the exact interpreter/env
    we're about to launch it with, and fail with a clear, actionable
    message now — rather than have Step 9 crash a few hundred
    milliseconds later with a bare ModuleNotFoundError that's easy to
    miss in scrolling Mininet output.
    """
    result = subprocess.run(
        [interpreter, "-c", "import flask, scapy"],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot start {MTD_CONTROLLER_SCRIPT}: interpreter "
            f"'{interpreter}' cannot import flask/scapy.\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--------------\n"
            f"This almost always means Step 10 was launched with `sudo` "
            f"(not `sudo -E`), which resets PATH and drops your venv "
            f"before Step 10 even starts. Fix by either:\n"
            f"  - Running Step 10 itself as: sudo -E python3 {os.path.basename(__file__)} ...\n"
            f"  - Or installing flask/scapy into the system-wide python3 "
            f"that plain `sudo python3` resolves to.\n"
            f"(Manually running `sudo -E python3 {MTD_CONTROLLER_SCRIPT}` "
            f"working, while this fails, is the classic symptom of the "
            f"first case.)"
        )


def ensure_mtd_controller():
    """
    Start step9_mutation_engine.py as a subprocess if it isn't already
    running (idempotent — safe to call before every trial). Waits for
    its REST API to respond before returning.
    """
    global _mtd_process
    if _mtd_process is not None and _mtd_process.poll() is None:
        return  # already running

    interpreter = _resolve_mtd_interpreter()
    env = os.environ.copy()  # forward explicitly rather than rely on Popen's implicit inheritance
    _preflight_check_dependencies(interpreter, env)

    log(f"Starting MTD controller subprocess: {MTD_CONTROLLER_SCRIPT} "
        f"(interpreter={interpreter})")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MTD_CONTROLLER_SCRIPT)
    _mtd_process = subprocess.Popen([interpreter, script_path], env=env)

    deadline = time.time() + 15
    while time.time() < deadline:
        if _mtd_process.poll() is not None:
            raise RuntimeError(
                f"{MTD_CONTROLLER_SCRIPT} exited immediately with code "
                f"{_mtd_process.returncode} — check its log output above."
            )
        try:
            resp = requests.get(
                f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/current_vip",
                headers={"X-MTD-Key": MTD_API_KEY}, timeout=1,
            )
            if resp.status_code == 200:
                log("MTD controller REST API is up.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"{MTD_CONTROLLER_SCRIPT} did not become reachable on port "
        f"{CONTROLLER_REST_PORT} within 15s — check its own log output."
    )


def reinit_mtd_controller():
    """
    Tell Step 9 to (re)push its flows onto the current OVS bridge. Must
    be called after every build_topology(), including the first — this
    is the explicit stand-in for Ryu's automatic switch_features_handler
    reconnect behavior (see module docstring).
    """
    resp = requests.post(
        f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/reinit",
        headers={"X-MTD-Key": MTD_API_KEY}, timeout=5,
    )
    resp.raise_for_status()
    log(f"MTD controller reinitialized: {resp.json()}")


def stop_mtd_controller():
    global _mtd_process
    if _mtd_process is not None and _mtd_process.poll() is None:
        log("Stopping MTD controller subprocess.")
        _mtd_process.terminate()
        try:
            _mtd_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mtd_process.kill()
    _mtd_process = None


# ----------------------------------------------------------------------
# TOPOLOGY
# ----------------------------------------------------------------------
def _bring_up_switch_interface(name):
    """
    Standalone OVS creates a netdev for the bridge itself (same name as
    the bridge, e.g. "s1") but does NOT bring it up automatically —
    Mininet only manages the host-facing veth ends, not the bridge's own
    interface. Step 9's ARP responder sniffs/injects on exactly this
    interface. Left down, the interface is still visible to tools that
    just list interfaces (which is why it didn't fail at "interface not
    found" anymore after the earlier fix) but every socket bound to it
    fails with OSError("Network is down") — repeatedly, since nothing
    ever changes that state on its own. This is the actual root cause of
    both the ARP responder's error loop and, downstream, of the legit
    client/attacker never being able to resolve the VIP's MAC address at
    all (no ARP replies were ever actually being sent).
    """
    result = subprocess.run(["ip", "link", "set", "dev", name, "up"],
                             capture_output=True, text=True)
    if result.returncode != 0:
        log(f"WARNING: failed to bring up interface '{name}': {result.stderr.strip()}")
    else:
        log(f"Brought up switch interface '{name}' for ARP responder use.")


def _start_victim_service(h2):
    """
    The attacker (unchanged) scans ports 22, 80, 443 with --open. Nothing
    was ever listening on h2, so nmap's --open filter had zero matching
    ports and — this is the actual mechanism, not a scan-logic bug —
    --open suppresses the "Nmap scan report for ..." line ENTIRELY for
    hosts with no open ports. That's why _extract_first_live_ip() always
    returned None and ports_found was always 0, even though NAT/ARP were
    both working correctly.

    Starts Python's built-in http.server on VICTIM_HTTP_PORT (80),
    bound to h2's real IP, using Host.popen() so we get back a handle
    that can be terminated cleanly during trial teardown instead of
    leaking a background process per trial.
    """
    log(f"Starting lightweight HTTP service on victim (h2) port {VICTIM_HTTP_PORT}")
    proc = h2.popen(
        ["python3", "-m", "http.server", str(VICTIM_HTTP_PORT), "--bind", VICTIM_REAL_IP],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)  # give it a moment to bind before scans can start
    if proc.poll() is not None:
        log(f"WARNING: victim HTTP service exited immediately "
            f"(code {proc.returncode}) — check port {VICTIM_HTTP_PORT} isn't "
            f"already in use.")
    else:
        log(f"Victim HTTP service listening on {VICTIM_REAL_IP}:{VICTIM_HTTP_PORT}")
    return proc


def _stop_victim_service(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def build_topology():
    # controller=None: no OpenFlow controller session. failMode='standalone'
    # on the switch itself is what makes it behave like a normal learning
    # switch without one (see module docstring).
    net = Mininet(controller=None, switch=OVSSwitch, autoSetMacs=True)

    s1 = net.addSwitch("s1", failMode="standalone")

    h1 = net.addHost("h1", ip="10.0.0.1/24")   # attacker
    h2 = net.addHost("h2", ip=f"{VICTIM_REAL_IP}/24")  # victim (real IP, stable)
    h3 = net.addHost("h3", ip="10.0.0.3/24")   # legitimate client

    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)

    net.build()
    s1.start([])  # no controllers — standalone fail-mode handles L2 forwarding
    _bring_up_switch_interface("s1")

    # Bring the MTD controller up (first call only) and (re)push its
    # flows onto this specific bridge instance (every call).
    ensure_mtd_controller()
    reinit_mtd_controller()
   
    os.system("sudo ovs-ofctl show s1")


    h2_service = _start_victim_service(h2)

    return net, h1, h2, h3, h2_service


# ----------------------------------------------------------------------
# ATTACKER PROCESS — repeated targeted scans against last-known VIP
# ----------------------------------------------------------------------
def run_attacker_rescan(h1, mode, results, h1_lock):
    """
    Attacker strategy 1: rescan_last_known.

    Attacker does NOT have access to /mtd/current_vip — that endpoint is
    for LEGITIMATE clients only (and is key-gated, see Step 9 must-fix
    #1). The attacker's strategy here is to remember the last VIP it
    successfully found (from an earlier scan or reconnaissance) and keep
    probing it, which is exactly what static IP-based attacks do. This
    is a WEAK adversary: it never revisits its assumption.

    h1_lock: run_attacker_sweep() targets the SAME Mininet host (h1) from
    a separate thread. Mininet's Host.cmd() drives a single shared shell
    process per host (it writes the command, then reads until it sees a
    sentinel marking the prompt) — it is NOT safe to call concurrently
    from two threads, since their reads/writes interleave on the same
    shell and corrupt each other's output parsing (this is what caused
    the AssertionError). h1_lock serializes every h1.cmd() call across
    both attacker strategies so they still both run, just never
    simultaneously.
    """
    log("Attacker (rescan): starting initial discovery scan across the VIP pool range")
    # initial discovery: attacker sweeps the known candidate range once,
    # simulating prior recon that found a live host at some point
    # PATCH: retry discovery. If the scan finds nothing (e.g. it ran while
    # the VIP was hopping), the old code fell back to the REAL IP, which the
    # switch blocks -> attacker scored 0 hits for the whole run (invalid).
    known_vip = VICTIM_REAL_IP
    for _try in range(1, 11):
        with h1_lock:
            discovery = h1.cmd(f"{NMAP} -n -p 22,80,443 --open -T4 10.0.0.100-149 2>&1")
        known_vip = _extract_first_live_ip(discovery, fallback=VICTIM_REAL_IP)
        if known_vip != VICTIM_REAL_IP:
            break
        log(f"Attacker (rescan): discovery try {_try} found no live VIP, retrying")
        time.sleep(2)
    log(f"Attacker (rescan): initial discovery believes victim is at {known_vip}")

    start = time.time()
    attempt = 0
    while time.time() - start < EXPERIMENT_DURATION_SECONDS:
        attempt += 1
        attempt_start = time.time()
        with h1_lock:
            out = h1.cmd(f"{NMAP} -n -p 22,80,443 --open -T4 {known_vip} 2>&1")
        print("================ RAW NMAP OUTPUT ================")
        print(out)
        print(repr(out))
        print("================================================")
        elapsed = time.time() - attempt_start
        service_found = "open" in out
        ports_found = out.count("open")

        # Evaluator-only check: compare remembered IP with VIP active
        # when this scan finishes. The attacker never receives this value.
        try:
            resp = requests.get(
                f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/current_vip",
                headers={"X-MTD-Key": MTD_API_KEY}, timeout=2,
            )
            resp.raise_for_status()
            active_vip_at_scan_end = resp.json().get("vip")
        except Exception as e:
            active_vip_at_scan_end = None
            log(f"Evaluator: current VIP lookup failed ({e})")

        hit = (
            service_found
            and active_vip_at_scan_end is not None
            and known_vip == active_vip_at_scan_end
        )

        results.append({
            "mode": mode, "role": "attacker", "attacker_strategy": "rescan_last_known",
            "attempt": attempt, "timestamp": attempt_start + elapsed, "target_ip": known_vip,
            "active_vip_at_scan_end": active_vip_at_scan_end,
            "service_found": service_found, "hit": hit,
            "ports_found": ports_found, "scan_duration_s": elapsed,
        })
        log(f"Attacker (rescan) attempt {attempt}: target={known_vip} hit={hit} "
            f"ports_found={ports_found} duration={elapsed:.2f}s")

        time.sleep(ATTACKER_SCAN_INTERVAL)


def run_attacker_sweep(h1, mode, results, h1_lock):
    """
    Attacker strategy 2: pool_sweep (must-fix #3).

    A more capable adversary: instead of trusting one remembered address,
    it re-sweeps the entire candidate VIP range on every attempt. This is
    strictly harder for the defense to beat than rescan_last_known, and
    reporting only the weaker strategy's number would be a strawman
    comparison an examiner or reviewer would flag immediately.

    h1_lock: see run_attacker_rescan()'s docstring — same shared-host
    thread-safety concern, since this function targets h1 too.
    """
    log("Attacker (sweep): starting periodic full-pool sweeps")
    start = time.time()
    attempt = 0
    while time.time() - start < EXPERIMENT_DURATION_SECONDS:
        attempt += 1
        attempt_start = time.time()
        with h1_lock:
            out = h1.cmd(f"{NMAP} -n -p 22,80,443 --open -T4 10.0.0.100-149 2>&1")
        print("================ RAW NMAP OUTPUT ================")
        print(out)
        print(repr(out))
        print("================================================")
        elapsed = time.time() - attempt_start
        live_ip = _extract_first_live_ip(out)
        ports_found = out.count("open")

        # Evaluator-only check: compare discovery with the VIP active
        # when the scan finishes. The attacker never receives this value.
        try:
            resp = requests.get(
                f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/current_vip",
                headers={"X-MTD-Key": MTD_API_KEY}, timeout=2,
            )
            resp.raise_for_status()
            active_vip_at_scan_end = resp.json().get("vip")
        except Exception as e:
            active_vip_at_scan_end = None
            log(f"Evaluator: current VIP lookup failed ({e})")

        hit = (
            live_ip is not None
            and active_vip_at_scan_end is not None
            and live_ip == active_vip_at_scan_end
            and ports_found > 0
        )

        results.append({
            "mode": mode, "role": "attacker", "attacker_strategy": "pool_sweep",
            "attempt": attempt, "timestamp": attempt_start + elapsed, "target_ip": live_ip,
            "active_vip_at_scan_end": active_vip_at_scan_end,
            "hit": hit, "ports_found": ports_found, "scan_duration_s": elapsed,
        })
        log(f"Attacker (sweep) attempt {attempt}: found={live_ip} hit={hit} "
            f"ports_found={ports_found} duration={elapsed:.2f}s")

        time.sleep(ATTACKER_SWEEP_INTERVAL)


def _extract_first_live_ip(nmap_output, fallback=None):
    for line in nmap_output.splitlines():
        if "Nmap scan report for" in line:
            return line.split()[-1].strip("()")
    return fallback  # None = genuine miss; callers that need a starting
                      # assumption (initial discovery) pass an explicit fallback


# ----------------------------------------------------------------------
# LEGITIMATE CLIENT — polls the registry, measures continuity
# ----------------------------------------------------------------------
def run_legit_client(h3, mode, results):
    start = time.time()
    attempt = 0
    while time.time() - start < EXPERIMENT_DURATION_SECONDS:
        attempt += 1
        attempt_start = time.time()
        try:
            resp = requests.get(
                f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/current_vip",
                headers={"X-MTD-Key": MTD_API_KEY}, timeout=2,
            )
            current_vip = resp.json().get("vip", VICTIM_REAL_IP)
        except Exception as e:
            current_vip = None
            log(f"Legit client: registry lookup failed ({e})")

        success = False
        if current_vip:
            out = h3.cmd(f"ping -c 1 -W 2 {current_vip} 2>&1")
            success = "1 received" in out or "1 packets received" in out

        results.append({
            "mode": mode, "role": "legit_client", "attempt": attempt,
            "timestamp": attempt_start, "target_ip": current_vip, "success": success,
        })
        log(f"Legit client attempt {attempt}: target={current_vip} success={success}")
        time.sleep(LEGIT_CLIENT_INTERVAL)


# ----------------------------------------------------------------------
# SCORE FEED — replays Step 6's real LSTM test scores on a clock, so the
# adaptive run is driven by genuine detector output, not synthetic data
#
# Why LSTM and not EWMA: Step 7's evaluation showed LSTM outperforming
# EWMA across all metrics, so the final architecture uses LSTM as the
# primary runtime detector; EWMA is retained only as a baseline
# comparison in the ML pipeline, not as the score source here. This is a
# deliberate architectural choice carried over from Steps 1-8 (frozen),
# not an oversight.
# ----------------------------------------------------------------------
def feed_scores(mode):
    if mode != "adaptive":
        log("Static mode: no score feed running (hopping stays disabled).")
        return

    try:
        import pandas as pd
        df = pd.read_csv("step6_lstm_predictions.csv", parse_dates=["target_minute"])
        agg = (df.groupby("target_minute")["predicted_proba"].max()
               .reset_index().sort_values("target_minute"))
    except FileNotFoundError:
        log("step6_lstm_predictions.csv not found — cannot run adaptive mode "
            "without a real score feed. Falling back to a flat synthetic score "
            "for smoke-testing the pipeline only (NOT a valid experiment run).")
        agg = None

    start = time.time()
    idx = 0
    while time.time() - start < EXPERIMENT_DURATION_SECONDS:
        if agg is not None and idx < len(agg):
            score = float(agg.iloc[idx]["predicted_proba"])
            idx += 1
        else:
            score = 0.1  # synthetic idle score if no real data available
        print(f"[TRACE-1] feed_scores about to POST idx={idx-1} score={score}", flush=True)
        try:
            resp = requests.post(
                f"http://{CONTROLLER_IP}:{CONTROLLER_REST_PORT}/mtd/score",
                json={"score": score, "timestamp": datetime.now().isoformat()},
                timeout=2,
            )
            print(f"[TRACE-2] status={resp.status_code} body={resp.text}", flush=True)
            resp.raise_for_status()
        except Exception as e:
            print(f"[TRACE-2] POST EXCEPTION: {e}", flush=True)
            log(f"Score feed POST failed: {e}")
        time.sleep(SCORE_POLL_INTERVAL_SECONDS)  # see must-fix #4 note above config block

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def run_single_trial(mode, trial_num, cli=False):
    """One full experiment run: build topology, run both attacker
    strategies plus the legit client concurrently, tear down, return rows.
    Factored out of main() so --trials can call it in a loop (must-fix #5)."""
    net, h1, h2, h3, h2_service = build_topology()
    results = []
    h1_lock = threading.Lock()  # serializes both attacker strategies' access
                                 # to the shared h1 Mininet shell — see the
                                 # docstrings on run_attacker_rescan/sweep

    score_thread = threading.Thread(target=feed_scores, args=(mode,), daemon=True)
    score_thread.start()

    rescan_thread = threading.Thread(target=run_attacker_rescan, args=(h1, mode, results, h1_lock))
    sweep_thread = threading.Thread(target=run_attacker_sweep, args=(h1, mode, results, h1_lock))
    legit_thread = threading.Thread(target=run_legit_client, args=(h3, mode, results))
    rescan_thread.start()
    sweep_thread.start()
    legit_thread.start()

    try:
        for t in [rescan_thread, sweep_thread, legit_thread]:
            t.join()

        for r in results:
            r["trial"] = trial_num

        if cli:
            CLI(net)
    finally:
        # try/finally so a hung/interrupted trial still cleans up the
        # victim's HTTP service and the Mininet topology instead of
        # leaking a "python3 -m http.server 80" process (and stale
        # network state) into the next run.
        _stop_victim_service(h2_service)
        net.stop()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "adaptive"], required=True)
    parser.add_argument("--trials", type=int, default=5,
                         help="Repeated runs of this mode, different RNG each time "
                              "(must-fix #5: a single run is a pilot, not a result)")
    parser.add_argument("--cli", action="store_true",
                         help="Drop into Mininet CLI after the LAST trial")
    args = parser.parse_args()

    setLogLevel("info")

    all_results = []
    try:
        for trial_num in range(1, args.trials + 1):
            log(f"=== Starting trial {trial_num}/{args.trials} (mode={args.mode}) ===")
            drop_to_cli = args.cli and trial_num == args.trials
            trial_results = run_single_trial(args.mode, trial_num, cli=drop_to_cli)
            all_results.extend(trial_results)
            log(f"=== Finished trial {trial_num}/{args.trials}: {len(trial_results)} rows ===")
    finally:
        # Unlike Ryu (which you started/stopped yourself in another
        # terminal), this script owns the Step 9 subprocess's lifecycle,
        # so it must clean it up here even if a trial raises.
        stop_mtd_controller()

    out_file = f"step10_experiment_{args.mode}.csv"
    if all_results:
        keys = sorted({k for r in all_results for k in r.keys()})
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        log(f"Saved: {out_file} ({len(all_results)} rows across {args.trials} trials)")


if __name__ == "__main__":
    main()
