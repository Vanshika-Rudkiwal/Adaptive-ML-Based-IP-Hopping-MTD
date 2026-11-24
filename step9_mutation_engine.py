"""
Step 9 — MTD Mutation Mechanism + Controller Coordination
CICIDS2017 PortScan / Adaptive MTD project
(FLASK + ovs-ofctl EDITION — Ryu removed, Python 3.10+ compatible)

WHY THIS REWRITE EXISTS
------------------------
Ryu 4.34 does not install on modern Python/setuptools. Rather than fight
that dependency, this file removes Ryu entirely and splits its two jobs
onto tools that don't have that problem:

  1. OpenFlow southbound (talk to the OVS switch: install/remove flow
     rules, learn about ARP requests) — now done with `ovs-ofctl`, the
     command-line tool that ships with every Open vSwitch install
     (Mininet already depends on it). No persistent OpenFlow controller
     session is required for this: ovs-ofctl pushes flow-table entries
     directly over OVS's local management socket.

  2. REST northbound (the API your ML process POSTs threat scores to)
     — now a plain Flask app instead of Ryu's WSGIApplication.

Everything ELSE is unchanged in behavior:
  - AddressPool (VIP selection + cooldown) — identical.
  - Step 8's AdaptiveHoppingEngine — imported unchanged, untouched.
  - NAT-style VIP<->real-IP flow translation — same match/action logic,
    expressed as ovs-ofctl flow strings instead of OFPFlowMod objects.
  - Grace-period retirement of the old VIP's flows — identical timing.
  - Default-deny flow for direct access to the real IP — identical.
  - hop_log, /mtd/score, /mtd/current_vip, /mtd/hop_log — identical
    JSON shapes and the same X-MTD-Key shared-secret gate.

WHAT GENUINELY CHANGES (and why it's not a functionality loss)
-----------------------------------------------------------------
- ARP handling: Ryu's packet_in_handler required an ACTIVE OpenFlow
  session so the controller could intercept ARP requests and hand-craft
  a reply. Without a controller library, we instead run OVS in
  "standalone" fail-mode (set in Step 10) so it behaves like a normal
  learning switch for everything by default, and add a small scapy
  sniff/reply thread that listens on the switch's own interface and
  answers ARP *only* for the current VIP — the same behavior (answering
  ARP-for-VIP as the victim's MAC), just implemented at the packet-capture
  layer instead of inside an OpenFlow controller callback.
- Table-miss + MAC learning: Ryu had to hand-implement a learning-switch
  fallback (see the old packet_in_handler's non-ARP branch). Standalone
  OVS already does MAC learning natively, so that hand-rolled code is
  simply not needed anymore — this is a simplification of *plumbing*,
  not of the security-relevant behavior.

ASSUMPTIONS YOU MUST VERIFY FOR YOUR TOPOLOGY (see the CONFIG block):
  - SWITCH_NAME matches the OVS bridge name Step 10's Mininet topology
    creates (Mininet's default is "s1" for a single switch).
  - VICTIM_PORT is the OVS **ofport number** the victim host is attached
    to. Run `sudo ovs-ofctl show s1` after Step 10 starts the topology
    to confirm it (it is usually, but not always, the same as the
    Mininet interface index).
  - SWITCH_IFACE_FOR_ARP is the interface scapy sniffs on to see ARP
    broadcasts. If Mininet is run in the root network namespace (the
    default), the bridge's own device name (e.g. "s1") is directly
    visible there and this works out of the box.

Run with:   sudo python3 step9_mutation_engine.py
Requires:   pip install flask scapy pandas
            (ovs-ofctl comes with openvswitch-switch, already required
            by Mininet — nothing new to install for that part)
Must run as root (or with CAP_NET_RAW/CAP_NET_ADMIN) because scapy needs
raw-socket access and ovs-ofctl needs to talk to the OVS bridge — same
privilege level Ryu + Mininet already required.

Startup order changed from the Ryu version: this process installs NO
flows at import time (see reinit() below for why). Step 10's
build_topology() launches this as a subprocess after Mininet exists, and
then calls POST /mtd/reinit once the OVS bridge is up — every trial, not
just the first. You do not need to start this process manually; Step 10
does it for you. It's still safe to launch manually for standalone
testing (e.g. against an already-running Mininet topology) — just POST
to /mtd/reinit yourself afterward.
"""

import json
import random
import time
import threading
import subprocess
import socket as socket_mod  # aliased: this module already imports `socket`
                              # implicitly via scapy/flask internals in some
                              # versions, and we don't want to shadow that.
                              # Used ONLY for if_nametoindex() diagnostics
                              # below (INSTRUMENTATION, temporary).
from collections import deque

from flask import Flask, request, jsonify
from scapy.all import sniff, sendp, Ether, ARP, get_if_list, conf

# ----------------------------------------------------------------------
# CONFIG — adjust to match your Mininet topology (see docstring above)
# ----------------------------------------------------------------------
VICTIM_REAL_IP = "10.0.0.2"       # victim's stable, never-exposed IP
VICTIM_MAC = "00:00:00:00:00:02"  # victim's MAC (from Mininet topology)
VICTIM_PORT = 2                   # OVS ofport number victim is attached to

SWITCH_NAME = "s1"                # OVS bridge name from Step 10's topology
SWITCH_IFACE_FOR_ARP = "s1"       # interface scapy sniffs/sends ARP on

# VIP_POOL size = 50 addresses. Justification: large enough that a single
# nmap sweep of the /24 takes noticeably longer than the hop interval
# (forcing the attacker to choose between "scan everything" and "re-hit
# the last known address"), small enough to fit in one lab-scale /24
# subnet alongside the attacker/victim/legit-client hosts. Not tuned for
# optimality — see the parameter-sensitivity note in Step 11 for the
# honest caveat: effectiveness at other pool sizes is untested.
VIP_POOL = [f"10.0.0.{i}" for i in range(100, 150)]  # candidate virtual IPs

# COOLDOWN_SIZE = 10 (20% of the pool). Justification: prevents the RNG
# from picking a VIP the attacker may have JUST scanned within the last
# few hops, without shrinking the effective pool so much that hops become
# near-deterministic. Chosen conservatively; not swept.
COOLDOWN_SIZE = 10               # don't reuse last N VIPs (avoid predictable cycling)

# GRACE_PERIOD_SECONDS = 8. Justification: must exceed one typical TCP
# request/response round trip on a Mininet-emulated LAN (sub-100ms) by a
# wide margin so in-flight legitimate connections aren't cut mid-session,
# while staying short relative to the shortest hop interval used in Step 8
# (so an old VIP isn't live for a large fraction of the next hop cycle).
# Chosen conservatively for continuity; sensitivity untested.
GRACE_PERIOD_SECONDS = 8         # old VIP flows kept alive this long after a hop

RANDOM_SEED = None  # None = OS entropy (do NOT use a fixed seed in a real
                     # deployment — a fixed seed is exactly the kind of
                     # predictability an adversary model should assume away)

REST_HOST = "0.0.0.0"
REST_PORT = 8080     # UNCHANGED from Ryu's WSGI default — Step 10/Step 11
                      # need zero URL changes because of this.

# ----------------------------------------------------------------------
# ACCESS CONTROL (must-fix #1) — unchanged from the Ryu version
# ----------------------------------------------------------------------
# Without this, "the attacker doesn't know the current VIP" is only true
# because the attacker's script chooses not to call an endpoint it is
# fully able to call. This shared-secret header is the load-bearing
# security property that makes the adversary model hold. It is
# intentionally NOT full PKI/mTLS — that would be disproportionate for a
# lab prototype; a static shared secret is the minimum control that makes
# "the attacker cannot read the registry" a true statement instead of an
# assumption.
#
# CHANGE THIS before running — do not use the default in any experiment
# whose results you intend to report.
MTD_API_KEY = "CHANGE-ME-lab-shared-secret"


def log(msg):
    print(f"[MTD] {msg}", flush=True)


def run_ofctl(args):
    """
    Run an `ovs-ofctl` command against the switch bridge. This is the
    direct replacement for Ryu sending OFPFlowMod messages over a live
    OpenFlow session: ovs-ofctl talks to the switch over OVS's local
    management socket, so no controller process or protocol session is
    required at all.
    """
    cmd = ["ovs-ofctl", "-O", "OpenFlow13"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ovs-ofctl error ({' '.join(cmd)}): {result.stderr.strip()}")
    return result


# ----------------------------------------------------------------------
# MUTATION STRATEGY — how the next VIP is chosen (UNCHANGED from Step 9)
# ----------------------------------------------------------------------
class AddressPool:
    def __init__(self, pool, cooldown_size=COOLDOWN_SIZE, seed=RANDOM_SEED):
        self.pool = list(pool)
        self.recent = deque(maxlen=cooldown_size)
        self.rng = random.Random(seed)

    def select_next(self, exclude=None):
        exclude = set(exclude or [])
        exclude |= set(self.recent)
        candidates = [ip for ip in self.pool if ip not in exclude]
        if not candidates:
            # cooldown exhausted the pool — relax constraint rather than fail
            candidates = [ip for ip in self.pool if ip != (self.recent[-1] if self.recent else None)]
        chosen = self.rng.choice(candidates)
        self.recent.append(chosen)
        return chosen


# ----------------------------------------------------------------------
# CONTROLLER — replaces Ryu's app_manager.RyuApp
# ----------------------------------------------------------------------
class MTDController:
    """
    Owns all controller state: AddressPool, current_vip, the Step 8
    engine, flow install/retire, and hop_log. Behaviorally identical to
    the Ryu RyuApp subclass it replaces. Only the transport for talking
    to the switch (ovs-ofctl subprocess calls instead of an OpenFlow
    session) and for answering ARP (a scapy sniff/reply thread instead
    of a packet_in handler) have changed.
    """

    def __init__(self):
        self.pool = AddressPool(VIP_POOL)
        self.current_vip = None
        self.previous_vip = None
        self.hop_count = 0
        self.hop_log = []  # for later export to CSV by Step 11
        # RLock (not Lock): reinit() and _perform_hop() both need to hold
        # this, and reinit() calls _perform_hop() directly on first init.
        self._lock = threading.RLock()
        # Guards against starting a second sniff thread on a later
        # reinit() call (e.g. trial 2, 3, ...) — one persistent ARP
        # responder thread is enough; it re-resolves SWITCH_IFACE_FOR_ARP
        # each time it (re)starts, see _start_arp_responder() below for
        # why even that single start has to wait and poll.
        self._arp_responder_started = False

        # Step 8's AdaptiveHoppingEngine imported UNCHANGED. Place
        # step8_adaptive_decision_engine.py on the same PYTHONPATH.
        #
        # Cadence note (unchanged from the Ryu version): this controller
        # only ever sees a new score when the external ML process POSTs
        # one to /mtd/score — in this prototype, once per minute (see
        # Step 10, SCORE_POLL_INTERVAL_SECONDS). The system's actual,
        # honest responsiveness is bounded by that polling interval, not
        # by the engine's internal tick granularity. Any claim in the
        # paper must say "reacts within one polling interval of a threat
        # score becoming available (currently 60s in this prototype)."
        from step8_adaptive_decision_engine import AdaptiveHoppingEngine, HoppingConfig
        self.engine = AdaptiveHoppingEngine(HoppingConfig())

        # NOTE: flows are NOT installed here, and the ARP responder is NOT
        # started here either. Ryu's switch_features_handler fired
        # automatically every time the OVS bridge (re)connected to the
        # controller — including on every Step 10 trial's
        # net.stop()/build_topology() cycle. ovs-ofctl has no equivalent
        # "bridge (re)appeared" notification, and neither does scapy: the
        # interface named by SWITCH_IFACE_FOR_ARP simply doesn't exist
        # yet at process-import time (this process starts before Mininet
        # builds the topology — see Step 10's ensure_mtd_controller()).
        # So both flow install AND the ARP sniff thread wait for an
        # explicit reinit() call (exposed as POST /mtd/reinit) that
        # Step 10 makes once per trial, right after the topology is
        # built. See Step 10's build_topology() for the caller.

    def reinit(self):
        """
        Call this once the OVS bridge named SWITCH_NAME exists (or has
        just been rebuilt). Installs the base flows, either performs the
        very first hop (current_vip is None) or re-installs the NAT flow
        for the VIP that was already active (so a mid-experiment bridge
        rebuild doesn't silently lose translation for the in-progress
        VIP), and — on the first call only — starts the ARP responder
        thread. This is the explicit stand-in for Ryu's
        switch_features_handler reconnect behavior.
        """
        with self._lock:
            self._install_base_flows()
            if self.current_vip is None:
                self._perform_hop(reason="initial")
            else:
                self._install_nat_flows(self.current_vip)
            if not self._arp_responder_started:
                self._arp_responder_started = True
                self._start_arp_responder()
            log(f"Reinitialized against switch '{SWITCH_NAME}', current_vip={self.current_vip}")

    # ------------------------------------------------------------------
    # Base flows: table-miss fallback + deny-direct-access-to-real-IP
    # ------------------------------------------------------------------
    def _install_base_flows(self):
        # Table-miss -> NORMAL. Standalone OVS (fail-mode=standalone, set
        # in Step 10) already does MAC learning on its own, so this just
        # makes the fallback explicit rather than implicit in fail-mode.
        run_ofctl(["add-flow", SWITCH_NAME, "priority=0,actions=NORMAL"])

        # must-fix #2 (unchanged): deny direct traffic to the real IP.
        # Without this rule, the hopping mechanism is entirely cosmetic —
        # an attacker who scans the real subnet directly (e.g. nmap
        # 10.0.0.0/24) reaches the unmoved real IP regardless of how often
        # the VIP changes. Priority 5 sits below the NAT flows (10), which
        # only ever match on VIP addresses, so legitimate VIP-translated
        # traffic to the real IP is untouched; this rule only catches
        # traffic that names the real IP directly.
        run_ofctl(["add-flow", SWITCH_NAME,
                   f"priority=5,dl_type=0x0800,nw_dst={VICTIM_REAL_IP},actions=drop"])
        log(f"Installed default-deny flow for direct access to real IP {VICTIM_REAL_IP}")
        log("Switch bridge configured, table-miss flow installed.")

    # ------------------------------------------------------------------
    # NAT-style flow install/retire for VIP <-> real IP translation
    # ------------------------------------------------------------------
    def _install_nat_flows(self, vip):
        # inbound: dst=VIP -> rewrite dst=REAL_IP, output to victim port
        run_ofctl(["add-flow", SWITCH_NAME,
                   f"priority=10,dl_type=0x0800,nw_dst={vip},"
                   f"actions=mod_nw_dst:{VICTIM_REAL_IP},output:{VICTIM_PORT}"])
        # outbound: src=REAL_IP -> rewrite src=VIP (so return traffic looks
        # like it's coming from the VIP the client expects), then let
        # normal L2 forwarding deliver it (equivalent of OFPP_NORMAL).
        run_ofctl(["add-flow", SWITCH_NAME,
                   f"priority=10,dl_type=0x0800,nw_src={VICTIM_REAL_IP},"
                   f"actions=mod_nw_src:{vip},NORMAL"])
        log(f"Installed NAT flows for VIP {vip} -> {VICTIM_REAL_IP}")

    def _retire_nat_flows_after_grace(self, old_vip):
        def _delayed_remove():
            time.sleep(GRACE_PERIOD_SECONDS)
            run_ofctl(["del-flows", SWITCH_NAME, f"dl_type=0x0800,nw_dst={old_vip}"])
            log(f"Retired old VIP {old_vip} after {GRACE_PERIOD_SECONDS}s grace period")
        threading.Thread(target=_delayed_remove, daemon=True).start()

    # ------------------------------------------------------------------
    # THE ACTUAL MUTATION (unchanged logic)
    # ------------------------------------------------------------------
    def _perform_hop(self, reason="scheduled"):
        with self._lock:
            new_vip = self.pool.select_next()
            old_vip = self.current_vip

            self._install_nat_flows(new_vip)
            if old_vip is not None:
                self._retire_nat_flows_after_grace(old_vip)

            self.previous_vip = old_vip
            self.current_vip = new_vip
            self.hop_count += 1

            record = {
                "hop_index": self.hop_count,
                "timestamp": time.time(),
                "old_vip": old_vip,
                "new_vip": new_vip,
                "reason": reason,
            }
            self.hop_log.append(record)
            log(f"HOP #{self.hop_count}: {old_vip} -> {new_vip} (reason={reason})")
            return record

    # ------------------------------------------------------------------
    # Called by the REST endpoint when a new threat score arrives
    # ------------------------------------------------------------------
    def on_score_received(self, score, timestamp):
        import pandas as pd
        ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
        decision = self.engine.update(score, ts)
        if decision.hop_now and decision.tier != "NORMAL":
            # only mutate when tier indicates elevated risk — avoid churning
            # the network on every NORMAL-tier tick
            self._perform_hop(reason=f"score={score:.3f}_tier={decision.tier}")
        return decision

    # ------------------------------------------------------------------
    # ARP responder — replaces Ryu's packet_in_handler + _send_arp_reply
    # ------------------------------------------------------------------
    def _start_arp_responder(self):
        def _handle(pkt):
            if ARP in pkt and pkt[ARP].op == 1:  # who-has
                dst_ip = pkt[ARP].pdst
                if dst_ip == self.current_vip:
                    reply = (
                        Ether(src=VICTIM_MAC, dst=pkt[Ether].src)
                        / ARP(op=2, hwsrc=VICTIM_MAC, psrc=dst_ip,
                              hwdst=pkt[Ether].src, pdst=pkt[ARP].psrc)
                    )
                    sendp(reply, iface=SWITCH_IFACE_FOR_ARP, verbose=False)

        def _iface_admin_up(name):
            """
            Checks the kernel's IFF_UP flag directly rather than
            inferring readiness from presence in get_if_list() alone —
            an interface can exist (and show up in that list) while
            still administratively down, which is exactly the state a
            freshly created standalone-OVS bridge netdev is left in
            until something explicitly brings it up (see Step 10's
            _bring_up_switch_interface()). Binding a raw socket to a
            down interface fails with OSError("Network is down"), which
            is what caused the earlier continuous retry/log loop.
            """
            try:
                with open(f"/sys/class/net/{name}/flags") as f:
                    flags = int(f.read().strip(), 16)
                return bool(flags & 0x1)  # IFF_UP
            except (FileNotFoundError, ValueError, OSError):
                return False

        def _wait_for_iface(timeout=30, poll_interval=0.5):
            """
            Poll until SWITCH_IFACE_FOR_ARP both exists AND is
            administratively up. reinit() is called right after Step 10
            does net.build() + s1.start([]) + bringing the bridge
            interface up, but there can still be a brief window where
            any of that hasn't fully landed in the kernel from this
            process's point of view yet.
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                if SWITCH_IFACE_FOR_ARP in get_if_list() and _iface_admin_up(SWITCH_IFACE_FOR_ARP):
                    return True
                time.sleep(poll_interval)
            return False

        def _sniff_loop():
            # Runs for the lifetime of the process (daemon thread), but
            # the bridge itself gets torn down and rebuilt once per Step
            # 10 trial. sniff() raises/returns when the interface it's
            # bound to disappears or goes down, so this loop waits for
            # the interface to become usable, sniffs until it isn't, then
            # waits again for the next trial's bridge. Backoff (capped,
            # reset on every successful sniff start) and logging only on
            # state changes replace the earlier continuous log spam when
            # the interface stays down for a stretch.
            backoff = 1
            max_backoff = 8
            iface_was_reachable = None  # None = no verdict logged yet
            while True:
                if not _wait_for_iface():
                    if iface_was_reachable is not False:
                        log(f"ARP responder: interface '{SWITCH_IFACE_FOR_ARP}' not up; "
                            f"still waiting (this message won't repeat until it recovers).")
                        iface_was_reachable = False
                    continue  # _wait_for_iface already paced itself over 30s

                iface_was_reachable = True
                backoff = 1  # reset now that we're about to sniff successfully
                log(f"ARP responder listening on {SWITCH_IFACE_FOR_ARP} for current VIP")

                # --- INSTRUMENTATION (temporary — root-cause verification) ---
                # Capture the kernel ifindex right before we hand the
                # interface name to scapy/sniff(). If OVS bridge
                # recreation (del-br/add-br) is producing a NEW kernel
                # netdev for the same name "s1", this value will differ
                # from the one printed after the ENODEV below. If OVS is
                # instead reusing the same netdev object (e.g. only
                # flapping admin state), this value will be identical
                # across the before/after prints.
                try:
                    pre_idx = socket_mod.if_nametoindex(SWITCH_IFACE_FOR_ARP)
                    log(f"[IFINDEX-PROBE] pre-sniff ts={time.time():.6f} "
                        f"iface={SWITCH_IFACE_FOR_ARP} ifindex={pre_idx}")
                except OSError as probe_err:
                    log(f"[IFINDEX-PROBE] pre-sniff ts={time.time():.6f} "
                        f"iface={SWITCH_IFACE_FOR_ARP} lookup failed: {probe_err}")
                # --- END INSTRUMENTATION ---

                try:
                    # Refresh Scapy's interface registry because Mininet recreates
                    # bridge devices between trials, changing kernel ifindex values.
                    conf.ifaces.reload()
                    sniff(iface=SWITCH_IFACE_FOR_ARP, filter="arp", prn=_handle, store=False)
                except OSError as e:
                    # --- INSTRUMENTATION (temporary — root-cause verification) ---
                    # Capture the ifindex again immediately after the
                    # ENODEV/OSError. Do this BEFORE any sleep/backoff so
                    # the timestamp is as close as possible to the actual
                    # failure, not to when we got around to retrying.
                    try:
                        post_idx = socket_mod.if_nametoindex(SWITCH_IFACE_FOR_ARP)
                        log(f"[IFINDEX-PROBE] post-ENODEV ts={time.time():.6f} "
                            f"iface={SWITCH_IFACE_FOR_ARP} ifindex={post_idx}")
                    except OSError as probe_err:
                        log(f"[IFINDEX-PROBE] post-ENODEV ts={time.time():.6f} "
                            f"iface={SWITCH_IFACE_FOR_ARP} lookup failed: {probe_err} "
                            f"(device does not currently exist under this name)")
                    # --- END INSTRUMENTATION ---

                    log(f"ARP responder: lost interface '{SWITCH_IFACE_FOR_ARP}' ({e}); "
                        f"backing off {backoff}s before re-checking.")
                    iface_was_reachable = None  # allow the next down-state to log again
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue
                # sniff() returning at all (no stop_filter set) also means
                # the interface is gone rather than a normal exit.
                log(f"ARP responder: sniff() on '{SWITCH_IFACE_FOR_ARP}' ended; "
                    f"will wait for the interface to become usable again.")
                iface_was_reachable = None

        threading.Thread(target=_sniff_loop, daemon=True).start()


# ----------------------------------------------------------------------
# REST API — replaces Ryu's WSGIApplication/ControllerBase
# ----------------------------------------------------------------------
mtd = MTDController()
app = Flask(__name__)


def _authorized(req):
    return req.headers.get("X-MTD-Key") == MTD_API_KEY


@app.route("/mtd/score", methods=["POST"])
def post_score():
    try:
        body = request.get_json(force=True)
        score = float(body["score"])
        timestamp = body.get("timestamp")
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        return jsonify({"error": str(e)}), 400

    decision = mtd.on_score_received(score, timestamp)
    return jsonify({
        "tier": decision.tier,
        "interval_seconds": decision.interval_seconds,
        "current_vip": mtd.current_vip,
        "hop_count": mtd.hop_count,
    })


@app.route("/mtd/current_vip", methods=["GET"])
def get_current_vip():
    # this is the "out-of-band registry" a legitimate client polls
    # instead of relying on DNS (explicit limitation, unchanged from the
    # original design).
    #
    # must-fix #1 (unchanged): gated by shared secret. This is the single
    # load-bearing security property of the whole system.
    if not _authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"vip": mtd.current_vip})


@app.route("/mtd/reinit", methods=["POST"])
def reinit_switch():
    # Gated the same as current_vip/hop_log: this reinstalls flows and
    # exposes the current VIP in the response, so it carries the same
    # controller-internal-state sensitivity as those two endpoints.
    if not _authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    mtd.reinit()
    return jsonify({"status": "reinitialized", "current_vip": mtd.current_vip})


@app.route("/mtd/hop_log", methods=["GET"])
def get_hop_log():
    # same reasoning as current_vip: the hop schedule is exactly the
    # controller-internal state the adversary model says the attacker
    # does not have. Gate it the same way.
    if not _authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(mtd.hop_log)


if __name__ == "__main__":
    # threaded=True so a hop's grace-period timer thread and incoming
    # /mtd/score requests can run concurrently, matching Ryu's
    # multi-threaded WSGI behavior.
    app.run(host=REST_HOST, port=REST_PORT, threaded=True)
