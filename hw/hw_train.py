# -*- coding: utf-8 -*-
"""
Reward-only training of the lightweight quantum agent ON THE CHIP (hardware-in-the-loop), hardware-feasible version
of stage 3 (e2e_site.py).

Agent (n-independent, 13 + 3x16 + 3 parameters): site-local front-end V(theta) on every pair (A_i, B_i) with the
shared Pauli twirl, computational-basis measurement, decoder score(a|b) = sum_k c_k prod_i h_k[a_i, b_i] summed over
the m pairs of an episode, action a ~ softmax(score). Reward 1{a = P}; the agent never sees P otherwise.

Cost on WK_C180 is about 10 ms per circuit plus about 0.45 ms per shot, so stage 3 (26,000 pair samples per
iteration) is replaced by:
  * every iteration measures 2K+1 front-ends (theta, theta +- sigma d_k) on all N Paulis, C circuits each, S shots per
    circuit; theta + d_k and theta - d_k use the SAME random state preparations and twirl signs (common random numbers);
  * an episode takes m DISTINCT circuits of one Pauli and one shot of each, so every pair is an exact, independent
    sample of the two-copy measurement; many episodes (one action and one reward each) are formed from the same
    circuits at no extra QPU cost;
  * front-end: antithetic ES on the mean reward; decoder: REINFORCE on the episodes of the centre front-end.

Backends: 'sim' (exact per-circuit sampling with the numpy simulator of hw_core plus a per-site depolarizing error
that mimics the measured two-copy visibility) and 'origin' (WK_C180, pinned qubits).
Usage:  python hw_train.py --backend sim --iters 300 --out runs/train_sim.json
        python hw_train.py --backend origin --iters 10 --out runs/train_pilot.json      (ORIGIN_API_KEY set)
The run can be interrupted and resumed: state is saved after every iteration (--out).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_site import SiteModel, NPAR, bell_theta          # noqa: E402
from hw_core import prep_ops, sample_copy, site_V_ops, outcome_probs, to_qprog, counts_to_logical  # noqa: E402

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ agent
class Decoder(torch.nn.Module):
    def __init__(self, n, K=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.K = n, K
        self.h = torch.nn.Parameter(1 + 0.05 * torch.randn(K, 4, 4, generator=g))
        self.c = torch.nn.Parameter(0.05 * torch.randn(K, generator=g))

    def scores(self, b):
        """b: (E, m, n) site digits -> (E, N) scores over the non-identity Paulis (SiteModel order)."""
        E, m, n = b.shape
        v = self.h[:, :, b.reshape(-1, n)].permute(2, 0, 3, 1)            # (E*m, K, n, 4)
        out = v[:, :, 0, :]
        for i in range(1, n):
            out = (out[..., :, None] * v[:, :, i, None, :]).reshape(E * m, self.K, -1)
        s = (self.c[None, :, None] * out).sum(1)[:, 1:]
        return s.reshape(E, m, -1).sum(1)


# ------------------------------------------------------------------ circuits
def make_preps(n, Ps, C, eps, rng, twirl=True):
    """C random preparations (two copies + shared twirl sign) per Pauli; reused by every front-end of an iteration.
    twirl=False (ablation): no shared random sign, so single-copy information about P is available to the agent."""
    preps = []
    for P in Ps:
        for _ in range(C):
            s = int(rng.choice([1, -1])) if twirl else 1
            ops = prep_ops(sample_copy(list(P), s, eps, rng), 0) + prep_ops(sample_copy(list(P), s, eps, rng), n)
            preps.append(ops)
    return preps


def single_copy_visibility(theta, n):
    """How much single-copy information the measured front-end keeps: mean over Paulis and outcomes of the squared
    single-copy terms (copy A or B alone), the analogue of SiteModel.visibility for the two-copy term. A front-end
    that just measures each copy separately scores high here and low on the two-copy visibility."""
    from fast_site import tables
    g = tables(np.asarray(theta, float))
    S = SiteModel(n)
    out = []
    for k in (0, 1):
        prod = np.ones((S.N, S.D))
        for i in range(n):
            prod *= g[k][S.P[:, i]][:, S.B[:, i]]
        out.append(float((prod ** 2).mean()))
    return max(out)


def front_end_ops(theta, n):
    ops = []
    for i in range(n):
        ops += site_V_ops(theta, i, n + i)
    return ops


def site_digits(idx, n):
    bits = [(idx >> (2 * n - 1 - q)) & 1 for q in range(2 * n)]
    return [2 * bits[i] + bits[n + i] for i in range(n)]


# ------------------------------------------------------------------ backends
class SimBackend:
    """Exact per-circuit outcome distribution (hw_core numpy simulator) + per-site depolarizing error p_dep on the
    measured pair (replaces the site digit by a uniform one), which scales the two-copy visibility by (1 - p_dep)."""
    def __init__(self, n, p_dep, seed):
        self.n, self.p, self.rng = n, p_dep, np.random.default_rng(seed + 991)
        self.qpu_s = 0.0

    def run(self, circuits, shots, pending_file=None, it=None):
        n, out = self.n, []
        for ops in circuits:
            pr = outcome_probs(ops, 2 * n); pr = pr / pr.sum()
            idx = self.rng.choice(len(pr), size=shots, p=pr)
            dig = np.array([site_digits(int(i), n) for i in idx])
            flip = self.rng.random(dig.shape) < self.p
            dig[flip] = self.rng.integers(0, 4, size=int(flip.sum()))
            out.append(dig)
        self.qpu_s += len(circuits) * 0.010 + len(circuits) * shots * 0.00045       # billing model of the chip
        return out, {}


class OriginBackend:
    def __init__(self, n, phys, batch=200, qpu="WK_C180"):
        from pyqpanda3 import qcloud
        key = os.environ.get("ORIGIN_API_KEY")
        if not key:
            raise SystemExit("set ORIGIN_API_KEY first (the key is never written to disk)")
        self.qc, self.n, self.phys, self.batch = qcloud, n, phys, batch
        self.be = qcloud.QCloudService(api_key=key).backend(qpu)
        o = qcloud.QCloudOptions(); o.set_amend(True); o.set_mapping(False); o.set_optimization(True)
        self.opt = o

    def run(self, circuits, shots, pending_file=None, it=None):
        """Submit all circuits (in jobs of `batch`), wait, return per-circuit site digits (shots x n).
        The job ids of the iteration are written to `pending_file` right after submission; if the script is stopped
        and restarted, the same iteration (same circuits, regenerated from the per-iteration seed) picks the jobs up
        again instead of submitting and paying for them twice."""
        n = self.n
        jobs = None
        if pending_file and os.path.exists(pending_file):
            pf = json.load(open(pending_file))
            if pf.get("it") == it and pf.get("n_circuits") == len(circuits) and pf.get("shots") == shots:
                jobs = [tuple(x) for x in pf["jobs"]]
                dead = []
                for _, jid in jobs:                      # jobs cancelled in the console (or failed) cannot be re-used
                    try:
                        nm = getattr(self.qc.QCloudJob(jid).status(), "name", "")
                    except Exception:
                        nm = ""
                    if nm.upper().startswith(("FAIL", "CANCEL")):
                        dead.append((jid, nm))
                if dead:
                    print(f"    iteration {it}: saved jobs {dead} are cancelled/failed; submitting the iteration again", flush=True)
                    jobs = None
                else:
                    print(f"    iteration {it}: re-attaching to {len(jobs)} jobs submitted before the restart", flush=True)
        if jobs is None:
            progs = [to_qprog(ops, 2 * n, self.phys) for ops in circuits]
            jobs = []
            queue = [(i, min(i + self.batch, len(progs))) for i in range(0, len(progs), self.batch)]
            while queue:
                lo, hi = queue.pop(0)
                for attempt in range(3):
                    try:
                        jobs.append((lo, self.be.run(progs[lo:hi], shots, self.opt).job_id())); break
                    except Exception as e:
                        print(f"    submit retry {attempt + 1} ({hi - lo} circuits): {e}", flush=True)
                        time.sleep(10 * (attempt + 1))
                else:
                    if hi - lo <= 25:
                        raise RuntimeError("submission failed")
                    mid = (lo + hi) // 2                       # rejected repeatedly: split the job and try again
                    print(f"    splitting {hi - lo} circuits into two jobs", flush=True)
                    queue[:0] = [(lo, mid), (mid, hi)]
            if pending_file:
                json.dump(dict(it=it, n_circuits=len(circuits), shots=shots, jobs=jobs,
                               submitted=time.strftime("%Y-%m-%d %H:%M:%S")), open(pending_file, "w"))
        out = [None] * len(circuits); t0 = time.time(); info = dict(job_ids=[j for _, j in jobs], qpu_ms=0)
        pending = dict(jobs); last_report = t0; status = {}
        while pending:
            time.sleep(5)
            for i, jid in list(pending.items()):
                try:
                    st = self.qc.QCloudJob(jid).status(); name = getattr(st, "name", str(st)); status[jid] = name
                except Exception as e:                    # transient cloud/network error: keep polling
                    self.poll_errors = getattr(self, "poll_errors", 0) + 1
                    if self.poll_errors % 10 == 1:
                        print(f"    status query failed ({e}); retrying", flush=True)
                    if self.poll_errors > 500:
                        raise
                    continue
                if name == "FAILED" or name.upper().startswith("CANCEL"):
                    if pending_file and os.path.exists(pending_file):
                        try:
                            os.remove(pending_file)                 # do not re-attach to a dead job after restart
                        except OSError:
                            pass
                    raise RuntimeError(f"job {jid} {name}")
                if name != "FINISHED":
                    continue
                try:
                    res = self.qc.QCloudJob(jid).result()
                    probs = res.get_probs_list()
                except Exception as e:
                    print(f"    result query failed ({e}); retrying", flush=True)
                    continue
                try:
                    info["qpu_ms"] += int(res.timing_info().get("qpuRunTime", 0))
                except Exception:
                    pass
                rng = np.random.default_rng(int(jid[:8], 16) if all(ch in "0123456789abcdefABCDEF" for ch in jid[:8]) else 0)
                for k, pdict in enumerate(probs):
                    out[i + k] = probs_to_digits(pdict, shots, n, rng)
                del pending[i]
            if pending and time.time() - last_report > 60:
                cnt = {}
                for jid in pending.values():
                    cnt[status.get(jid, "?")] = cnt.get(status.get(jid, "?"), 0) + 1
                print(f"    iteration {it}: waiting {time.time() - t0:.0f} s, {len(jobs) - len(pending)}/{len(jobs)} jobs "
                      f"finished, others {cnt}", flush=True)
                last_report = time.time()
            if time.time() - t0 > 3600:
                raise RuntimeError("jobs not finished after one hour; rerun later (state and job ids are saved)")
        info["wall_s"] = time.time() - t0
        if pending_file and os.path.exists(pending_file):
            try:
                os.remove(pending_file)
            except OSError:
                pass
        return out, info


def probs_to_digits(pdict, shots, n, rng):
    """Readout-mitigated probabilities of one circuit (pyqpanda3 keys, cbit 0 rightmost) -> `shots` site-digit rows,
    resampled from the clipped and renormalized distribution."""
    keys, vals = [], []
    for key, v in pdict.items():
        idx = next(iter(counts_to_logical({key: 1}, 2 * n)))
        keys.append(idx); vals.append(max(float(v), 0.0))
    vals = np.array(vals)
    if vals.sum() <= 0:
        vals = np.ones(len(vals))
    pick = rng.choice(len(keys), size=shots, p=vals / vals.sum())
    return np.array([site_digits(int(keys[j]), n) for j in pick])


# ------------------------------------------------------------------ episodes
def episodes(data, nP, C, m, E, rng):
    """data: list over (Pauli, circuit) of (S, n) digit arrays. Returns b (E, m, n) and target index (E,).
    Each episode: one Pauli, m distinct circuits of that Pauli, one shot from each."""
    tgt = rng.integers(nP, size=E)
    b = []
    for t in tgt:
        cs = rng.choice(C, size=m, replace=False)
        b.append([data[t * C + c][rng.integers(len(data[t * C + c]))] for c in cs])
    return np.array(b), tgt


def decoder_restart(dec, D0, n, nP, a, it, rng):
    """Offline, purely classical: train `dec_restart_cands` freshly initialized decoders with REINFORCE (one sampled
    action and one reward per episode) on episodes built from the centre front-end's circuits of this iteration, then
    compare them with the current decoder on common held-out episode draws (again reward only). Escapes the plateau in
    which the front-end is already Bell-like but the decoder has converged to a poor local optimum."""
    def reinforce(d, steps):
        opt_ = torch.optim.Adam(d.parameters(), lr=a.lr_dec); b0 = 1.0 / nP
        for _ in range(steps):
            b, tgt = episodes(D0, nP, a.C, a.m, 1000, rng)
            sc = d.scores(torch.tensor(b)); ac = act(sc, rng); r = (ac == tgt).astype(float)
            logp = torch.log_softmax(sc, 1); pr = logp.exp()
            loss = -(torch.tensor(r - b0) * logp[torch.arange(len(r)), torch.tensor(ac)]).mean() \
                   - a.ent * (-(pr * logp).sum(1)).mean()
            opt_.zero_grad(); loss.backward(); opt_.step(); b0 = 0.9 * b0 + 0.1 * r.mean()
        return d

    def score(d):
        r_ = np.random.default_rng([a.seed, it, 99])
        with torch.no_grad():
            b, tgt = episodes(D0, nP, a.C, a.m, 6000, r_)
            return float((act(d.scores(torch.tensor(b)), r_) == tgt).mean())
    cur = score(dec)
    cands = [reinforce(Decoder(n, seed=int(1000 * a.seed + 10 * it + j)), a.dec_restart_steps) for j in range(a.dec_restart_cands)]
    sc = [score(d) for d in cands]
    j = int(np.argmax(sc)); acc = sc[j] > cur + 0.02
    out = dict(current=cur, cands=[round(x, 3) for x in sc], accepted=bool(acc), best=sc[j] if acc else cur)
    if acc:
        out.update(h=cands[j].h.detach().clone(), c=cands[j].c.detach().clone())
    return out


def act(sc, rng):
    pr = torch.softmax(sc, 1).detach().numpy()
    u = rng.random(len(pr))
    return (pr.cumsum(1) < u[:, None]).sum(1).clip(0, pr.shape[1] - 1)


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "origin"], default="sim")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--m", type=int, default=8, help="pairs per episode")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--K", type=int, default=2, help="ES directions per iteration (antithetic)")
    ap.add_argument("--C", type=int, default=8, help="circuits per Pauli and front-end")
    ap.add_argument("--S", type=int, default=20, help="shots per circuit")
    ap.add_argument("--E", type=int, default=2000, help="episodes formed per front-end (classical, free)")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--lr_th", type=float, default=0.08)
    ap.add_argument("--inject_rz", default="",
                    help="'dA,dB' radians: a Z rotation on every qubit of copy A / copy B between preparation and the "
                         "agent's front-end, i.e. a deliberate, known coherent miscalibration the agent is not told about "
                         "(online-adaptation experiment)")
    ap.add_argument("--inject_from", type=int, default=1, help="first iteration at which --inject_rz is applied")
    ap.add_argument("--reheat", type=float, default=0.0,
                    help="if annealing has started and the 5-iteration mean reward falls below anneal - reheat, go back "
                         "to the full step size (0 = off)")
    ap.add_argument("--warm_annealed", action="store_true",
                    help="with --init: start at the annealed (smallest) step size, as befits a converged agent; "
                         "--reheat restores the full step size if the reward collapses")
    ap.add_argument("--fe_restart_patience", type=int, default=0,
                    help="front-end restart: if the 10-iteration mean reward stays below --fe_restart_below and its best "
                         "has not improved by --fe_restart_gain for this many iterations, re-draw the front-end at "
                         "random (0 = off)")
    ap.add_argument("--fe_restart_below", type=float, default=0.5)
    ap.add_argument("--fe_restart_gain", type=float, default=0.05)
    ap.add_argument("--no_twirl", action="store_true",
                    help="ablation: no shared Pauli twirl (single-copy information available: the classical shortcut)")
    ap.add_argument("--anneal", type=float, default=0.0,
                    help="once the 10-iteration mean reward exceeds this value, halve the front-end step size every "
                         "20 iterations (down to 1/8); 0 = off. Adam-normalized ES keeps a constant step size, so without "
                         "annealing the converged front-end random-walks.")
    ap.add_argument("--lr_dec", type=float, default=0.05)
    ap.add_argument("--dec_steps", type=int, default=3)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--dec_restart_every", type=int, default=0,
                    help="every R iterations train fresh decoders offline (classical, reward only) on the centre "
                         "front-end data and keep the best if it earns more reward; 0 = off (version used for seeds 1, 10-12)")
    ap.add_argument("--dec_restart_cands", type=int, default=3)
    ap.add_argument("--dec_restart_now", action="store_true", help="also restart at the first iteration of this invocation")
    ap.add_argument("--dec_restart_steps", type=int, default=800)
    ap.add_argument("--p_dep", type=float, default=0.12, help="sim only: per-site depolarizing error")
    ap.add_argument("--init", default="random", help="random | bell | file.json (theta [, h, c])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200, help="circuits per cloud job (Origin recommends 200)")
    ap.add_argument("--qpu", default="WK_C180", help="Origin backend name (python list_backends.py prints them)")
    ap.add_argument("--chip", default="", help="calibration snapshot; default runs/chip.json (WK_C180) or runs/chip_<qpu>.json")
    ap.add_argument("--phys", default="", help="comma-separated physical qubits A0..,B0.. (origin); default: best couplers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.chip:
        a.chip = os.path.join(HERE, "runs", "chip.json" if a.qpu == "WK_C180" else f"chip_{a.qpu}.json")

    n, eps = a.n, 1.0
    S_model = SiteModel(n, eps=eps, twirl=True)
    Ps = [tuple(int(x) for x in p) for p in S_model.P]; nP = len(Ps)
    rng = np.random.default_rng(a.seed)
    dec = Decoder(n, seed=a.seed)
    opt = torch.optim.Adam(dec.parameters(), lr=a.lr_dec)
    th = rng.uniform(-np.pi, np.pi, NPAR)
    if a.init == "bell":
        th = bell_theta()
    elif a.init.endswith(".json"):
        # warm start from a trained agent (online adaptation / transfer to another chip): use its best front-end
        # if it recorded one, and its decoder
        d = json.load(open(a.init)); src = d.get("best", d)
        th = np.array(src["theta"], float)
        if "h" in src and "c" in src:
            with torch.no_grad():
                dec.h.copy_(torch.tensor(src["h"])); dec.c.copy_(torch.tensor(src["c"]))
    state = dict(args=vars(a), hist=[], it=0, init=a.init)
    if a.warm_annealed:
        state["anneal_from"] = -100        # warm start from a converged agent: begin at the smallest step size
    mT = np.zeros(NPAR); vT = np.zeros(NPAR); base = 1.0 / nP
    if os.path.exists(a.out):                                              # resume
        state = json.load(open(a.out))
        th = np.array(state["theta"]); mT = np.array(state["mT"]); vT = np.array(state["vT"]); base = state["base"]
        with torch.no_grad():
            dec.h.copy_(torch.tensor(state["h"])); dec.c.copy_(torch.tensor(state["c"]))
        rng = np.random.default_rng([a.seed, state["it"]])
        if os.path.exists(a.out + ".opt.pt"):
            opt.load_state_dict(torch.load(a.out + ".opt.pt"))
        print(f"resuming at iteration {state['it']}", flush=True)

    if a.backend == "sim":
        be = SimBackend(n, a.p_dep, a.seed)
    else:
        if state.get("phys"):
            phys = state["phys"]                                          # resuming: same qubits as before
        elif a.phys:
            phys = [int(x) for x in a.phys.split(",")]
        else:
            from hw_noise import load_chip, select_sites
            _, s1, f2 = load_chip(a.chip); sites = select_sites(n, s1, f2)
            phys = [s["a"] for s in sites] + [s["b"] for s in sites]
        if state.get("qpu", "WK_C180") != a.qpu and state.get("it", 0) > 0:
            raise SystemExit(f"{a.out} was trained on {state.get('qpu', 'WK_C180')}; refusing to continue it on {a.qpu}")
        state["phys"] = phys; state["qpu"] = a.qpu
        be = OriginBackend(n, phys, a.batch, a.qpu)
        print("backend", a.qpu, flush=True)
        print("pinned qubits", phys, flush=True)

    per_iter = (2 * a.K + 1) * nP * a.C
    print(f"{per_iter} circuits x {a.S} shots per iteration; billing model {per_iter * 0.010 + per_iter * a.S * 0.00045:.1f} s",
          flush=True)
    t_start = time.time()
    first_it = state["it"] + 1
    state.setdefault("versions", []).append(dict(from_it=first_it, dec_restart_every=a.dec_restart_every,
                                                 dec_restart_now=a.dec_restart_now, batch=a.batch, anneal=a.anneal,
                                                 dec_restart_cands=a.dec_restart_cands, dec_restart_steps=a.dec_restart_steps,
                                                 no_twirl=a.no_twirl, inject_rz=a.inject_rz,
                                                 inject_from=a.inject_from, reheat=a.reheat,
                                                 warm_annealed=a.warm_annealed,
                                                 fe_restart_patience=a.fe_restart_patience,
                                                 fe_restart_below=a.fe_restart_below, fe_restart_gain=a.fe_restart_gain))
    for it in range(state["it"] + 1, a.iters + 1):
        rng = np.random.default_rng([a.seed, it, 7])           # per-iteration stream: same circuits after a restart
        preps = make_preps(n, Ps, a.C, eps, rng, twirl=not a.no_twirl)
        dirs = [rng.normal(0, a.sigma, NPAR) for _ in range(a.K)]
        fronts = [th] + [th + s * d for d in dirs for s in (1, -1)]
        drift = []                                              # injected, known miscalibration (online-adaptation test)
        if a.inject_rz and it >= a.inject_from:
            dA, dB = (float(x) for x in a.inject_rz.split(","))
            for i in range(n):
                drift += [("RZ", (i,), (dA,)), ("RZ", (n + i,), (dB,))]
        circuits = [p + drift + front_end_ops(f, n) for f in fronts for p in preps]
        data, info = be.run(circuits, a.S, pending_file=a.out + ".pending.json", it=it)
        block = nP * a.C
        D = [data[i * block:(i + 1) * block] for i in range(len(fronts))]
        # common episode structure for all front-ends (same Paulis, circuits, shot positions, action randomness)
        seed_ep = int(rng.integers(2 ** 31))
        rewards = []
        with torch.no_grad():
            for Df in D:
                r_ = np.random.default_rng(seed_ep)
                b, tgt = episodes(Df, nP, a.C, a.m, a.E, r_)
                rewards.append(float((act(dec.scores(torch.tensor(b)), r_) == tgt).mean()))
        # front-end: antithetic ES with Adam
        g = np.zeros(NPAR)
        for k, d in enumerate(dirs):
            g += (rewards[1 + 2 * k] - rewards[2 + 2 * k]) / (2 * a.sigma ** 2) * d
        g /= a.K
        mT = 0.9 * mT + 0.1 * g; vT = 0.999 * vT + 0.001 * g ** 2
        lr_mult = 1.0
        if a.anneal:
            recent = [h_["reward"] for h_ in state["hist"][-9:]] + [rewards[0]]
            if state.get("anneal_from") is None and len(recent) == 10 and np.mean(recent) > a.anneal:
                state["anneal_from"] = it
            elif (state.get("anneal_from") is not None and a.reheat
                  and np.mean(recent[-5:]) < a.anneal - a.reheat):
                # performance collapsed (e.g. the device drifted): full step size again, anneal later from scratch
                state.setdefault("reheats", []).append(it); state["anneal_from"] = None
            if state.get("anneal_from") is not None:
                lr_mult = max(0.125, 0.5 ** ((it - state["anneal_from"]) // 20 + 1))
        th = th + lr_mult * a.lr_th * (mT / (1 - 0.9 ** it)) / (np.sqrt(vT / (1 - 0.999 ** it)) + 1e-8)
        fe_restart = None
        if a.fe_restart_patience:
            # front-end restart: the front-end has not learned (10-iteration mean reward below fe_restart_below) and the
            # best 10-iteration mean since the last (re)start has not improved by fe_restart_gain for `patience`
            # iterations -> it sits on a flat region with almost no two-copy signal; draw a fresh random front-end
            r10 = float(np.mean([h_["reward"] for h_ in state["hist"][-9:]] + [rewards[0]]))
            fr = state.setdefault("fe", dict(start=it, best=-1.0, best_it=it, restarts=[]))
            if it - fr["start"] >= 10 and r10 > fr["best"] + a.fe_restart_gain:
                fr["best"], fr["best_it"] = r10, it
            if (r10 < a.fe_restart_below and it - fr["best_it"] >= a.fe_restart_patience
                    and it - fr["start"] >= a.fe_restart_patience):
                r_fe = np.random.default_rng([a.seed, it, 13])
                th = r_fe.uniform(-np.pi, np.pi, NPAR); mT = np.zeros(NPAR); vT = np.zeros(NPAR)
                fr["restarts"].append(it); fr.update(start=it, best=-1.0, best_it=it)
                state["anneal_from"] = None
                fe_restart = dict(it=it, r10=r10)
                print(f"    front-end restart at iteration {it}: 10-iteration reward {r10:.3f} has not improved for "
                      f"{a.fe_restart_patience} iterations", flush=True)
        # decoder: REINFORCE on centre-front-end episodes (one action and one reward per episode)
        for _ in range(a.dec_steps):
            b, tgt = episodes(D[0], nP, a.C, a.m, a.E, rng)
            sc = dec.scores(torch.tensor(b)); act_ = act(sc, rng); r = (act_ == tgt).astype(float)
            logp = torch.log_softmax(sc, 1); pr = logp.exp()
            loss = -(torch.tensor(r - base) * logp[torch.arange(len(r)), torch.tensor(act_)]).mean() \
                   - a.ent * (-(pr * logp).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            base = 0.9 * base + 0.1 * r.mean()
        restart = None
        if (a.dec_restart_every and it % a.dec_restart_every == 0) or (a.dec_restart_now and it == first_it):
            restart = decoder_restart(dec, D[0], n, nP, a, it, rng)
            if restart["accepted"]:
                with torch.no_grad():
                    dec.h.copy_(restart.pop("h")); dec.c.copy_(restart.pop("c"))
                opt = torch.optim.Adam(dec.parameters(), lr=a.lr_dec); base = restart["best"]
            restart.pop("h", None); restart.pop("c", None)
            print(f"    decoder restart at iteration {it}: current {restart['current']:.3f}, fresh {restart['cands']}, "
                  f"{'accepted' if restart['accepted'] else 'kept current'}", flush=True)
        vis = S_model.visibility(th)
        rec = dict(it=it, reward=rewards[0], fe_restart=fe_restart, vis_ideal=vis, vis_single=single_copy_visibility(th, n), dec_restart=restart, info={k: v for k, v in info.items() if k != "job_ids"},
                   t=time.time() - t_start)
        if a.backend == "sim":
            rec["qpu_s_model"] = be.qpu_s
        state["hist"].append(rec)
        # best front-end so far by the 10-iteration mean reward (reward only), kept for deployment
        rr = [h_["reward"] for h_ in state["hist"][-10:]]
        if len(rr) == 10 and np.mean(rr) > state.get("best", {}).get("mean_reward10", -1):
            state["best"] = dict(it=it, mean_reward10=float(np.mean(rr)), theta=th.tolist(),
                                 h=dec.h.detach().tolist(), c=dec.c.detach().tolist())
        rec["lr_mult"] = lr_mult
        state.update(it=it, theta=th.tolist(), mT=mT.tolist(), vT=vT.tolist(), base=base,
                     h=dec.h.detach().tolist(), c=dec.c.detach().tolist())
        torch.save(opt.state_dict(), a.out + ".opt.pt")
        json.dump(state, open(a.out, "w"), indent=1, default=str)
        print(f"it {it:4d}  reward(centre) {rewards[0]:.3f}  ideal two-copy visibility of theta {vis:.3f}  "
              f"{'' if a.backend == 'sim' else 'wall %.0f s qpu %d ms' % (info.get('wall_s', 0), info.get('qpu_ms', 0))}",
              flush=True)


if __name__ == "__main__":
    main()
