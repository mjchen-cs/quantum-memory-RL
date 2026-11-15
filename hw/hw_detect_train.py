# -*- coding: utf-8 -*-
"""
Task 2 - reward-only training for DETECTION of a hidden quantum signal (hardware-in-the-loop, like hw_train.py).

Episode: with probability 1/2 the agent receives m pairs of copies of the maximally mixed state I/d (no signal), else
m pairs of the signal state rho_{sP} = (I + s eps P)/d with a hidden uniformly random non-identity Pauli P and a sign s
shared by the two copies of a pair (shared Pauli twirl, fresh for every pair). The agent answers "signal" / "no signal"
and gets reward 1 if correct. Because of the twirl, a single copy of the signal state averages to I/d exactly: no
single-copy measurement carries any information, and the agent can only succeed through two-copy (quantum-memory)
processing.

Agent: the same site-local front-end V(theta) and decoder tables as task 1; detection head
    logit = a * (logsumexp_Q score_Q - log N) + b,   P(signal) = sigmoid(logit),
which is the Bayes-optimal mixture likelihood-ratio test when score_Q is the log-likelihood ratio of hypothesis Q.
Learning signal: reward only. Front-end: antithetic ES with common random numbers; decoder + head: REINFORCE,
with the decoder restarts, step-size annealing and front-end restarts of hw_train.py.

Circuits per front-end and iteration: C signal circuits per Pauli (twirled) + C0 null circuits (each copy prepared in
a uniformly random computational basis state, i.e. I/d exactly in expectation).
Usage:  python hw_detect_train.py --backend sim --iters 150 --out runs/detect_sim.json
        python hw_detect_train.py --backend origin --iters 150 --seed 50 --out runs/detect_s50.json --fe_restart_patience 40
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
from fast_site import SiteModel, NPAR                                                       # noqa: E402
from hw_core import prep_ops                                                                # noqa: E402
from hw_train import (Decoder, make_preps, front_end_ops, single_copy_visibility, SimBackend,  # noqa: E402
                      OriginBackend)

torch.set_default_dtype(torch.float64)


class DetectAgent(torch.nn.Module):
    def __init__(self, n, seed):
        super().__init__()
        self.dec = Decoder(n, seed=seed)
        self.a = torch.nn.Parameter(torch.tensor(1.0)); self.b = torch.nn.Parameter(torch.tensor(0.0))
        self.N = 4 ** n - 1

    def logit(self, b):
        sc = self.dec.scores(b)
        return self.a * (torch.logsumexp(sc, 1) - np.log(self.N)) + self.b


def null_preps(n, C0, rng):
    """C0 preparations of two copies of I/d: every qubit in a uniformly random computational basis state."""
    out = []
    for _ in range(C0):
        bits = rng.integers(2, size=2 * n)
        out.append(prep_ops([(0, int(x)) for x in bits[:n]], 0) + prep_ops([(0, int(x)) for x in bits[n:]], n))
    return out


def det_episodes(sig, null, nP, C, C0, m, E, rng):
    """sig: list over (Pauli, circuit) of (S, n) digit arrays; null: list of C0 such arrays.
    Returns b (E, m, n) and labels y (E,) (1 = signal)."""
    y = rng.integers(2, size=E); b = []
    for yy in y:
        if yy:
            t = rng.integers(nP); cs = rng.choice(C, size=m, replace=False)
            b.append([sig[t * C + c][rng.integers(len(sig[t * C + c]))] for c in cs])
        else:
            cs = rng.choice(C0, size=m, replace=False)
            b.append([null[c][rng.integers(len(null[c]))] for c in cs])
    return np.array(b), y


def act_bin(logit, rng):
    p = torch.sigmoid(logit).detach().numpy()
    return (rng.random(len(p)) < p).astype(int)


def reinforce_steps(agent, opt, sig, null, nP, C, C0, m, E, steps, rng, ent, base):
    for _ in range(steps):
        b, y = det_episodes(sig, null, nP, C, C0, m, E, rng)
        lg = agent.logit(torch.tensor(b)); ac = act_bin(lg, rng); r = (ac == y).astype(float)
        lp = torch.nn.functional.logsigmoid(lg); ln = torch.nn.functional.logsigmoid(-lg)
        logpa = torch.where(torch.tensor(ac == 1), lp, ln)
        p = torch.sigmoid(lg); H = -(p * lp + (1 - p) * ln)
        loss = -(torch.tensor(r - base) * logpa).mean() - ent * H.mean()
        opt.zero_grad(); loss.backward(); opt.step(); base = 0.9 * base + 0.1 * r.mean()
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "origin"], default="sim")
    ap.add_argument("--qpu", default="WK_C180")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--m", type=int, default=8, help="pairs per episode")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--C", type=int, default=8, help="signal circuits per Pauli and front-end")
    ap.add_argument("--C0", type=int, default=64, help="null circuits per front-end")
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--E", type=int, default=3000)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--lr_th", type=float, default=0.08)
    ap.add_argument("--lr_dec", type=float, default=0.05)
    ap.add_argument("--dec_steps", type=int, default=3)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--dec_restart_every", type=int, default=20)
    ap.add_argument("--dec_restart_cands", type=int, default=3)
    ap.add_argument("--dec_restart_steps", type=int, default=600)
    ap.add_argument("--anneal", type=float, default=0.62, help="10-iteration mean reward that starts step-size annealing")
    ap.add_argument("--fe_restart_patience", type=int, default=0)
    ap.add_argument("--fe_restart_below", type=float, default=0.58)
    ap.add_argument("--fe_restart_gain", type=float, default=0.03)
    ap.add_argument("--no_twirl", action="store_true", help="ablation: signal pairs without the shared random sign")
    ap.add_argument("--p_dep", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--chip", default="")
    ap.add_argument("--init", default="", help="json with theta (and h, c) to warm start, e.g. runs/train_s20.json")
    ap.add_argument("--head_steps", type=int, default=600, help="with --init: REINFORCE steps fitting the head first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.chip:
        a.chip = os.path.join(HERE, "runs", "chip.json" if a.qpu == "WK_C180" else f"chip_{a.qpu}.json")
    n, eps = a.n, 1.0
    S_model = SiteModel(n, eps=eps, twirl=True)
    Ps = [tuple(int(x) for x in p) for p in S_model.P]; nP = len(Ps)
    rng0 = np.random.default_rng(a.seed)
    th = rng0.uniform(-np.pi, np.pi, NPAR)
    agent = DetectAgent(n, a.seed)
    if a.init:
        d = json.load(open(a.init)); src = d.get("best", d); th = np.array(src["theta"], float)
        if "h" in src:
            with torch.no_grad():
                agent.dec.h.copy_(torch.tensor(src["h"])); agent.dec.c.copy_(torch.tensor(src["c"]))
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr_dec)
    state = dict(args=vars(a), hist=[], it=0, task="detect")
    if a.init:
        state["anneal_from"] = -100        # transfer from a converged task-1 agent: smallest front-end step size
    mT = np.zeros(NPAR); vT = np.zeros(NPAR); base = 0.5
    if os.path.exists(a.out):
        state = json.load(open(a.out))
        th = np.array(state["theta"]); mT = np.array(state["mT"]); vT = np.array(state["vT"]); base = state["base"]
        agent.load_state_dict({k: torch.tensor(v) for k, v in state["agent"].items()})
        if os.path.exists(a.out + ".opt.pt"):
            opt.load_state_dict(torch.load(a.out + ".opt.pt"))
        print(f"resuming at iteration {state['it']}", flush=True)
    if a.backend == "sim":
        be = SimBackend(n, a.p_dep, a.seed)
    else:
        if state.get("phys"):
            phys = state["phys"]
        else:
            from hw_noise import load_chip, select_sites
            _, s1, f2 = load_chip(a.chip); sites = select_sites(n, s1, f2)
            phys = [s["a"] for s in sites] + [s["b"] for s in sites]
        state["phys"] = phys; state["qpu"] = a.qpu
        be = OriginBackend(n, phys, a.batch, a.qpu)
        print("backend", a.qpu, "pinned qubits", phys, flush=True)
    per = nP * a.C + a.C0; per_iter = (2 * a.K + 1) * per
    print(f"{per_iter} circuits x {a.S} shots per iteration; billing model {per_iter * 0.010 + per_iter * a.S * 0.00045:.1f} s",
          flush=True)
    first_it = state["it"] + 1; t0 = time.time()
    for it in range(first_it, a.iters + 1):
        rng = np.random.default_rng([a.seed, it, 17])
        preps = make_preps(n, Ps, a.C, eps, rng, twirl=not a.no_twirl) + null_preps(n, a.C0, rng)
        dirs = [rng.normal(0, a.sigma, NPAR) for _ in range(a.K)]
        fronts = [th] + [th + s * d for d in dirs for s in (1, -1)]
        circuits = [p + front_end_ops(f, n) for f in fronts for p in preps]
        data, info = be.run(circuits, a.S, pending_file=a.out + ".pending.json", it=it)
        D = [data[i * per:(i + 1) * per] for i in range(len(fronts))]
        split = lambda Df: (Df[:nP * a.C], Df[nP * a.C:])
        seed_ep = int(rng.integers(2 ** 31)); rewards = []
        with torch.no_grad():
            for Df in D:
                r_ = np.random.default_rng(seed_ep); sg, nl = split(Df)
                b, y = det_episodes(sg, nl, nP, a.C, a.C0, a.m, a.E, r_)
                rewards.append(float((act_bin(agent.logit(torch.tensor(b)), r_) == y).mean()))
        g = np.zeros(NPAR)
        for k, d in enumerate(dirs):
            g += (rewards[1 + 2 * k] - rewards[2 + 2 * k]) / (2 * a.sigma ** 2) * d
        g /= a.K
        mT = 0.9 * mT + 0.1 * g; vT = 0.999 * vT + 0.001 * g ** 2
        recent = [h_["reward"] for h_ in state["hist"][-9:]] + [rewards[0]]
        if state.get("anneal_from") is None and len(recent) == 10 and np.mean(recent) > a.anneal:
            state["anneal_from"] = it
        lr_mult = max(0.125, 0.5 ** ((it - state["anneal_from"]) // 20 + 1)) if state.get("anneal_from") is not None else 1.0
        th = th + lr_mult * a.lr_th * (mT / (1 - 0.9 ** it)) / (np.sqrt(vT / (1 - 0.999 ** it)) + 1e-8)
        fe_restart = None
        if a.fe_restart_patience:
            r10 = float(np.mean(recent)); fr = state.setdefault("fe", dict(start=it, best=-1.0, best_it=it, restarts=[]))
            if it - fr["start"] >= 10 and r10 > fr["best"] + a.fe_restart_gain:
                fr["best"], fr["best_it"] = r10, it
            if r10 < a.fe_restart_below and it - fr["best_it"] >= a.fe_restart_patience and it - fr["start"] >= a.fe_restart_patience:
                th = np.random.default_rng([a.seed, it, 13]).uniform(-np.pi, np.pi, NPAR); mT[:] = 0; vT[:] = 0
                fr["restarts"].append(it); fr.update(start=it, best=-1.0, best_it=it); state["anneal_from"] = None
                fe_restart = it; print(f"    front-end restart at iteration {it}", flush=True)
        sg0, nl0 = split(D[0])
        if a.init and it == 1 and a.head_steps:
            # transfer: the task-1 decoder scores are not calibrated log-likelihood ratios (a softmax over hypotheses
            # ignores their scale and offset), so first fit the detection head and decoder on this iteration's data
            base = reinforce_steps(agent, opt, sg0, nl0, nP, a.C, a.C0, a.m, 1000, a.head_steps, rng, a.ent, base)
        base = reinforce_steps(agent, opt, sg0, nl0, nP, a.C, a.C0, a.m, a.E, a.dec_steps, rng, a.ent, base)
        restart = None
        if a.dec_restart_every and it % a.dec_restart_every == 0:
            def score(ag):
                r_ = np.random.default_rng([a.seed, it, 99])
                with torch.no_grad():
                    b, y = det_episodes(sg0, nl0, nP, a.C, a.C0, a.m, 6000, r_)
                    return float((act_bin(ag.logit(torch.tensor(b)), r_) == y).mean())
            cur = score(agent); cands = []
            for j in range(a.dec_restart_cands):
                ag = DetectAgent(n, int(1000 * a.seed + 10 * it + j)); op = torch.optim.Adam(ag.parameters(), lr=a.lr_dec)
                reinforce_steps(ag, op, sg0, nl0, nP, a.C, a.C0, a.m, 1000, a.dec_restart_steps, rng, a.ent, 0.5)
                cands.append((score(ag), ag))
            bs, bag = max(cands, key=lambda x: x[0])
            restart = dict(current=cur, cands=[round(c[0], 3) for c in cands], accepted=bool(bs > cur + 0.02))
            if restart["accepted"]:
                agent.load_state_dict(bag.state_dict()); opt = torch.optim.Adam(agent.parameters(), lr=a.lr_dec); base = bs
            print(f"    decoder restart at iteration {it}: {restart}", flush=True)
        vis = S_model.visibility(th)
        rec = dict(it=it, reward=rewards[0], vis_ideal=vis, vis_single=single_copy_visibility(th, n), lr_mult=lr_mult,
                   dec_restart=restart, fe_restart=fe_restart, info={k: v for k, v in info.items() if k != "job_ids"},
                   t=time.time() - t0)
        if a.backend == "sim":
            rec["qpu_s_model"] = be.qpu_s
        state["hist"].append(rec)
        rr = [h_["reward"] for h_ in state["hist"][-10:]]
        if len(rr) == 10 and np.mean(rr) > state.get("best", {}).get("mean_reward10", -1):
            state["best"] = dict(it=it, mean_reward10=float(np.mean(rr)), theta=th.tolist(),
                                 agent={k: v.tolist() for k, v in agent.state_dict().items()})
        state.update(it=it, theta=th.tolist(), mT=mT.tolist(), vT=vT.tolist(), base=base,
                     agent={k: v.tolist() for k, v in agent.state_dict().items()})
        torch.save(opt.state_dict(), a.out + ".opt.pt")
        json.dump(state, open(a.out, "w"), indent=1, default=str)
        print(f"it {it:4d}  reward(centre) {rewards[0]:.3f}  two-copy visibility {vis:.3f}  lr {lr_mult}"
              + ("" if a.backend == "sim" else f"  wall {info.get('wall_s', 0):.0f} s qpu {info.get('qpu_ms', 0)} ms"), flush=True)


if __name__ == "__main__":
    main()
