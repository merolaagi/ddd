"""
Fate-memory bench engine.

Arms (matched parameter count, identical starting weights):
  coop      W = S * softplus(V), S_ij = s_i s_j   sign-consistent, monotone (Hirsch / Angeli-Sontag)
  broken    same with one off-diagonal sign flipped -> exactly one negative cycle
  contract  W = (3.6 / ||V||_F) V                  contraction -> unique equilibrium
  free      W = V                                  unconstrained
  feedback  sign-consistent backbone with a few negative feedback edges, the pattern real metabolic
            pathways show: monotone wiring plus a handful of regulatory loops that break it
  pool      unconstrained wiring, but units are grouped into conserved pools whose totals never change,
            the circuit analogue of conserved quantities like ATP + ADP

Tasks:
  flipflop  3 channels, output = sign of last pulse on each channel
  xor       2 latched channels, output = product of the two latched signs
  parity    1 channel of identical pulses, output flips on every pulse
  dose      flipflop, but each sequence's pulses are scaled by a random gain (0.3x to 3x)
  background flipflop on top of a slowly wandering background level on every channel

Cell-inspired stress tests on the trained circuit:
  drift     sign-preserving multiplicative drift of weight magnitudes (enzyme levels vary)
  noise     Langevin noise on the state during the test rollout (gene-expression noise)
  division  every 40 steps the state is multiplied by random partition noise (cell division)

  inhibitor a drug partially blocks a random fraction of the circuit's units (activity x 0.3)

Membrane (input stage, identical for every arm):
  direct       inputs drive the circuit directly
  transporter  inputs split into positive and negative "molecules", each taken up through a saturating
               Michaelis-Menten transporter: intake = Vmax x / (Km + x), Vmax and Km learned per channel
  gated        transporter uptake further gated by channels the circuit itself opens or closes:
               gate = sigmoid(G h + g0). For the sign-consistent arm and its control, U and G carry signs
               s_i t_c and t_c s_j, so the closed loop through the membrane adds no negative cycle.

Drug tasks:
  commit      a drug arrives as brief spikes or sustained exposures; once its leaky accumulated exposure
              crosses a threshold the cell commits and the output switches on for good
  resistance  a drug makes the cell respond, but sustained exposure switches it into a drug-tolerant state that
              ignores the drug; a long enough drug holiday resensitizes it (reversible drug tolerance)
  antagonist  an agonist and a competing antagonist wander slowly; the output is on while the receptor
              is more than half occupied, A / (A + K (1 + B / Kb)) > 0.5 with K = 0.5, Kb = 1

Landscape: 2D principal-component map of settling trajectories (a Waddington landscape view).
"""
import time
import autograd.numpy as np
from autograd import grad
import numpy as onp

ARMS = ["coop", "broken", "contract", "free", "feedback", "pool"]
TASKS = {
    "flipflop": {"nin": 3, "nout": 3},
    "xor": {"nin": 2, "nout": 1},
    "parity": {"nin": 1, "nout": 1},
    "dose": {"nin": 3, "nout": 3},
    "background": {"nin": 3, "nout": 3},
    "commit": {"nin": 1, "nout": 1},
    "antagonist": {"nin": 2, "nout": 1},
    "resistance": {"nin": 1, "nout": 1},
}
MEMBRANES = ["direct", "transporter", "gated"]
DT = 0.5

DEFAULTS = {
    "tasks": ["flipflop", "xor", "parity"],
    "arms": ARMS,
    "hidden": 24,
    "substeps": 3,
    "iters": 500,
    "lr": 0.01,
    "batch": 32,
    "train_len": 60,
    "test_len": 400,
    "pulse_prob": 0.05,
    "self_excitation": 5.0,
    "coupling": 0.05,
    "drift": [0.1, 0.2, 0.3],
    "noise": [0.02, 0.05, 0.1],
    "division": [0.1, 0.25, 0.5],
    "inhibitor": [0.1, 0.25, 0.5],
    "membrane": "direct",
    "seed": 0,
    "repeats": 3,
    "feedback_edges": 3,
    "pool_size": 4,
}


class Cancelled(Exception):
    pass


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def sign_matrices(n, seed, feedback_edges=3):
    r = onp.random.default_rng(seed)
    s = r.choice([-1.0, 1.0], size=n)
    coop = onp.outer(s, s)
    broken = coop.copy()
    broken[0, 1] *= -1.0
    fb = coop.copy()
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    for k in r.permutation(len(off))[: max(0, int(feedback_edges))]:
        i, j = off[k]
        fb[i, j] *= -1.0
    return coop, broken, s, fb


def pool_projection(n, size, h0=0.5):
    """Group units into pools whose totals never change, the circuit analogue of a conserved quantity."""
    groups = [list(range(i, min(i + int(max(1, size)), n))) for i in range(0, n, int(max(1, size)))]
    M = onp.zeros((len(groups), n))
    for g, idx in enumerate(groups):
        M[g, idx] = 1.0
    totals = onp.array([len(g) * h0 for g in groups])

    def project(h):
        sums = h @ M.T
        factor = totals / (sums + 1e-9)
        return h * (factor @ M)

    return project, groups


def recurrent(arm, V, S):
    if arm == "coop":
        return S["coop"] * softplus(V)
    if arm == "broken":
        return S["broken"] * softplus(V)
    if arm == "feedback":
        return S["feedback"] * softplus(V)
    if arm == "contract":
        return 3.6 * V / np.sqrt(np.sum(V ** 2))
    return V


def init_params(arm, cfg, S, nin, nout):
    n = cfg["hidden"]
    mode = cfg.get("membrane", "direct")
    ncin = nin if mode == "direct" else 2 * nin
    r = onp.random.default_rng(cfg["seed"] + 1)
    mag = onp.abs(r.normal(cfg["coupling"], cfg["coupling"] * 0.4 + 1e-6, (n, n)))
    onp.fill_diagonal(mag, cfg["self_excitation"])
    w0 = S["coop"] * mag
    V = onp.log(onp.expm1(mag)) if arm in ("coop", "broken", "feedback") else w0.copy()
    U0 = r.normal(0, 1.0, (n, ncin))
    signed = mode == "gated" and arm in ("coop", "broken", "feedback")
    P = {
        "V": V,
        "U": onp.log(onp.expm1(onp.abs(U0) + 1e-3)) if signed else U0,
        "b": -0.5 * onp.diag(w0).copy(),
        "R": r.normal(0, 0.1, (nout, n)),
        "c": onp.zeros(nout),
    }
    if mode != "direct":
        P["vmax"] = onp.full(ncin, onp.log(onp.expm1(1.5)))
        P["km"] = onp.full(ncin, onp.log(onp.expm1(0.5)))
    if mode == "gated":
        G0 = r.normal(0, 0.3, (ncin, n))
        P["G"] = onp.log(onp.expm1(onp.abs(G0) + 1e-3)) if signed else G0
        P["g0"] = onp.full(ncin, 2.0)
    return P


def channel_signs(ncin, seed):
    return onp.random.default_rng(seed + 29).choice([-1.0, 1.0], size=ncin)


def make_drive(arm, S, mode):
    signed = mode == "gated" and arm in ("coop", "broken", "feedback")

    def drive(P, x, h):
        if mode == "direct":
            return x @ P["U"].T + P["b"]
        xs = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=-1)
        vmax, km = softplus(P["vmax"]), softplus(P["km"]) + 1e-3
        intake = vmax * xs / (km + xs)
        if signed:
            U = S["s"][:, None] * S["t"][None, :] * softplus(P["U"])
        else:
            U = P["U"]
        if mode == "gated":
            G = S["t"][:, None] * S["s"][None, :] * softplus(P["G"]) if signed else P["G"]
            intake = intake * sig(h @ G.T + P["g0"])
        return intake @ U.T + P["b"]

    return drive


def effective(arm, P, S, mode):
    signed = mode == "gated" and arm in ("coop", "broken", "feedback")
    out = {"U": onp.asarray(P["U"]), "b": onp.asarray(P["b"]), "R": onp.asarray(P["R"]), "c": onp.asarray(P["c"])}
    if mode != "direct":
        out["vmax"] = onp.log1p(onp.exp(P["vmax"]))
        out["km"] = onp.log1p(onp.exp(P["km"])) + 1e-3
        if signed:
            out["U"] = S["s"][:, None] * S["t"][None, :] * onp.log1p(onp.exp(P["U"]))
    if mode == "gated":
        out["G"] = S["t"][:, None] * S["s"][None, :] * onp.log1p(onp.exp(P["G"])) if signed else onp.asarray(P["G"])
        out["g0"] = onp.asarray(P["g0"])
    return out


def export_spec(arm, P, S, W, mode, cfg, tk):
    eff = effective(arm, P, S, mode)
    spec = {"format": "fate-memory-lab/cell-model", "version": 1, "arm": arm, "membrane": mode,
            "dt": DT, "substeps": cfg["substeps"], "hidden": int(W.shape[0]), "nin": tk["nin"], "nout": tk["nout"],
            "W": onp.round(W, 5).tolist(), "orthant": S["s"].tolist(), "channel_signs": S["t"].tolist(),
            "task": None}
    for k, v in eff.items():
        spec[k] = onp.round(v, 5).tolist()
    return spec


def uptake_curve(P):
    if "vmax" not in P:
        return None
    vmax = onp.log1p(onp.exp(P["vmax"]))
    km = onp.log1p(onp.exp(P["km"])) + 1e-3
    xs = onp.linspace(0, 3, 16)
    ys = (vmax[None, :] * xs[:, None] / (km[None, :] + xs[:, None])).mean(axis=1)
    return {"x": onp.round(xs, 3).tolist(), "y": onp.round(ys, 4).tolist(),
            "vmax": onp.round(vmax, 3).tolist(), "km": onp.round(km, 3).tolist()}


TOL_ON, TOL_DECAY, HOLIDAY = 5.0, 0.9, 25


def commit_truth(x):
    T, B = x.shape[0], x.shape[1]
    y = -onp.ones((T, B, 1))
    acc = onp.zeros(B)
    done = onp.zeros(B, dtype=bool)
    for t in range(T):
        acc = 0.85 * acc + x[t, :, 0]
        done |= acc > 3.2
        y[t, done, 0] = 1.0
    return y


def resistance_truth(x, tolerant0=None):
    T, B = x.shape[0], x.shape[1]
    y = -onp.ones((T, B, 1))
    exposure = onp.zeros(B)
    tolerant = onp.zeros(B, dtype=bool) if tolerant0 is None else tolerant0.copy()
    free = onp.zeros(B)
    for t in range(T):
        d = x[t, :, 0]
        exposure = TOL_DECAY * exposure + d
        free = onp.where(d < 0.05, free + 1, 0)
        tolerant = (tolerant | (exposure > TOL_ON)) & (free < HOLIDAY)
        y[t, (d > 0.3) & ~tolerant, 0] = 1.0
    return y


def make_batch(task, B, T, p, seed):
    r = onp.random.default_rng(seed)
    nin = TASKS[task]["nin"]
    mask = r.random((T, B, nin)) < p
    if task == "commit":
        x = onp.zeros((T, B, 1))
        for b in range(B):
            t = 0
            while t < T:
                if r.random() < 0.35 * p:
                    if r.random() < 0.5:
                        amp, dur = r.uniform(1.0, 1.6), int(r.integers(1, 3))
                    else:
                        amp, dur = r.uniform(0.2, 1.0), int(r.integers(6, 17))
                    x[t:t + dur, b, 0] = amp
                    t += dur + 1
                else:
                    t += 1
        return x, commit_truth(x)
    if task == "resistance":
        x = onp.zeros((T, B, 1))
        for b in range(B):
            t = int(r.integers(0, 10))
            while t < T:
                dur = int(r.integers(3, 31))
                x[t:t + dur, b, 0] = r.uniform(0.3, 1.2)
                t += dur + int(r.integers(2, 41))
        return x, resistance_truth(x)
    if task == "antagonist":
        lv = r.uniform(0, 2, (B, 2))
        x = onp.zeros((T, B, 2))
        for t in range(T):
            lv = onp.clip(lv + r.normal(0, 0.08, (B, 2)), 0.0, 2.0)
            x[t] = lv
        occ = x[:, :, 0] / (x[:, :, 0] + 0.5 * (1 + x[:, :, 1] / 1.0) + 1e-9)
        return x, onp.where(occ > 0.5, 1.0, -1.0)[:, :, None]
    if task == "parity":
        u = mask.astype(float)
        y = onp.zeros((T, B, 1))
        state = onp.ones((B, 1))
        for t in range(T):
            state = onp.where(u[t] > 0, -state, state)
            y[t] = state
        return u, y
    mask[0] = True
    base = "flipflop" if task in ("dose", "background") else task
    u = onp.zeros((T, B, nin))
    u[mask] = r.choice([-1.0, 1.0], size=int(mask.sum()))
    latched = onp.zeros((T, B, nin))
    last = onp.zeros((B, nin))
    for t in range(T):
        last = onp.where(u[t] != 0, u[t], last)
        latched[t] = last
    if task == "dose":
        gain = onp.exp(r.uniform(onp.log(0.3), onp.log(3.0), size=(1, B, 1)))
        return u * gain, latched
    if task == "background":
        steps = r.normal(0, 0.04, (T, B, nin))
        bg = onp.zeros((T, B, nin))
        level = r.uniform(-0.5, 0.5, (B, nin))
        for t in range(T):
            level = onp.clip(level + steps[t], -0.5, 0.5)
            bg[t] = level
        return u + bg, latched
    if base == "flipflop":
        return u, latched
    return u, latched[:, :, :1] * latched[:, :, 1:2]


def rollout(P, W, u, K, D, proj=None):
    h = 0.5 * np.ones((u.shape[1], W.shape[0]))
    outs = []
    for t in range(u.shape[0]):
        for _ in range(K):
            h = h + DT * (-h + sig(h @ W.T + D(P, u[t], h)))
            if proj is not None:
                h = proj(h)
        outs.append(h @ P["R"].T + P["c"])
    return np.stack(outs)


def rollout_perturbed(P, W, u, K, D, noise=0.0, division=0.0, inhibit=0.0, seed=11, every=40, proj=None):
    r = onp.random.default_rng(seed)
    h = 0.5 * onp.ones((u.shape[1], W.shape[0]))
    block = onp.ones(W.shape[0])
    if inhibit > 0:
        block[r.permutation(W.shape[0])[: max(1, int(round(inhibit * W.shape[0])))]] = 0.3
    outs = []
    for t in range(u.shape[0]):
        if division > 0 and t > 0 and t % every == 0:
            h = onp.clip(h * onp.exp(division * r.normal(size=h.shape)), 0.0, 1.0)
        for _ in range(K):
            h = h + DT * (-h + block * onp.asarray(sig(h @ W.T + D(P, u[t], h))))
            if proj is not None:
                h = proj(h)
            if noise > 0:
                h = onp.clip(h + noise * onp.sqrt(DT) * r.normal(size=h.shape), 0.0, 1.0)
        outs.append(h @ P["R"].T + P["c"])
    return onp.stack(outs)


def perturbed_accuracy(task, P, W, cfg, D, noise=0.0, division=0.0, inhibit=0.0, B=48, seed=17, proj=None):
    u, y = make_batch(task, B, cfg["test_len"], cfg["pulse_prob"], seed)
    accs = []
    for k in range(3):
        out = rollout_perturbed(P, W, u, cfg["substeps"], D, noise, division, inhibit, seed=seed + 101 * k, proj=proj)
        accs.append(onp.mean(onp.sign(out[10:]) == y[10:]))
    return float(onp.mean(accs))


def landscape(P, W, runs=96, traj_runs=24, steps=600, every=20, seed=3, ids=None, proj=None):
    r = onp.random.default_rng(seed)
    h = r.random((runs, W.shape[0]))
    frames = [h.copy()]
    for s in range(1, 2501):
        h = h + DT * (-h + 1.0 / (1.0 + onp.exp(-(h @ W.T + P["b"]))))
        if proj is not None:
            h = proj(h)
        if s <= steps and s % every == 0:
            frames.append(h.copy())
    ends = h
    X = onp.concatenate(frames + [ends], axis=0)
    mu = X.mean(axis=0)
    _, sv, Vt = onp.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:2]
    var = (sv[:2] ** 2) / max(float((sv ** 2).sum()), 1e-12)
    proj = lambda A: onp.round((A - mu) @ comps.T, 4)
    traj = [proj(onp.stack([f[i] for f in frames])).tolist() for i in range(traj_runs)]
    return {"traj": traj, "ends": proj(ends).tolist(), "ids": ids or [], "var": [round(float(v), 3) for v in var]}


def settle_test(P, W, runs=96, steps=2500, tol=1e-6, seed=3, proj=None):
    r = onp.random.default_rng(seed)
    h = r.random((runs, W.shape[0]))
    speed = onp.zeros(runs)
    first_still = onp.full(runs, -1)
    for s in range(steps):
        hn = h + DT * (-h + 1.0 / (1.0 + onp.exp(-(h @ W.T + P["b"]))))
        if proj is not None:
            hn = proj(hn)
        step_speed = onp.abs(hn - h).max(axis=1)
        moving = step_speed >= tol
        first_still = onp.where(moving, -1, onp.where(first_still < 0, s, first_still))
        if s >= steps - 300:
            speed = onp.maximum(speed, step_speed)
        h = hn
    ids, table = [], {}
    for end, sp in zip(onp.round(h, 2), speed):
        if sp >= tol:
            ids.append(-1)
            continue
        key = tuple(end)
        if key not in table:
            table[key] = len(table)
        ids.append(table[key])
    settled_mask = speed < tol
    settled = float(onp.mean(settled_mask))
    times = first_still[settled_mask & (first_still >= 0)]
    settle_steps = float(onp.median(times)) if times.size else None
    return {"settled": settled, "attractors": len(table), "wells": ids, "settle_steps": settle_steps}


def accuracy(task, P, W, cfg, T, D, B=128, seed=7, proj=None):
    u, y = make_batch(task, B, T, cfg["pulse_prob"], seed)
    out = onp.asarray(rollout(P, W, u, cfg["substeps"], D, proj))
    return float(onp.mean(onp.sign(out[10:]) == y[10:]))


def negative_cycle_edges(W, S):
    off = ~onp.eye(W.shape[0], dtype=bool)
    return int(((S["coop"] * W)[off] < 0).sum())


def run_job(task, arm, cfg, progress=None, rep=0):
    cfg = dict(cfg)
    cfg["seed"] = cfg["seed"] + 7919 * rep
    t0 = time.time()
    tk = TASKS[task]
    Sc, Sb, sv, Sf = sign_matrices(cfg["hidden"], cfg["seed"], cfg.get("feedback_edges", 3))
    proj, pool_groups = pool_projection(cfg["hidden"], cfg.get("pool_size", 4)) if arm == "pool" else (None, [])
    mode = cfg.get("membrane", "direct")
    ncin = tk["nin"] if mode == "direct" else 2 * tk["nin"]
    S = {"coop": Sc, "broken": Sb, "feedback": Sf, "s": sv, "t": channel_signs(ncin, cfg["seed"])}
    P = init_params(arm, cfg, S, tk["nin"], tk["nout"])
    K = cfg["substeps"]
    D = make_drive(arm, S, mode)

    def loss(P, u, y):
        W = recurrent(arm, P["V"], S)
        return np.mean((rollout(P, W, u, K, D, proj) - y) ** 2)

    g = grad(loss)
    m = {k: onp.zeros_like(v) for k, v in P.items()}
    v2 = {k: onp.zeros_like(v) for k, v in P.items()}
    curve = []
    lr = cfg["lr"]
    for i in range(1, cfg["iters"] + 1):
        u, y = make_batch(task, cfg["batch"], cfg["train_len"], cfg["pulse_prob"], cfg["seed"] * 100003 + 1000 + i)
        G = g(P, u, y)
        for k in P:
            gk = onp.nan_to_num(G[k])
            m[k] = 0.9 * m[k] + 0.1 * gk
            v2[k] = 0.999 * v2[k] + 0.001 * gk ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** i)) / (onp.sqrt(v2[k] / (1 - 0.999 ** i)) + 1e-8)
        if i % 10 == 0 or i == 1:
            curve.append([i, float(loss(P, u, y))])
            if progress:
                progress(i / cfg["iters"])

    W = onp.asarray(recurrent(arm, P["V"], S))
    result = {
        "task": task,
        "arm": arm,
        "rep": rep,
        "seed": cfg["seed"],
        "params": int(sum(v.size for v in P.values())),
        "acc_train_len": accuracy(task, P, W, cfg, cfg["train_len"], D, proj=proj),
        "acc_test_len": accuracy(task, P, W, cfg, cfg["test_len"], D, proj=proj),
        "negative_edges": negative_cycle_edges(W, S),
        "curve": curve,
    }
    result.update(settle_test(P, W, proj=proj))
    result["landscape"] = landscape(P, W, ids=result["wells"], proj=proj)
    drift = []
    for eps in cfg["drift"]:
        fs, accs = [], []
        for d in range(4):
            r = onp.random.default_rng(100 + d)
            Wd = W * onp.exp(eps * r.normal(size=W.shape))
            fs.append(settle_test(P, Wd, runs=48, steps=1500, proj=proj)["settled"])
            accs.append(accuracy(task, P, Wd, cfg, cfg["test_len"], D, B=48, proj=proj))
        drift.append({"eps": eps, "settled": float(onp.mean(fs)), "acc": float(onp.mean(accs))})
    result["drift"] = drift
    result["drift_base"] = accuracy(task, P, W, cfg, cfg["test_len"], D, B=48, proj=proj)
    result["stress_base"] = perturbed_accuracy(task, P, W, cfg, D, proj=proj)
    result["noise"] = [{"eps": e, "acc": perturbed_accuracy(task, P, W, cfg, D, noise=e, proj=proj)} for e in cfg.get("noise", [])]
    result["division"] = [{"eps": e, "acc": perturbed_accuracy(task, P, W, cfg, D, division=e, proj=proj)} for e in cfg.get("division", [])]
    result["inhibitor"] = [{"eps": e, "acc": perturbed_accuracy(task, P, W, cfg, D, inhibit=e, proj=proj)} for e in cfg.get("inhibitor", [])]
    result["membrane"] = mode
    result["uptake"] = uptake_curve(P)
    result["model"] = export_spec(arm, P, S, W, mode, cfg, tk)
    result["model"]["pool_groups"] = pool_groups
    result["seconds"] = round(time.time() - t0, 1)
    return result
