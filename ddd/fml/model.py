"""Turn a trained arm into a standalone cell model: simulation, probes, layer graph and code export."""
import json

import numpy as np

from .engine import TASKS, commit_truth, make_batch, resistance_truth

TASK_IO = {
    "flipflop": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "xor": (["channel 1", "channel 2"], ["product of memories"]),
    "parity": (["pulse"], ["parity"]),
    "dose": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "background": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "commit": (["drug concentration"], ["committed"]),
    "antagonist": (["agonist", "antagonist"], ["receptor active"]),
    "resistance": (["drug"], ["responding to drug"]),
}
ARM_RULE = {
    "coop": "sign-consistent: W = S ⊙ softplus(V), S_ij = s_i s_j, no negative cycles",
    "broken": "sign-consistent except one flipped off-diagonal sign (one negative cycle)",
    "contract": "contraction: W = 3.6 V / ‖V‖_F, unique equilibrium",
    "free": "unconstrained W",
}


class CellModel:
    def __init__(self, spec):
        self.spec = spec
        self.mode = spec["membrane"]
        self.dt, self.K = spec["dt"], spec["substeps"]
        for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
            if k in spec:
                setattr(self, k, np.asarray(spec[k], dtype=float))

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def drive(self, x, h):
        if self.mode == "direct":
            return x @ self.U.T + self.b
        xs = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=-1)
        intake = self.vmax * xs / (self.km + xs)
        if self.mode == "gated":
            intake = intake * self._sig(h @ self.G.T + self.g0)
        return intake @ self.U.T + self.b

    def run(self, u, h0=None, block=None):
        u = np.asarray(u, dtype=float)
        if u.ndim == 2:
            u = u[:, None, :]
        h = np.full((u.shape[1], self.W.shape[0]), 0.5) if h0 is None else np.asarray(h0, dtype=float)
        outs, states = [], []
        for t in range(u.shape[0]):
            for _ in range(self.K):
                act = self._sig(h @ self.W.T + self.drive(u[t], h))
                if block is not None:
                    act = act * block
                h = h + self.dt * (-h + act)
            outs.append(h @ self.R.T + self.c)
            states.append(h.copy())
        return np.stack(outs), np.stack(states)


THERAPY = {
    "resistance": {
        "bad": "drug-tolerant", "good": "drug-sensitive again",
        "induce": (1.0, 40),
        "goal": "After the schedule, a test dose of the drug makes the cell respond again.",
    },
    "commit": {
        "bad": "committed", "good": "uncommitted",
        "induce": (1.0, 16),
        "goal": "After the schedule, the cell is no longer committed. The true rule says commitment is permanent, so any success here is something the model does that real commitment would not.",
    },
}
AMPS = [0.0, 0.3, 0.6, 1.0, 1.5]
ONS = [2, 5, 10]
OFFS = [5, 15, 30]
INHIB = [0.0, 0.25]
HORIZON = 120
PROBE = 6


def _schedule(a, on, off):
    x = np.zeros(HORIZON)
    if a == 0:
        return x
    t = 0
    while t < HORIZON:
        x[t:t + on] = a
        t += on + off
    return x


def therapy(spec, task):
    if task not in THERAPY:
        return None
    info = THERAPY[task]
    m = CellModel(spec)
    n = m.W.shape[0]
    amp, dur = info["induce"]
    induce = np.zeros((dur + 5, 1, 1))
    induce[:dur, 0, 0] = amp
    out_i, st_i = m.run(induce)
    h_bad = st_i[-1]
    if task == "resistance":
        test = np.full((PROBE, 1, 1), 0.8)
        out_t, _ = m.run(test, h0=h_bad)
        induced = bool(out_t[-4:, 0, 0].mean() < 0)
        truth_induced = resistance_truth(np.concatenate([induce, test]))[-1, 0, 0] < 0
    else:
        induced = bool(out_i[-1, 0, 0] > 0)
        truth_induced = bool(commit_truth(induce)[-1, 0, 0] > 0)
    combos = [(a, on, off, inh) for inh in INHIB for a in AMPS for on in ONS for off in OFFS]
    X = np.stack([_schedule(a, on, off) for a, on, off, _ in combos], axis=1)[:, :, None]
    order = np.random.default_rng(5).permutation(n)
    blocks = np.ones((len(combos), n))
    for i, (_, _, _, inh) in enumerate(combos):
        if inh > 0:
            blocks[i, order[: max(1, int(round(inh * n)))]] = 0.3
    h0 = np.repeat(h_bad, len(combos), axis=0)
    _, st_s = m.run(X, h0=h0, block=blocks)
    h_end = st_s[-1]
    if task == "resistance":
        test = np.full((PROBE, len(combos), 1), 0.8)
        out_p, _ = m.run(test, h0=h_end)
        model_ok = out_p[-4:, :, 0].mean(axis=0) > 0
        full = np.concatenate([np.repeat(induce, len(combos), axis=1), X, test], axis=0)
        truth_ok = resistance_truth(full)[-1, :, 0] > 0
    else:
        rest = np.zeros((10, len(combos), 1))
        out_p, _ = m.run(rest, h0=h_end)
        model_ok = out_p[-1, :, 0] < 0
        truth_ok = np.zeros(len(combos), dtype=bool)
    dose = X[:, :, 0].sum(axis=0)
    rows = []
    for i, (a, on, off, inh) in enumerate(combos):
        rows.append({"amp": a, "on": on, "off": off, "inhibitor": inh, "dose": round(float(dose[i]), 2),
                     "model": bool(model_ok[i]), "truth": bool(truth_ok[i])})
    ok = [r for r in rows if r["model"]]
    best = min(ok, key=lambda r: (r["dose"], r["inhibitor"])) if ok else None
    holiday = [r for r in rows if r["amp"] == 0 and r["inhibitor"] == 0][0]
    agree = float(np.mean(model_ok == truth_ok))
    inh_only = [r for r in rows if r["amp"] == 0 and r["inhibitor"] > 0][0]
    by_inh = [sum(1 for r in rows if r["model"] and r["inhibitor"] == inh) for inh in INHIB]
    false_cures = sum(1 for r in rows if r["model"] and not r["truth"])
    missed = sum(1 for r in rows if r["truth"] and not r["model"])
    return {"by_inhibitor": by_inh, "false_cures": false_cures, "missed": missed,
            "task": task, "bad": info["bad"], "good": info["good"], "goal": info["goal"], "induced": induced,
            "truth_induced": bool(truth_induced), "rows": rows, "n_success": len(ok), "n": len(rows),
            "best": best, "holiday_works": holiday["model"], "holiday_truth": holiday["truth"],
            "inhibitor_alone_works": inh_only["model"], "agreement": agree, "horizon": HORIZON,
            "amps": AMPS, "ons": ONS, "offs": OFFS, "inhibitors": INHIB}


def simulate(spec, task, T=200, seed=0, p=0.05):
    u, y = make_batch(task, 1, T, p, 50000 + seed)
    out, _ = CellModel(spec).run(u)
    acc = float(np.mean(np.sign(out[10:]) == y[10:]))
    return {"inputs": np.round(u[:, 0, :], 4).T.tolist(), "target": y[:, 0, :].T.tolist(),
            "output": np.round(out[:, 0, :], 4).T.tolist(), "accuracy": acc,
            "input_names": TASK_IO[task][0], "output_names": TASK_IO[task][1]}


def probe(spec, task):
    m = CellModel(spec)
    if task == "antagonist":
        ag = np.linspace(0, 2, 21)
        curves = []
        for b in (0.0, 0.5, 1.0, 1.5):
            u = np.zeros((80, len(ag), 2))
            u[:, :, 0] = ag
            u[:, :, 1] = b
            out, _ = m.run(u)
            truth = ag / (ag + 0.5 * (1 + b)) > 0.5
            curves.append({"antagonist": b, "response": np.round(out[-1, :, 0], 4).tolist(),
                           "threshold_true": round(0.5 * (1 + b), 3), "truth": truth.astype(int).tolist()})
        return {"kind": "dose_response", "agonist": np.round(ag, 3).tolist(), "curves": curves}
    if task == "commit":
        amps = np.round(np.linspace(0.1, 1.6, 16), 3)
        durs = np.arange(1, 17)
        grid, truth = [], []
        for d in durs:
            u = np.zeros((d + 60, len(amps), 1))
            u[:d, :, 0] = amps
            out, _ = m.run(u)
            grid.append(np.round(out[-1, :, 0], 3).tolist())
            acc = np.zeros(len(amps))
            hit = np.zeros(len(amps), dtype=bool)
            for t in range(d):
                acc = 0.85 * acc + amps
                hit |= acc > 3.2
            truth.append(hit.astype(int).tolist())
        return {"kind": "commit_map", "amplitude": amps.tolist(), "duration": durs.tolist(), "output": grid, "truth": truth}
    return None


def graph(spec, task, meta):
    ins, outs = TASK_IO[task]
    n, mode = spec["hidden"], spec["membrane"]
    W = np.asarray(spec["W"])
    off = ~np.eye(n, dtype=bool)
    nodes = [{"id": "in", "type": "Input", "title": "Signals", "eq": "x_t ∈ R^%d" % spec["nin"],
              "detail": "Inputs: " + ", ".join(ins), "params": 0}]
    edges = []
    prev = "in"
    if mode != "direct":
        vmax, km = np.asarray(spec["vmax"]), np.asarray(spec["km"])
        nodes.append({"id": "split", "type": "SignSplit", "title": "Molecule split",
                      "eq": "m = [max(x, 0), max(−x, 0)]",
                      "detail": "Each signal becomes a positive and a negative molecule, since concentrations cannot be negative.", "params": 0})
        edges.append([prev, "split"])
        nodes.append({"id": "transport", "type": "MichaelisMenten", "title": "Transporters",
                      "eq": "intake = Vmax ⊙ m / (Km + m)",
                      "detail": "Saturating membrane uptake. Vmax %.2f–%.2f, Km %.2f–%.2f across %d transporters." % (vmax.min(), vmax.max(), km.min(), km.max(), len(km)),
                      "params": int(2 * len(km)), "vectors": {"Vmax": vmax.round(3).tolist(), "Km": km.round(3).tolist()}})
        edges.append(["split", "transport"])
        prev = "transport"
        if mode == "gated":
            nodes.append({"id": "gate", "type": "StateGate", "title": "Gated channels",
                          "eq": "intake ← intake ⊙ σ(G h + g₀)",
                          "detail": "The cell opens and closes its own channels based on its current state.",
                          "params": int(np.asarray(spec["G"]).size + len(spec["g0"])), "matrix": {"G": np.asarray(spec["G"]).round(3).tolist()}})
            edges.append([prev, "gate"])
            prev = "gate"
    U = np.asarray(spec["U"])
    nodes.append({"id": "proj", "type": "Linear", "title": "Input projection", "eq": "d = U·intake + b" if mode != "direct" else "d = U·x + b",
                  "detail": "Maps what gets in onto the %d circuit units." % n, "params": int(U.size + n), "matrix": {"U": U.round(3).tolist()}})
    edges.append([prev, "proj"])
    nodes.append({"id": "cell", "type": "CellCircuit", "title": "Cell circuit",
                  "eq": "h ← h + Δt(−h + σ(W h + d)), ×%d per input step" % spec["substeps"],
                  "detail": "%s. %d units, %d excitatory and %d inhibitory connections. %d resting states found." % (
                      ARM_RULE[spec["arm"]].capitalize(), n, int((W[off] > 0).sum()), int((W[off] < 0).sum()), meta.get("attractors", 0)),
                  "params": int(W.size), "matrix": {"W": W.round(3).tolist()}})
    edges.append(["proj", "cell"])
    edges.append(["cell", "cell", "recurrent"])
    if mode == "gated":
        edges.append(["cell", "gate", "feedback"])
    R = np.asarray(spec["R"])
    nodes.append({"id": "readout", "type": "Linear", "title": "Readout", "eq": "y = R h + c",
                  "detail": "Reads the circuit state out as: " + ", ".join(outs) + ".", "params": int(R.size + len(spec["c"])),
                  "matrix": {"R": R.round(3).tolist()}})
    edges.append(["cell", "readout"])
    nodes.append({"id": "decide", "type": "Sign", "title": "Decision", "eq": "on if y > 0, off otherwise",
                  "detail": "Outputs: " + ", ".join(outs), "params": 0})
    edges.append(["readout", "decide"])
    return {"nodes": nodes, "edges": edges, "total_params": int(sum(nd["params"] for nd in nodes))}


def _arr(name, a):
    return "%s = np.array(%s)" % (name, json.dumps(np.asarray(a).round(6).tolist()))


def code_numpy(spec, task, title):
    ins, outs = TASK_IO[task]
    lines = ['"""', title, "",
             "Standalone cell model exported from Fate Memory Lab. Needs only NumPy.",
             "Inputs per time step: " + ", ".join(ins) + ".",
             "Outputs per time step: " + ", ".join(outs) + " (positive means on).",
             "",
             "Usage:",
             "    from cell_model import CellModel",
             "    m = CellModel()",
             "    outputs, states = m.run(inputs)   # inputs shape (T, %d) or (T, batch, %d)" % (spec["nin"], spec["nin"]),
             '"""', "import numpy as np", "",
             "MEMBRANE = %r" % spec["membrane"], "DT = %r" % spec["dt"], "SUBSTEPS = %d" % spec["substeps"]]
    for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
        if k in spec:
            lines.append(_arr(k, spec[k]))
    lines += ["", "", "def _sig(z):", "    return 1.0 / (1.0 + np.exp(-z))", "", "",
              "class CellModel:",
              "    def drive(self, x, h):",
              '        if MEMBRANE == "direct":',
              "            return x @ U.T + b",
              "        m = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=-1)",
              "        intake = vmax * m / (km + m)",
              '        if MEMBRANE == "gated":',
              "            intake = intake * _sig(h @ G.T + g0)",
              "        return intake @ U.T + b",
              "",
              "    def run(self, inputs, h0=None):",
              "        u = np.asarray(inputs, dtype=float)",
              "        if u.ndim == 2:",
              "            u = u[:, None, :]",
              "        h = np.full((u.shape[1], W.shape[0]), 0.5) if h0 is None else np.asarray(h0, dtype=float)",
              "        outs, states = [], []",
              "        for t in range(u.shape[0]):",
              "            for _ in range(SUBSTEPS):",
              "                h = h + DT * (-h + _sig(h @ W.T + self.drive(u[t], h)))",
              "            outs.append(h @ R.T + c)",
              "            states.append(h.copy())",
              "        return np.stack(outs), np.stack(states)",
              "", "",
              'if __name__ == "__main__":',
              "    x = np.zeros((50, %d))" % spec["nin"],
              "    x[5] = 1.0",
              "    y, _ = CellModel().run(x)",
              '    print("output at last step:", y[-1, 0])', ""]
    return "\n".join(lines)


def code_torch(spec, task, title):
    ins, outs = TASK_IO[task]
    mode = spec["membrane"]
    L = ['"""', title, "",
         "PyTorch version of a cell model exported from Fate Memory Lab.",
         "Weights load as trainable parameters, so the model can be fine-tuned or dropped into a larger network.",
         "Note: fine-tuning W freely does not keep the sign pattern of a sign-consistent circuit.",
         "Inputs: tensor (T, batch, %d): %s." % (spec["nin"], ", ".join(ins)),
         "Outputs: tensor (T, batch, %d): %s (positive means on)." % (spec["nout"], ", ".join(outs)),
         '"""', "import torch", "import torch.nn as nn", "",
         "INIT = {"]
    for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
        if k in spec:
            L.append("    %r: %s," % (k, json.dumps(np.asarray(spec[k]).round(6).tolist())))
    L += ["}", "", "",
          "class MichaelisMentenMembrane(nn.Module):",
          "    def __init__(self, vmax, km):",
          "        super().__init__()",
          "        self.vmax = nn.Parameter(torch.tensor(vmax))",
          "        self.km = nn.Parameter(torch.tensor(km))",
          "",
          "    def forward(self, x):",
          "        m = torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)",
          "        return self.vmax * m / (self.km.clamp_min(1e-3) + m)",
          "", "",
          "class CellCircuit(nn.Module):",
          "    def __init__(self, dt=%r, substeps=%d):" % (spec["dt"], spec["substeps"]),
          "        super().__init__()",
          "        self.dt, self.substeps = dt, substeps",
          "        self.W = nn.Parameter(torch.tensor(INIT['W']))",
          "        self.U = nn.Parameter(torch.tensor(INIT['U']))",
          "        self.b = nn.Parameter(torch.tensor(INIT['b']))",
          "        self.R = nn.Parameter(torch.tensor(INIT['R']))",
          "        self.c = nn.Parameter(torch.tensor(INIT['c']))"]
    if mode != "direct":
        L.append("        self.membrane = MichaelisMentenMembrane(INIT['vmax'], INIT['km'])")
    if mode == "gated":
        L += ["        self.G = nn.Parameter(torch.tensor(INIT['G']))",
              "        self.g0 = nn.Parameter(torch.tensor(INIT['g0']))"]
    L += ["", "    def drive(self, x, h):"]
    if mode == "direct":
        L.append("        return x @ self.U.T + self.b")
    else:
        L.append("        intake = self.membrane(x)")
        if mode == "gated":
            L.append("        intake = intake * torch.sigmoid(h @ self.G.T + self.g0)")
        L.append("        return intake @ self.U.T + self.b")
    L += ["",
          "    def forward(self, u, h=None):",
          "        if h is None:",
          "            h = torch.full((u.shape[1], self.W.shape[0]), 0.5, dtype=u.dtype, device=u.device)",
          "        outs = []",
          "        for t in range(u.shape[0]):",
          "            for _ in range(self.substeps):",
          "                h = h + self.dt * (-h + torch.sigmoid(h @ self.W.T + self.drive(u[t], h)))",
          "            outs.append(h @ self.R.T + self.c)",
          "        return torch.stack(outs), h",
          "", "",
          'if __name__ == "__main__":',
          "    model = CellCircuit()",
          "    x = torch.zeros(50, 1, %d)" % spec["nin"],
          "    x[5] = 1.0",
          "    y, h = model(x)",
          '    print("output at last step:", y[-1, 0].tolist())', ""]
    return "\n".join(L)


POSITIVE_ONLY = {"commit", "resistance", "antagonist"}


def find_attractors(m, n_starts=96, steps=600, seed=3, merge=0.05, cap=10):
    n = m.W.shape[0]
    h0 = np.random.default_rng(seed).random((n_starts, n))
    u = np.zeros((steps, n_starts, m.U.shape[1] if m.mode == "direct" else m.U.shape[1] // 2))
    _, st = m.run(u, h0=h0)
    end, prev = st[-1], st[-2]
    still = np.abs(end - prev).max(axis=1) < 1e-5
    centers, counts = [], []
    for e in end[still]:
        for i, c in enumerate(centers):
            if np.abs(e - c).max() < merge:
                counts[i] += 1
                break
        else:
            centers.append(e)
            counts.append(1)
    order = np.argsort(counts)[::-1][:cap]
    return [centers[i] for i in order], [counts[i] / n_starts for i in order], float(still.mean())


def _nearest(states, centers, tol=0.05):
    C = np.stack(centers)
    d = np.abs(states[:, None, :] - C[None, :, :]).max(axis=2)
    j = d.argmin(axis=1)
    return np.where(d[np.arange(len(states)), j] < tol, j, -1)


def reachability(spec, task):
    m = CellModel(spec)
    nin = spec["nin"]
    centers, basins, settled = find_attractors(m)
    if not centers:
        return {"attractors": [], "note": "No starting state settled, so there are no attractors to map."}
    basins = list(basins)
    found_by_kick = [False] * len(centers)
    signs = [1.0] if task in POSITIVE_ONLY else [1.0, -1.0]
    kicks = [(k, sg, a, d) for k in range(nin) for sg in signs for a in (0.5, 1.5) for d in (3, 15, 40)]
    K = len(kicks)
    L, REST = max(k[3] for k in kicks), 400
    for _round in range(3):
        A = len(centers)
        u = np.zeros((L + REST, A * K, nin))
        h0 = np.zeros((A * K, m.W.shape[0]))
        for i in range(A):
            for j, (k, sg, a, d) in enumerate(kicks):
                u[:d, i * K + j, k] = sg * a
                h0[i * K + j] = centers[i]
        _, st = m.run(u, h0=h0)
        final = st[-1]
        still = np.abs(st[-1] - st[-2]).max(axis=1) < 1e-5
        land = _nearest(final, centers)
        added = False
        for f, ok, l in zip(final, still, land):
            if l < 0 and ok and len(centers) < 16 and not any(np.abs(f - c).max() < 0.05 for c in centers):
                centers.append(f)
                basins.append(0.0)
                found_by_kick.append(True)
                added = True
        if not added:
            break
    A = len(centers)
    edges = {}
    for i in range(A):
        for j, (k, sg, a, d) in enumerate(kicks):
            t = int(land[i * K + j])
            if t == i:
                continue
            cost = a * d
            key = (i, t)
            if key not in edges or cost < edges[key]["cost"]:
                edges[key] = {"from": i, "to": t, "channel": k, "sign": sg, "amp": a, "steps": d, "cost": cost}
    left = [any(int(land[i * K + j]) != i for j in range(K)) for i in range(A)]
    reach = []
    for i in range(A):
        seen, stack = {i}, [i]
        while stack:
            x = stack.pop()
            for (f, t) in edges:
                if f == x and t >= 0 and t not in seen:
                    seen.add(t)
                    stack.append(t)
        reach.append(sorted(seen - {i}))
    s = np.asarray(spec["orthant"])
    theory = spec["arm"] in ("coop", "broken") and spec["membrane"] == "gated"
    order = [[bool(np.all(s * (centers[b] - centers[a]) >= -1e-3)) and a != b for b in range(A)] for a in range(A)]
    check = None
    if theory:
        t = np.asarray(spec["channel_signs"])
        viol, tested = 0, 0
        dirs = set()
        for i in range(A):
            for j, (k, sg, a, d) in enumerate(kicks):
                c = k if sg > 0 else k + nin
                dirn = t[c]
                dirs.add(float(dirn))
                diff = s * (final[i * K + j] - centers[i]) * dirn
                tested += 1
                viol += int(np.any(diff < -1e-3))
        provable = []
        if len(dirs) == 1:
            dn = dirs.pop()
            for i in range(A):
                beyond = any(order[i][b] if dn > 0 else order[b][i] for b in range(A))
                if not beyond:
                    provable.append(i)
            direction = "up" if dn > 0 else "down"
        else:
            direction = "both"
        check = {"tested": tested, "violations": viol, "direction": direction, "provable_traps": provable,
                 "holds": spec["arm"] == "coop"}
    rng = np.random.default_rng(9)
    sigmas = [0.02, 0.05, 0.1, 0.2, 0.3]
    R = 16
    barriers = []
    for i in range(A):
        esc = []
        for sg in sigmas:
            h = np.repeat(centers[i][None, :], R, axis=0)
            for _ in range(150 * m.K):
                h = h + m.dt * (-h + m._sig(h @ m.W.T + m.b))
                h = np.clip(h + sg * np.sqrt(m.dt) * rng.normal(size=h.shape), 0, 1)
            for _ in range(300 * m.K):
                h = h + m.dt * (-h + m._sig(h @ m.W.T + m.b))
            esc.append(float(np.mean(_nearest(h, centers) != i)))
        b = next((sg for sg, e in zip(sigmas, esc) if e >= 0.5), None)
        barriers.append({"escape": esc, "barrier": b})
    out = np.asarray(m.R @ np.stack(centers).T + m.c[:, None]).T
    atts = []
    for i in range(A):
        atts.append({"id": i, "basin": round(basins[i], 3), "height": round(float(np.mean(s * centers[i])), 4), "readout": np.round(out[i], 3).tolist(),
                     "reaches": reach[i], "reached_from": [j for j in range(A) if i in reach[j]],
                     "trap": len(reach[i]) == 0 and not left[i] and A > 1, "found_by_kick": found_by_kick[i], "barrier": barriers[i]["barrier"],
                     "escape": barriers[i]["escape"]})
    return {"attractors": atts, "edges": list(edges.values()), "kicks": K, "settled": settled, "sigmas": sigmas,
            "order": order, "theory": check, "positive_only": task in POSITIVE_ONLY,
            "output_names": TASK_IO[task][1]}
