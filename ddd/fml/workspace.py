"""Editable model workspace: a chain of layers you can add to, edit, compile, simulate and export.

Workspaces are not trained here. They are for building, inspecting and running models by hand, starting
from a trained circuit, from a metabolic pathway, or from nothing.
"""
import json

import numpy as np

from .model import TASK_IO
from .pathways import PATHWAYS, initial_state, rates, stoichiometry

LAYER_TYPES = {
    "input": {"title": "Signals", "eq": "x_t", "fields": [["channels", "int", 1, 12]]},
    "split": {"title": "Molecule split", "eq": "m = [max(x, 0), max(-x, 0)]", "fields": []},
    "transporter": {"title": "Transporters", "eq": "intake = Vmax m / (Km + m)",
                    "fields": [["vmax", "vector", 0.05, 10], ["km", "vector", 0.01, 10]]},
    "gate": {"title": "Gated channels", "eq": "intake <- intake * sigmoid(G h + g0)", "fields": [["bias", "float", -6, 6]]},
    "linear": {"title": "Linear map", "eq": "d = U x + b", "fields": [["out", "int", 1, 128], ["scale", "float", 0, 5]]},
    "circuit": {"title": "Cell circuit", "eq": "h <- h + dt(-h + sigmoid(W h + d))",
                "fields": [["units", "int", 2, 64], ["dt", "float", 0.05, 1], ["substeps", "int", 1, 10],
                           ["constraint", "choice", ["sign-consistent", "one-cycle-flipped", "contraction", "free"]],
                           ["self_excitation", "float", 0, 12], ["coupling", "float", 0, 3]]},
    "pathway": {"title": "Pathway or model", "eq": "dc/dt = N v(c) + inflow",
                "fields": [["pathway", "choice", list(PATHWAYS)], ["dt", "float", 0.01, 0.2], ["substeps", "int", 1, 20],
                           ["inflow", "float", 0, 5]]},
    "readout": {"title": "Readout", "eq": "y = R h + c", "fields": [["outputs", "int", 1, 8], ["scale", "float", 0, 5]]},
    "decision": {"title": "Decision", "eq": "on if y > 0", "fields": []},
}


def new_layer(kind, **params):
    base = {"id": f"{kind}-{np.random.default_rng().integers(1e6)}", "type": kind, "params": dict(params)}
    if kind == "circuit":
        base["params"].setdefault("units", 12)
        base["params"].setdefault("dt", 0.5)
        base["params"].setdefault("substeps", 3)
        base["params"].setdefault("constraint", "sign-consistent")
        base["params"].setdefault("self_excitation", 5.0)
        base["params"].setdefault("coupling", 0.05)
    if kind == "pathway":
        pid = base["params"].setdefault("pathway", "glycolysis")
        p = PATHWAYS[pid]
        base["params"].setdefault("inputs", [p["species"][0]])
        base["params"].setdefault("outputs", [s for s in ("ATP", "NADH", "Pyr", "NADPH", "Lac", "Cit") if s in p["species"]][:2]
                                  or p["species"][:2])
        base["params"].setdefault("dt", 0.05)
        base["params"].setdefault("substeps", 5)
        base["params"].setdefault("inflow", 1.0)
    if kind == "input":
        base["params"].setdefault("channels", 1)
    if kind == "linear":
        base["params"].setdefault("scale", 1.0)
    if kind == "readout":
        base["params"].setdefault("outputs", 1)
        base["params"].setdefault("scale", 1.0)
    if kind == "gate":
        base["params"].setdefault("bias", 2.0)
    return base


def default_workspace(name="New model"):
    return {"name": name, "layers": [new_layer("input", channels=1), new_layer("linear"), new_layer("circuit"),
                                     new_layer("readout"), new_layer("decision")]}


def from_model(spec, task, name):
    layers = [new_layer("input", channels=spec["nin"])]
    if spec["membrane"] != "direct":
        layers.append(new_layer("split"))
        layers.append(new_layer("transporter", vmax=spec["vmax"], km=spec["km"]))
        if spec["membrane"] == "gated":
            g = new_layer("gate", bias=float(np.mean(spec["g0"])))
            g["weights"] = {"G": spec["G"], "g0": spec["g0"]}
            layers.append(g)
    lin = new_layer("linear", out=spec["hidden"])
    lin["weights"] = {"U": spec["U"], "b": spec["b"]}
    layers.append(lin)
    cons = {"coop": "sign-consistent", "broken": "one-cycle-flipped", "contract": "contraction", "free": "free"}[spec["arm"]]
    cir = new_layer("circuit", units=spec["hidden"], dt=spec["dt"], substeps=spec["substeps"], constraint=cons)
    cir["weights"] = {"W": spec["W"]}
    layers.append(cir)
    ro = new_layer("readout", outputs=spec["nout"])
    ro["weights"] = {"R": spec["R"], "c": spec["c"]}
    layers.append(ro)
    layers.append(new_layer("decision"))
    return {"name": name, "layers": layers, "from_task": task,
            "input_names": TASK_IO[task][0], "output_names": TASK_IO[task][1]}


def from_pathway(pid, name=None):
    if pid not in PATHWAYS:
        raise KeyError(pid)
    p = PATHWAYS[pid]
    pw = new_layer("pathway", pathway=pid)
    layers = [new_layer("input", channels=len(pw["params"]["inputs"])), pw, new_layer("linear", out=8),
              new_layer("circuit", units=8), new_layer("readout", outputs=1), new_layer("decision")]
    return {"name": name or f"{p['name']} model", "layers": layers,
            "input_names": [f"{s} inflow" for s in pw["params"]["inputs"]], "output_names": ["signal"]}


def _rng(seed):
    return np.random.default_rng(seed)


def compile_workspace(ws, seed=0):
    """Resolve shapes, initialise any missing or mismatched weights, and report what changed."""
    r = _rng(seed)
    width, errors, notes = None, [], []
    layers = ws["layers"]
    for i, L in enumerate(layers):
        k, P = L["type"], L["params"]
        W = L.setdefault("weights", {})
        if k == "input":
            width = int(P.get("channels", 1))
        elif width is None:
            errors.append(f"Layer {i + 1} ({k}) comes before any Signals layer.")
            break
        elif k == "split":
            width *= 2
        elif k == "transporter":
            for key, dflt in (("vmax", 1.5), ("km", 0.5)):
                v = np.asarray(P.get(key, W.get(key, [])), dtype=float).ravel()
                if v.size != width:
                    v = np.full(width, dflt)
                    notes.append(f"Layer {i + 1}: {key} resized to {width} channels.")
                P[key] = np.round(v, 4).tolist()
        elif k == "gate":
            nxt = next((x for x in layers[i:] if x["type"] == "circuit"), None)
            if nxt is None:
                errors.append(f"Layer {i + 1} (gated channels) needs a cell circuit after it to gate from.")
                continue
            n = int(nxt["params"].get("units", 12))
            G = np.asarray(W.get("G", []), dtype=float)
            if G.shape != (width, n):
                G = r.normal(0, 0.3, (width, n))
                notes.append(f"Layer {i + 1}: gate weights created for {width} channels and {n} units.")
            W["G"] = np.round(G, 5).tolist()
            W["g0"] = np.round(np.full(width, float(P.get("bias", 2.0))), 5).tolist()
        elif k == "linear":
            nxt = next((x for x in layers[i + 1:] if x["type"] in ("circuit", "readout")), None)
            out = int(nxt["params"].get("units", nxt["params"].get("outputs", 8))) if nxt else int(P.get("out", 8))
            P["out"] = out
            U = np.asarray(W.get("U", []), dtype=float)
            if U.shape != (out, width):
                U = r.normal(0, float(P.get("scale", 1.0)), (out, width))
                notes.append(f"Layer {i + 1}: linear map created at {out} by {width}.")
            W["U"] = np.round(U, 5).tolist()
            b = np.asarray(W.get("b", []), dtype=float)
            W["b"] = np.round(b if b.size == out else np.zeros(out), 5).tolist()
            width = out
        elif k == "circuit":
            n = int(P.get("units", 12))
            Wm = np.asarray(W.get("W", []), dtype=float)
            if Wm.shape != (n, n):
                Wm = init_circuit(n, P, r)
                notes.append(f"Layer {i + 1}: circuit weights created for {n} units under the {P.get('constraint')} rule.")
            Wm = enforce(Wm, P.get("constraint", "free"), seed)
            W["W"] = np.round(Wm, 5).tolist()
            width = n
        elif k == "pathway":
            pid = P.get("pathway", "glycolysis")
            if pid not in PATHWAYS:
                errors.append(f"Layer {i + 1}: unknown pathway {pid}.")
                continue
            p = PATHWAYS[pid]
            P["inputs"] = [s for s in P.get("inputs", []) if s in p["species"]] or [p["species"][0]]
            P["outputs"] = [s for s in P.get("outputs", []) if s in p["species"]] or p["species"][:1]
            if width != len(P["inputs"]):
                errors.append(f"Layer {i + 1}: pathway takes {len(P['inputs'])} inflow(s) but receives {width} value(s). "
                              f"Set the Signals layer to {len(P['inputs'])} channel(s) or change the inflow species.")
            width = len(P["outputs"])
        elif k == "readout":
            out = int(P.get("outputs", 1))
            R = np.asarray(W.get("R", []), dtype=float)
            if R.shape != (out, width):
                R = r.normal(0, float(P.get("scale", 1.0)) * 0.5, (out, width))
                notes.append(f"Layer {i + 1}: readout created at {out} by {width}.")
            W["R"] = np.round(R, 5).tolist()
            c = np.asarray(W.get("c", []), dtype=float)
            W["c"] = np.round(c if c.size == out else np.zeros(out), 5).tolist()
            width = out
        elif k == "decision":
            pass
        else:
            errors.append(f"Layer {i + 1}: unknown layer type {k}.")
    order = [L["type"] for L in layers]
    if "circuit" not in order and "pathway" not in order:
        notes.append("This model has no circuit and no pathway, so it has no internal state.")
    if order and order[0] != "input":
        errors.append("The first layer must be Signals.")
    return {"errors": errors, "notes": notes, "width": width}


def init_circuit(n, P, r):
    mag = np.abs(r.normal(float(P.get("coupling", 0.05)), 0.02, (n, n)))
    np.fill_diagonal(mag, float(P.get("self_excitation", 5.0)))
    s = r.choice([-1.0, 1.0], size=n)
    return np.outer(s, s) * mag


def enforce(W, constraint, seed=0):
    n = W.shape[0]
    s = _rng(seed + 11).choice([-1.0, 1.0], size=n)
    S = np.outer(s, s)
    if constraint == "sign-consistent":
        return S * np.abs(W)
    if constraint == "one-cycle-flipped":
        M = S.copy()
        M[0, 1] *= -1
        return M * np.abs(W)
    if constraint == "contraction":
        f = np.linalg.norm(W) or 1.0
        return 3.6 * W / f
    return W


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_workspace(ws, u, record_internal=True):
    """u: (T, batch, channels). Returns outputs and any internal traces."""
    layers = ws["layers"]
    B = u.shape[1]
    state = {}
    traces = {}
    for L in layers:
        if L["type"] == "circuit":
            state[L["id"]] = np.full((B, int(L["params"]["units"])), 0.5)
        if L["type"] == "pathway":
            p = PATHWAYS[L["params"]["pathway"]]
            c0 = initial_state(p)
            state[L["id"]] = np.array([[c0[s] for s in p["species"]]] * B, dtype=float)
    outs = []
    for t in range(u.shape[0]):
        x = u[t]
        pending_gate = None
        for L in layers:
            k, P, Wt = L["type"], L["params"], L.get("weights", {})
            if k == "input":
                continue
            if k == "split":
                x = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], axis=-1)
            elif k == "transporter":
                vmax, km = np.asarray(P["vmax"]), np.asarray(P["km"]) + 1e-3
                x = vmax * x / (km + x)
            elif k == "gate":
                pending_gate = L
            elif k == "linear":
                U, b = np.asarray(Wt["U"]), np.asarray(Wt["b"])
                nxt = next((y for y in layers[layers.index(L) + 1:] if y["type"] == "circuit"), None)
                if pending_gate is not None and nxt is not None:
                    G, g0 = np.asarray(pending_gate["weights"]["G"]), np.asarray(pending_gate["weights"]["g0"])
                    x = x * _sig(state[nxt["id"]] @ G.T + g0)
                    pending_gate = None
                x = x @ U.T + b
            elif k == "circuit":
                h, Wm = state[L["id"]], np.asarray(Wt["W"])
                for _ in range(int(P["substeps"])):
                    h = h + float(P["dt"]) * (-h + _sig(h @ Wm.T + x))
                state[L["id"]] = h
                x = h
            elif k == "pathway":
                p = PATHWAYS[L["params"]["pathway"]]
                idx = {s: i for i, s in enumerate(p["species"])}
                N = stoichiometry(p)
                c = state[L["id"]]
                clamp = np.array([s in p["clamped"] for s in p["species"]])
                base = np.array([initial_state(p)[s] for s in p["species"]])
                for _ in range(int(P["substeps"])):
                    v = rates(p, c, idx)
                    dc = v @ N.T
                    for j, sp in enumerate(P["inputs"]):
                        dc[:, idx[sp]] += float(P.get("inflow", 1.0)) * np.maximum(x[:, j], 0)
                    c = np.clip(c + float(P["dt"]) * dc, 0, None)
                    c[:, clamp] = base[clamp]
                state[L["id"]] = c
                if record_internal:
                    traces.setdefault(L["id"], []).append(np.round(c[0], 4).tolist())
                x = np.stack([c[:, idx[s]] for s in P["outputs"]], axis=-1)
            elif k == "readout":
                x = x @ np.asarray(Wt["R"]).T + np.asarray(Wt["c"])
            elif k == "decision":
                pass
        outs.append(x)
    return np.stack(outs), traces


def waveform(kind, T, amp=1.0, start=10, width=20, period=60):
    x = np.zeros(T)
    if kind == "constant":
        x[:] = amp
    elif kind == "step":
        x[start:] = amp
    elif kind == "pulse":
        x[start:start + width] = amp
    elif kind == "train":
        for s in range(start, T, period):
            x[s:s + width] = amp
    elif kind == "ramp":
        x[:] = np.linspace(0, amp, T)
    return x


def code_workspace(ws):
    L = ['"""', ws.get("name", "Workspace model"), "",
         "Exported from Fate Memory Lab. Runs on NumPy alone.",
         "Layers: " + " -> ".join(l["type"] for l in ws["layers"]),
         '"""', "import numpy as np", ""]
    if any(l["type"] == "pathway" for l in ws["layers"]):
        L += ["from pathway_data import PATHWAY, stoichiometry, rates, initial_state  # export the pathway separately", ""]
    L += ["LAYERS = " + json.dumps(ws["layers"], indent=1), "", "",
          "def sig(z):", "    return 1.0 / (1.0 + np.exp(-z))", "", "",
          "def run(u):",
          '    """u: (T, batch, channels) -> outputs (T, batch, outputs)"""',
          "    state = {}",
          "    for L in LAYERS:",
          "        if L['type'] == 'circuit':",
          "            state[L['id']] = np.full((u.shape[1], int(L['params']['units'])), 0.5)",
          "    outs = []",
          "    for t in range(u.shape[0]):",
          "        x, gate = u[t], None",
          "        for i, L in enumerate(LAYERS):",
          "            k, P, W = L['type'], L['params'], L.get('weights', {})",
          "            if k == 'split':",
          "                x = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], axis=-1)",
          "            elif k == 'transporter':",
          "                x = np.array(P['vmax']) * x / (np.array(P['km']) + 1e-3 + x)",
          "            elif k == 'gate':",
          "                gate = L",
          "            elif k == 'linear':",
          "                nxt = next((y for y in LAYERS[i + 1:] if y['type'] == 'circuit'), None)",
          "                if gate is not None and nxt is not None:",
          "                    x = x * sig(state[nxt['id']] @ np.array(gate['weights']['G']).T + np.array(gate['weights']['g0']))",
          "                    gate = None",
          "                x = x @ np.array(W['U']).T + np.array(W['b'])",
          "            elif k == 'circuit':",
          "                h = state[L['id']]",
          "                for _ in range(int(P['substeps'])):",
          "                    h = h + float(P['dt']) * (-h + sig(h @ np.array(W['W']).T + x))",
          "                state[L['id']] = h",
          "                x = h",
          "            elif k == 'readout':",
          "                x = x @ np.array(W['R']).T + np.array(W['c'])",
          "        outs.append(x)",
          "    return np.stack(outs)", "", "",
          'if __name__ == "__main__":',
          "    T = 60",
          "    x = np.zeros((T, 1, %d))" % int(ws["layers"][0]["params"].get("channels", 1)),
          "    x[10:30] = 1.0",
          "    print(run(x)[-1, 0])", ""]
    return "\n".join(L)
