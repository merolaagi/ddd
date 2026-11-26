"""Plain-language explanation of a finished run. Deterministic and local: no language model involved."""
import statistics as st

LABEL = {"coop": "sign-consistent", "broken": "one-cycle-flipped", "contract": "contraction", "free": "unconstrained"}
NEEDS_MEMORY = {"flipflop", "xor", "parity", "dose", "background", "commit", "resistance"}
BIO = {
    "flipflop": "holding three independent on/off states, like a cell keeping several genes locked on or off",
    "xor": "combining two stored states, like a cell whose response depends on two remembered signals together",
    "parity": "flipping state on every identical pulse, which order-preserving circuits find hardest",
    "dose": "remembering which signal came last regardless of how strong it was, as cells respond to relative change",
    "background": "telling sharp signals apart from a slowly drifting baseline, as adapting cells do",
    "commit": "committing irreversibly after sustained drug exposure while ignoring brief spikes, like a cell deciding to divide or die",
    "resistance": "becoming drug-tolerant after sustained exposure and regaining sensitivity after a long enough drug holiday, the reversible tolerance seen in drug-tolerant persister cells",
    "antagonist": "reading receptor occupancy when an agonist and a competing antagonist are both present",
}
STRESS = {"drift": "weight drift", "noise": "expression noise", "division": "cell division", "inhibitor": "the inhibitor drug"}


def _m(v):
    return st.mean(v) if v else 0.0


def _sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def _pct(x):
    return f"{x * 100:.1f}%"


def explain(run):
    cfg = run["config"]
    res = run["results"]
    reps = cfg.get("repeats", 1)
    out = {"summary": [], "tasks": [], "next": []}
    if not res:
        out["summary"].append("This run has no finished results yet.")
        return out
    arms_run = cfg["arms"]
    mem = cfg.get("membrane", "direct")
    out["summary"].append(
        f"{len(cfg['tasks'])} task{'s' if len(cfg['tasks']) > 1 else ''}, {len(arms_run)} arm{'s' if len(arms_run) > 1 else ''}, "
        f"{reps} seed{'s' if reps > 1 else ''} each, {cfg['iters']} training steps"
        + ("" if mem == "direct" else f", signals entering through a {'gated' if mem == 'gated' else 'transporter'} membrane") + ".")
    if len(arms_run) == 1:
        out["next"].append(f"Only the {LABEL[arms_run[0]]} arm was trained, so nothing can be compared. Add the sign-consistent arm and its one-cycle-flipped control to test whether structure matters.")
    elif not {"coop", "broken"} <= set(arms_run):
        out["next"].append("The sign-consistent arm and its one-cycle-flipped control are the key pair. Include both to isolate the effect of the sign pattern.")
    if reps < 3:
        out["next"].append("Use at least 3 seeds so differences between arms can be told apart from luck.")

    for task in cfg["tasks"]:
        rs = [r for r in res if r["task"] == task]
        if not rs:
            continue
        by = {}
        for r in rs:
            by.setdefault(r["arm"], []).append(r)
        stats = []
        for arm, group in by.items():
            acc = [g["acc_test_len"] for g in group]
            stats.append({"arm": arm, "acc": _m(acc), "sd": _sd(acc), "train": _m([g["acc_train_len"] for g in group]),
                          "attr": _m([g["attractors"] for g in group]), "settled": _m([g["settled"] for g in group]),
                          "group": group})
        stats.sort(key=lambda s: -s["acc"])
        best = stats[0]
        lines = []
        lines.append(f"This task tests {BIO[task]}.")
        if best["acc"] < 0.6:
            lines.append(f"No arm learned it yet: the best, {LABEL[best['arm']]}, reached {_pct(best['acc'])} on long sequences, close to guessing.")
            out["next"].append(f"{task}: train longer (for example 6000 steps) or raise the learning rate slightly; no arm got past guessing.")
        else:
            lines.append(f"The {LABEL[best['arm']]} arm did best, at {_pct(best['acc'])} on sequences {cfg['test_len']} steps long"
                         + (f" (± {_pct(best['sd'])} across seeds)." if best["sd"] > 0 else "."))
        if len(stats) > 1:
            second = stats[1]
            gap = best["acc"] - second["acc"]
            if gap > max(0.02, best["sd"] + second["sd"]):
                if reps >= 3:
                    lines.append(f"That is clearly ahead of the next arm, {LABEL[second['arm']]}, at {_pct(second['acc'])}.")
                else:
                    lines.append(f"That is ahead of the next arm, {LABEL[second['arm']]}, at {_pct(second['acc'])}, but with {reps} seed{'s' if reps > 1 else ''} the gap could still be luck.")
            else:
                lines.append(f"It is not clearly ahead of {LABEL[second['arm']]} ({_pct(second['acc'])}); the gap is too small to call.")
        sc = {s["arm"]: s for s in stats}
        if "coop" in sc and "broken" in sc:
            d = sc["coop"]["acc"] - sc["broken"]["acc"]
            if abs(d) > max(0.02, sc["coop"]["sd"] + sc["broken"]["sd"]):
                if reps >= 3:
                    lines.append(f"Flipping a single sign {'hurt' if d > 0 else 'helped'} ({_pct(sc['coop']['acc'])} versus {_pct(sc['broken']['acc'])}), so the sign pattern itself matters for this task.")
                else:
                    lines.append(f"Flipping a single sign {'hurt' if d > 0 else 'helped'} ({_pct(sc['coop']['acc'])} versus {_pct(sc['broken']['acc'])}). That hints the sign pattern matters here, but it needs 3 or more seeds to confirm.")
            else:
                lines.append("Flipping a single sign made no reliable difference, so there is no evidence yet that the sign pattern itself matters here.")
        if "contract" in sc:
            if task in NEEDS_MEMORY and sc["contract"]["acc"] < 0.62:
                lines.append("The contraction arm stayed near chance, as expected: with a single resting state it cannot store anything.")
            elif task not in NEEDS_MEMORY and best["arm"] == "contract":
                lines.append("A contraction circuit winning fits a task that needs no memory: one resting state is enough, and it is the most stable design.")
            elif task not in NEEDS_MEMORY and sc["contract"]["acc"] >= best["acc"] - 0.03:
                lines.append("The contraction arm kept up, which fits a task that needs no memory: one resting state is enough.")
        a = best["attr"]
        if task in NEEDS_MEMORY:
            lines.append(f"The best circuit has about {a:.0f} resting state{'s' if round(a) != 1 else ''}; "
                         + ("that is room to store memories." if a > 1.5 else "with only one, anything it remembers must come from slow dynamics, not stable states."))
        else:
            lines.append(f"The best circuit has about {a:.0f} resting state{'s' if round(a) != 1 else ''}, which is fine here because the answer depends only on current drug levels.")
        if best["train"] - best["acc"] > 0.05:
            lines.append(f"Accuracy falls from {_pct(best['train'])} at training length to {_pct(best['acc'])} on longer sequences, so its memory fades over time.")
        if any(s["settled"] < 0.99 for s in stats if s["arm"] == "coop"):
            lines.append("Unexpected: some starting states in the sign-consistent circuit did not settle. Theory says they should; this is worth checking.")

        stress_lines = []
        for key, name in STRESS.items():
            drops = []
            for s in stats:
                g = s["group"]
                levels = g[0].get(key) or []
                if not levels:
                    continue
                base_key = "drift_base" if key == "drift" else "stress_base"
                base = _m([x.get(base_key, x["acc_test_len"]) for x in g])
                worst = _m([x[key][-1]["acc"] for x in g if x.get(key)])
                drops.append((s["arm"], base - worst, levels[-1]["eps"]))
            if not drops:
                continue
            bd = [d for d in drops if d[0] == best["arm"]]
            if bd:
                arm, drop, lvl = bd[0]
                word = "barely hurt it" if drop < 0.03 else "moderately hurt it" if drop < 0.12 else "badly hurt it"
                pts = max(0, round(drop * 100))
                stress_lines.append(f"{name} {word} ({pts} point{'' if pts == 1 else 's'} lost at level {lvl})")
        if stress_lines:
            lines.append(f"Stress results for the {LABEL[best['arm']]} circuit: " + "; ".join(stress_lines) + ".")
        up = best["group"][0].get("uptake")
        if up:
            km = up["km"]
            lines.append(f"Its transporters half-saturate at concentrations around {_m(km):.2f}; well above that, intake levels off, so the model cannot tell very high doses apart.")
        out["tasks"].append({"task": task, "best_arm": best["arm"], "lines": lines})
    if "resistance" in cfg["tasks"] or "commit" in cfg["tasks"]:
        out["next"].append("Build the resistance or commit model and run its therapy search: it puts the model into the unwanted state and tests 90 drug schedules for a way back out.")
    if not out["next"]:
        out["next"].append("Build the model for the best arm below, check its probe chart against the true biology, then export it.")
    return out
