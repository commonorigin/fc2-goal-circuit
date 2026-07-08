"""FB5A divisive-normalization (soft-k-WTA) weights for FC2 goal selection.

Scoping (tested 2026-07-05) found NO within-FC2 winner-take-all in the connectome: FB5A's inhibition is
UNIFORM at every bearing distance and the FC2 ring has no local recurrent excitation, so no framing collapses
FC2 to a single winner. FB5A's real computation is **divisive normalization / soft k-WTA** (the same
global-inhibition motif as the mushroom body's APL) — it sharpens/salience-weights the held goal; it does not
originate the choice (that is distributed: few candidates upstream + commit-and-hold).

This builder extracts the REAL FB5A wiring into a small committed cache (`fc2_selection.json`): the per-FC2
inhibition weight `inh_w` (FB5A->FC2 synapse counts) and the pool weight `pool` (FC2->FB5A). Runtime applies
divisive normalization with these. Gains are set the fly's way (: from a task-agnostic objective — a clean
single held bump — NOT the driving task). Declared assumption: the FB5A inhibition is DIVISIVE (shunting), a
receptor property the connectome does not resolve.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require
from experiments.our_ai.fc2_selection import divisive_normalize

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_selection.json"
_WIRES = _D / "steering_wires.json"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ids(ann: pd.DataFrame, *types: str) -> list[int]:
    m = ann["cell_type"].astype(str).str.fullmatch("|".join(types), case=False)
    return ann[m]["root_id"].astype("int64").tolist()


def build_fc2_selection(cfg: dict, write: bool = True) -> dict:
    """Extract the real FB5A<->FC2 weights + set the normalization gains on a task-agnostic objective."""
    raw = Path(require(cfg, "our_ai.avp_cache_dir"))
    ann = pd.read_csv(raw / "flywire_neuron_annotations.tsv", sep="\t", low_memory=False)
    FC2, FB5A = _ids(ann, "FC2A", "FC2B", "FC2C"), _ids(ann, "FB5A")
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    n = len(FC2)
    feather = pd.read_feather(raw / "proofread_connections_783.feather")

    def wmat(pre, post):
        e = feather[feather["pre_pt_root_id"].isin(set(pre)) & feather["post_pt_root_id"].isin(set(post))]
        agg = e.groupby(["pre_pt_root_id", "post_pt_root_id"])["syn_count"].sum()
        pi = {p: i for i, p in enumerate(pre)}; qi = {q: j for j, q in enumerate(post)}
        M = np.zeros((len(pre), len(post)))
        for (p, q), w in agg.items():
            M[pi[p], qi[q]] = w
        return M

    inh_w = wmat(FB5A, FC2).sum(0)                       # per-FC2 FB5A inhibition (real synapse counts)
    inh_w = inh_w / (inh_w.mean() + 1e-9)                # scale-free; the global strength is g_inh (a gain)
    pool = wmat(FC2, FB5A).sum(1)                        # each FC2's drive to the FB5A pool (real)
    pool = pool / (pool.sum() + 1e-9)

    # gains set-way: a TASK-AGNOSTIC objective — a single committed goal renders as one clean bump of a
    # biologically-reasonable width (~the goal-ring bump). NOT tuned to the driving task.
    power = float(require(cfg, "our_ai.goal_fc2_norm_power"))
    g_inh = float(require(cfg, "our_ai.goal_fc2_norm_g_inh"))
    sigma = float(require(cfg, "our_ai.goal_fc2_norm_sigma"))

    out = dict(
        source="FlyWire proofread_connections_783: FB5A<->FC2 (divisive-normalization / soft k-WTA)",
        n_fc2=n, n_fb5a=len(FB5A), power=power, g_inh=g_inh, sigma=sigma,
        inh_w=inh_w.astype(np.float32).tolist(), pool=pool.astype(np.float32).tolist(),
    )
    if write:
        _OUT.write_text(json.dumps(out))
        digest = hashlib.sha256(_OUT.read_bytes()).hexdigest()
        _OUT.with_suffix(".sha256").write_text(f"{digest}  fc2_selection.json\n")

    # ---- self-verify: (1) single committed goal -> ONE clean bump; (2) competing candidates -> FB5A normalizes
    #      the field. NOTE: the SHAPE/bump-count effect below is FRAGILE (gain-dependent, ~30%; uniform inhibition
    #      C2 can't reshape). The ROBUST prediction is DISINHIBITION-without-reshaping (see analyze_fc2_prediction.py
    #      + paper §5/§7); the bump-count assert here is a reference-gains sanity check, not the paper's prediction. ----
    def vm(c, k=2.0):
        b = np.exp(k * np.cos(c - psi)); return b / b.sum() * n

    def rd(x):
        return float(np.angle((x * np.exp(1j * psi)).sum()))

    order = np.argsort(psi)
    def nb(x, rel=0.5):
        xa = x[order]; a = xa > rel * xa.max(); idx = np.where(a)[0]
        if idx.size == 0: return 0
        r = 1 + int((np.diff(idx) > 1).sum())
        return r - 1 if (a[0] and a[-1] and r > 1) else r

    single = divisive_normalize(vm(0.6), inh_w, pool, power, g_inh, sigma)
    conc = float(abs((single * np.exp(1j * psi)).sum()) / single.sum())
    # in-model at REFERENCE gains: competing candidates -> FB5A ON gives one dominant bump; g_inh=0 -> the weaker
    # candidate re-emerges (bump count rises). FRAGILE across gains (~30%) -> NOT the paper's headline prediction
    # (that is disinhibition-without-reshaping). Kept as a reference-point sanity check.
    comp_on = divisive_normalize(vm(-1.0) + vm(1.1), inh_w, pool, power, g_inh, sigma)
    comp_off = divisive_normalize(vm(-1.0) + vm(1.1), inh_w, pool, power, 0.0, sigma)   # FB5A silenced
    print(f"[fc2_selection] inh_w reaches {int((inh_w>0).sum())}/{n} | single-goal n_bumps={nb(single)} "
          f"conc={conc:.2f} | competing candidates: FB5A ON n_bumps={nb(comp_on)} vs OFF n_bumps={nb(comp_off)} "
          f"(silence FB5A -> loses single-bump enforcement)")
    assert nb(single) == 1, f"single committed goal is not one clean bump: {nb(single)}"
    assert nb(comp_on) < nb(comp_off), "FB5A does not enforce a single bump (prediction fails in-model)"
    return out


def main() -> None:
    build_fc2_selection(_cfg())


if __name__ == "__main__":
    main()
