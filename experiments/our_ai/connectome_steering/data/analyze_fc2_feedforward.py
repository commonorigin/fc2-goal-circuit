"""Feedforward max-selection test: does the divisive model's supralinear [.]^p term implement a WTA?

The seeded-basin test (prove_fc2_no_wta.py) rules out a *recurrent/bistable* WTA. This closes the
remaining route a reviewer would raise: a high-exponent divisive normalization can approach a hard
max-selector *without* any recurrence or bistability. We test it directly by sweeping the sharpening
exponent p with two competing goals, at and far beyond the task-agnostic operating point (p=P0).

Three measured facts settle it (none needs the FB5A transmitter — pure model geometry):
  1. AT THE OPERATING POINT (p=P0=2) the circuit CO-REPRESENTS competitors: two equal goals -> both
     bumps kept; the stronger goal correctly leads with the weaker retained. No collapse.
  2. Collapse to a single column appears only at p >> P0 (supra-physiological), and it is NOT a WTA:
     it is INIT-INDEPENDENT (seeding toward goal A vs goal B yields the identical winner at every p),
     i.e. it amplifies a fixed anatomical bias, not a history-dependent choice (the WTA signature).
  3. That high-p collapse does NOT track the goal: at high p the STRONGER input loses (the fixed bias
     beats the input), so it cannot function as a goal-selector at any exponent.

Conclusion: no within-FC2 WTA by the feedforward route either -- confirming the bistability result
from an independent axis. Geometry (sep/kappa/window/input_ratio) reuses the goal_comp_ config; the
exponent grid is a declared analysis sweep.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.config_access import require
from experiments.our_ai.fc2_selection import divisive_normalize

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_feedforward.json"
_SEL = _D / "fc2_selection.json"
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"

# Declared analysis sweep: the sharpening exponent, from linear (1) to supra-physiological (30).
_POWER_SWEEP = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
_COLLAPSE_FRAC = 0.05     # loser mass < 5% of total => "collapsed to one column"
_COREP_FRAC = 0.25        # loser mass > 25% of total => "co-represented"
_INIT_EPS = 1.0           # max |seedA - seedB| winner-% spread to count as init-INDEPENDENT (not bistable)


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def analyze_fc2_feedforward(cfg: dict, write: bool = True) -> dict:
    """Sweep the sharpening exponent: no feedforward WTA (co-representation at P0; init-independent bias at high p)."""
    g = lambda k: require(cfg, f"our_ai.goal_comp_{k}")
    sep = np.deg2rad(float(g("cand_sep_deg"))); rin = float(g("input_ratio"))
    kap = float(g("bump_kappa")); win = np.deg2rad(float(g("window_deg")))

    sel = json.loads(_SEL.read_text())
    inh_w = np.asarray(sel["inh_w"]); pool = np.asarray(sel["pool"])
    P0 = float(sel["power"]); G0 = float(sel["g_inh"]); S0 = float(sel["sigma"])
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"]); n = len(psi)
    cA, cB = -sep / 2, sep / 2

    def vm(c):
        b = np.exp(kap * np.cos(c - psi)); return b / b.sum() * n

    def col(x, c):
        d = np.abs(np.angle(np.exp(1j * (psi - c)))) < win
        return float(x[d].sum())

    def run(drive, power, x0):
        """divisive_normalize seeded from x0 (to probe init-dependence). Same dynamics as the shipped model."""
        x = np.maximum(x0, 0.0); x = x / (x.max() + 1e-9)
        for _ in range(60):
            num = np.maximum(drive, 0.0) ** power
            x = num / (S0 + G0 * inh_w * float(pool @ x) + 1e-9)
            x = x / (x.max() + 1e-9)
        return x

    def winnerA_frac(x):
        a, b = col(x, cA), col(x, cB)
        return a / max(a + b, 1e-9)

    # (1) EQUAL competing goals: sweep p; winner/loser mass + init-independence (seed A vs seed B)
    dEq = vm(cA) + vm(cB)
    sweep = {}
    init_spread_max = 0.0
    for p in _POWER_SWEEP:
        seedA = run(dEq, p, vm(cA)); seedB = run(dEq, p, vm(cB))
        wA, wB = winnerA_frac(seedA), winnerA_frac(seedB)
        # winner mass = whichever column dominates from the neutral (drive) init
        xN = run(dEq, p, np.maximum(dEq, 0.0))
        mA, mB = col(xN, cA), col(xN, cB)
        loser_frac = min(mA, mB) / max(mA + mB, 1e-9)
        init_spread = abs(wA - wB) * 100.0
        init_spread_max = max(init_spread_max, init_spread)
        sweep[f"{p:.0f}"] = dict(loser_mass_frac=float(loser_frac),
                                 init_spread_pct=float(init_spread))

    # (2) STRONGER goal (A stronger): does A win, or does the fixed bias take over at high p?
    dHi = 1.0 * vm(cA) + rin * vm(cB)
    stronger = {}
    for p in _POWER_SWEEP:
        x = run(dHi, p, np.maximum(dHi, 0.0))
        stronger[f"{p:.0f}"] = float(winnerA_frac(x))  # >0.5 => stronger goal A wins

    # Derived claims
    corep_at_P0 = float(sweep[f"{P0:.0f}"]["loser_mass_frac"])
    collapse_ps = [float(k) for k, v in sweep.items() if v["loser_mass_frac"] < _COLLAPSE_FRAC]
    first_collapse_p = min(collapse_ps) if collapse_ps else None
    stronger_at_P0 = stronger[f"{P0:.0f}"]
    stronger_at_max = stronger[f"{_POWER_SWEEP[-1]:.0f}"]

    out = dict(
        source="Feedforward max-selection test: sweep the sharpening exponent p; no WTA via the [.]^p route.",
        power_sweep=_POWER_SWEEP, operating_point_power=P0,
        equal_goals_sweep=sweep, stronger_goal_winnerA_frac=stronger,
        corepresentation_loser_frac_at_P0=corep_at_P0,
        first_collapse_power=first_collapse_p,
        init_spread_max_pct=float(init_spread_max),
        stronger_goal_winnerA_at_P0=stronger_at_P0,
        stronger_goal_winnerA_at_maxp=stronger_at_max,
        thresholds=dict(collapse_frac=_COLLAPSE_FRAC, corep_frac=_COREP_FRAC, init_eps_pct=_INIT_EPS),
        note="At P0 both goals co-represented; collapse only at p>>P0 and is init-INDEPENDENT (bias, not choice) "
             "and does not track the stronger goal -> no feedforward WTA at any exponent.",
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_feedforward.json\n")
        _render_figure(out)

    print(f"[fc2_feedforward] operating point p={P0:.0f}: loser keeps {corep_at_P0*100:.0f}% of mass (co-represented)")
    print(f"[fc2_feedforward] collapse (<{_COLLAPSE_FRAC*100:.0f}% loser) first at p={first_collapse_p} (>> P0={P0:.0f})")
    print(f"[fc2_feedforward] init-independence: max seedA-vs-seedB spread {init_spread_max:.2f}% (=> bias, NOT bistable choice)")
    print(f"[fc2_feedforward] stronger-goal-A wins: {stronger_at_P0*100:.0f}% at P0 -> {stronger_at_max*100:.0f}% at p={_POWER_SWEEP[-1]:.0f} (bias beats input at high p)")

    # gates
    assert corep_at_P0 > _COREP_FRAC, f"operating point does not co-represent competitors ({corep_at_P0})"
    assert init_spread_max < _INIT_EPS, f"collapse is init-DEPENDENT ({init_spread_max}%) -> would be a bistable WTA"
    assert stronger_at_max < 0.55, f"high-p collapse still tracks the stronger goal ({stronger_at_max}) -> could be a selector"
    return out


def _render_figure(out: dict) -> None:
    """Loser mass vs exponent (co-represented at P0, collapse only supra-physiological); init-independent; stronger-goal fails at high p."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_feedforward] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    ps = out["power_sweep"]; P0 = out["operating_point_power"]
    loser = [out["equal_goals_sweep"][f"{p:.0f}"]["loser_mass_frac"] * 100 for p in ps]
    strong = [out["stronger_goal_winnerA_frac"][f"{p:.0f}"] * 100 for p in ps]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    ax[0].plot(ps, loser, "o-", color="#1b6ca8")
    ax[0].axhline(50, color="#2a9d3a", ls=":", lw=0.8, label="equal (no selection)")
    ax[0].axhline(out["thresholds"]["collapse_frac"] * 100, color="#b8432b", ls=":", lw=0.8, label="collapse (<5%)")
    ax[0].axvline(P0, color="k", ls="--", lw=0.8, label=f"operating point (p={P0:.0f})")
    ax[0].set_xlabel("sharpening exponent $p$"); ax[0].set_ylabel("loser bump: % of competing mass")
    ax[0].set_title("Two EQUAL goals: co-represented at $p_0$,\ncollapse only supra-physiological", fontsize=9)
    ax[0].legend(fontsize=6.5)
    ax[1].plot(ps, strong, "o-", color="#b8432b")
    ax[1].axhline(50, color="k", ls=":", lw=0.8, label="tie")
    ax[1].axvline(P0, color="k", ls="--", lw=0.8, label=f"operating point (p={P0:.0f})")
    ax[1].set_xlabel("sharpening exponent $p$"); ax[1].set_ylabel("stronger goal A: % of competing mass")
    ax[1].set_title("Does it pick the STRONGER goal?\nYes at $p_0$; at high $p$ the bias wins instead", fontsize=9)
    ax[1].set_ylim(0, 100); ax[1].legend(fontsize=6.5)
    fig.savefig(_FIG / "cff_feedforward_no_wta.png", dpi=150); plt.close(fig)
    print(f"[fc2_feedforward] figure -> {_FIG}/cff_feedforward_no_wta.png")


def main() -> None:
    analyze_fc2_feedforward(_cfg(), write=True)


if __name__ == "__main__":
    main()
