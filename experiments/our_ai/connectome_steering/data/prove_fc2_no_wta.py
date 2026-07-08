"""No-within-FC2-WTA impossibility demonstration (C4/C5), reproducible from FlyWire.

`analyze_fc2_scoping.py` measured the two STRUCTURAL facts: FB5A inhibition is uniform across bearing (C2)
and the FC2 ring has no local recurrent excitation (C3). This builder shows the DYNAMICAL consequence (C4/C5):
no rate model parameterized by that wiring performs a winner-take-all.

The rigorous test is BISTABILITY, not "one bump". A winner-take-all is bistable — with two equal competing
candidates it breaks symmetry, and WHICH candidate wins depends on the initial state (two attractor basins). A
normalizer has a single input-slaved fixed point — the initial state is irrelevant. (A single bump alone is not
a WTA: the connectome's non-uniform per-cell inh_w gives a FIXED anatomical bias, which is init-invariant.) So
we seed the dynamics toward the LEFT vs the RIGHT candidate and measure whether the seed predicts the winner:

  basin_gap = <settled bump location | seeded-right> - <... | seeded-left>   (degrees)
  ~0  => one fixed point, seed irrelevant => NO winner-take-all
  ~sep => two basins, seed decides       => a winner-take-all

All families run under identical leaky-integration dynamics; only the recurrent operator differs.
  REAL subtractive / divisive / divisive+noise : the connectome wiring (global inhibition, no local excitation),
      the three mechanism families of the scoping pass (C4); the noisy one is the rate proxy for the spiking-LIF
      test (the impossibility is structural — the earlier LIF run, also failed).
  UNIF +local-excitation (POSITIVE control) : an idealized uniform ring WITH local excitation -> bistable ->
      proves the detector CAN see a WTA when one exists (else the negative result would be vacuous).
  UNIF global-only (NEGATIVE control) : the same idealized ring WITHOUT local excitation -> monostable ->
      shows it is the LOCAL EXCITATION (absent per C3), not the ring, that a WTA needs.
(The idealized ring is used for the controls because the real bearings are irregularly sampled, which itself
resists clean bistability; the controls isolate the local-excitation variable on an exactly-symmetric ring.)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_no_wta.json"
_WIRES = _D / "steering_wires.json"
_SEL = _D / "fc2_selection.json"
_REC = _D / "fc2_recurrence.json"          # FC2->hDelta->FC2 matrix (built by analyze_fc2_scoping.py)
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prove_fc2_no_wta(cfg: dict, write: bool = True) -> dict:
    """Show no connectome-parameterized rate model does a WTA (bistability test). C4/C5."""
    g = lambda k: require(cfg, f"our_ai.goal_nowta_{k}")
    sep_deg = float(g("sep_deg")); seed_noise = float(g("seed_noise")); trials = int(g("trials"))
    dt = float(g("dt")); iters = int(g("iters")); xmax = float(g("xmax")); drive_w = float(g("drive_w"))
    sub_g = float(g("sub_g")); div_g = float(g("div_g")); stoch_noise = float(g("stoch_noise"))
    neg_ginh = float(g("negctrl_ginh")); pos_ginh = float(g("posctrl_ginh"))
    pos_exc = float(g("posctrl_exc")); pos_kappa = float(g("posctrl_kappa"))
    asym_grid = [float(a) for a in g("asym_grid")]
    max_gap = float(g("max_gap_deg")); min_wta_gap = float(g("min_wta_gap_deg"))
    seed = int(g("seed")); bump_kappa = float(g("bump_kappa")); hd_gain = float(g("hdelta_gain"))

    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    sel = json.loads(_SEL.read_text())
    inh_w = np.asarray(sel["inh_w"]); pool = np.asarray(sel["pool"])
    power = float(sel["power"]); sigma = float(sel["sigma"])
    n = len(psi)
    psi_u = np.linspace(-np.pi, np.pi, n, endpoint=False)   # idealized uniform ring (controls only)
    rng = np.random.default_rng(seed)

    def vm(p, c, k=bump_kappa):
        b = np.exp(k * np.cos(p - c)); return b / b.max()

    def circ(p, x):
        return float(np.rad2deg(np.angle((x * np.exp(1j * p)).sum()))) if x.sum() > 1e-9 else 0.0

    def wloc(p, kap):
        w = np.exp(kap * np.cos(p[:, None] - p[None, :])); return w / w.sum(1, keepdims=True)

    Wl_unif = wloc(psi_u, pos_kappa)
    Wl_real = wloc(psi, pos_kappa)      # local excitation on the REAL irregular bearings (positive control on real geometry)

    def leaky(p, drive, x0, recur, noise=0.0):
        """x' = x + dt*(-x + clip(drive_w*drive + recur(x) + noise, 0, xmax)). Leaky integration, NO per-step
        renorm (renorm would suppress the WTA runaway and bias the test toward 'no WTA')."""
        x = x0.copy()
        for _ in range(iters):
            nz = noise * rng.standard_normal(n) if noise > 0 else 0.0
            x = x + dt * (-x + np.clip(drive_w * drive + recur(x) + nz, 0.0, xmax))
        return x

    def divisive(p, drive, x0, noise=0.0):
        """Divisive-normalization family (the shipped model's form) under leaky integration."""
        x = x0.copy()
        for _ in range(iters):
            tgt = np.maximum(drive, 0.0) ** power / (sigma + div_g * inh_w * float(pool @ x) + 1e-9)
            nz = noise * rng.standard_normal(n) if noise > 0 else 0.0
            x = np.maximum(x + dt * (-x + tgt) + dt * nz, 0.0)
        return x

    # recurrent operators
    r_sub = lambda x: -sub_g * float(inh_w @ x) * (inh_w / inh_w.mean())     # connectome global inhibition
    r_neg = lambda x: -neg_ginh * float(x.sum()) * np.ones(n)                # uniform global-only
    r_pos = lambda x: -pos_ginh * float(x.sum()) * np.ones(n) + pos_exc * (Wl_unif @ x)  # + local excitation (idealized ring)
    r_pos_real = lambda x: -pos_ginh * float(x.sum()) * np.ones(n) + pos_exc * (Wl_real @ x)  # + local excitation on REAL bearings
    # the anti-local hDelta recurrence is itself a between-column (mutual) inhibition that COULD, in principle,
    # latch a bistable choice; test it directly rather than asserting it is vector-summation. r_hd = connectome
    # global inhibition + the real FC2->hDelta->FC2 matrix (inhibitory, normalized).
    _rec = json.loads(_REC.read_text())
    Dhd = np.asarray(_rec["hdelta_matrix"]); Dhd = Dhd / (Dhd.mean() + 1e-9)
    r_hd = lambda x: r_sub(x) - hd_gain * (Dhd @ x)
    # vDelta is the one LOCAL recurrent route (decays with bearing distance) of undetermined sign. Local
    # EXCITATION is exactly what a ring-attractor WTA needs, so we test the worst case: vDelta as local
    # excitation on top of global inhibition. If even this does not latch, "no local excitation -> no WTA"
    # is a tested result, not a sign assumption.
    Dvd = np.asarray(_rec["vdelta_matrix"]); Dvd = Dvd / (Dvd.mean() + 1e-9)
    r_vd = lambda x: r_sub(x) + hd_gain * (Dvd @ x)

    def basin_gap(p, fam, asym=0.0):
        c = np.deg2rad(sep_deg) / 2
        drive = vm(p, -c) + (1.0 + asym) * vm(p, c)
        seedL = [circ(p, fam(drive, vm(p, -c) + seed_noise * rng.random(n))) for _ in range(trials)]
        seedR = [circ(p, fam(drive, vm(p, +c) + seed_noise * rng.random(n))) for _ in range(trials)]
        return float(np.mean(seedR) - np.mean(seedL))

    families = {
        "real_subtractive":   basin_gap(psi,   lambda d, x0: leaky(psi, d, x0, r_sub)),
        "real_divisive":      basin_gap(psi,   lambda d, x0: divisive(psi, d, x0)),
        "real_divisive_noise": basin_gap(psi,  lambda d, x0: divisive(psi, d, x0, noise=stoch_noise)),
        "real_hdelta_antilocal": basin_gap(psi, lambda d, x0: leaky(psi, d, x0, r_hd)),
        "real_vdelta_local_excit": basin_gap(psi, lambda d, x0: leaky(psi, d, x0, r_vd)),
        "ctrl_neg_global_only": basin_gap(psi_u, lambda d, x0: leaky(psi_u, d, x0, r_neg)),
        "ctrl_pos_local_excit": basin_gap(psi_u, lambda d, x0: leaky(psi_u, d, x0, r_pos)),
        "ctrl_pos_local_excit_realpsi": basin_gap(psi, lambda d, x0: leaky(psi, d, x0, r_pos_real)),
    }
    # C5: even at 20% amplitude asymmetry (and with noise) the divisive family does not become bistable
    div_vs_asym = {f"{a:.2f}": basin_gap(psi, lambda d, x0: divisive(psi, d, x0), asym=a) for a in asym_grid}

    # gain-contingency: the hDelta coupling MATRIX is connectome-pinned, but its scalar strength (hd_gain) is a
    # free parameter with no anatomical anchor. The reference gain is hd_gain; we report the hDelta basin gap across
    # gains and the latch-crossing point so the no-latch conclusion's gain range is explicit, not hidden. hDelta is
    # genuine between-column mutual inhibition, so it CAN latch if strong enough -- unlike the pure global families.
    hd_gain_grid = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    hd_vs_gain = {f"{hg:.1f}": basin_gap(psi, lambda d, x0, hg=hg: leaky(psi, d, x0, lambda x: r_sub(x) - hg * (Dhd @ x)))
                  for hg in hd_gain_grid}
    hd_latch_gain = next((hg for hg in hd_gain_grid if abs(hd_vs_gain[f"{hg:.1f}"]) >= max_gap), None)
    # same gain sweep for vDelta-as-local-excitation: unlike hDelta it never latches at any gain (it is local
    # but too weak/diffuse to form a ring attractor), so "no local excitation" is gain-robust, not reference-only.
    vd_vs_gain = {f"{vg:.1f}": basin_gap(psi, lambda d, x0, vg=vg: leaky(psi, d, x0, lambda x: r_sub(x) + vg * (Dvd @ x)))
                  for vg in hd_gain_grid}
    vd_latch_gain = next((vg for vg in hd_gain_grid if abs(vd_vs_gain[f"{vg:.1f}"]) >= max_gap), None)

    out = dict(
        source="No-within-FC2-WTA bistability demonstration (C4/C5); real FlyWire wiring vs idealized controls",
        n_fc2=n, sep_deg=sep_deg, trials=trials,
        basin_gap_deg=families,
        divisive_basin_gap_vs_asym=div_vs_asym,
        hdelta_reference_gain=hd_gain,
        hdelta_basin_gap_vs_gain=hd_vs_gain,
        hdelta_latch_crossing_gain=hd_latch_gain,
        vdelta_basin_gap_vs_gain=vd_vs_gain,
        vdelta_latch_crossing_gain=vd_latch_gain,
        thresholds=dict(max_gap_deg=max_gap, min_wta_gap_deg=min_wta_gap),
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_no_wta.json\n")
        _render_figure(out)

    # ---- self-verify: the connectome families + negative control do NOT commit; the positive control DOES ----
    print("[fc2_no_wta] basin_gap (deg): " + "  ".join(f"{k}={v:.1f}" for k, v in families.items()))
    print(f"[fc2_no_wta] divisive vs asym: {div_vs_asym}")
    for k in ("real_subtractive", "real_divisive", "real_divisive_noise", "real_hdelta_antilocal", "real_vdelta_local_excit", "ctrl_neg_global_only"):
        assert abs(families[k]) < max_gap, f"{k} shows bistability (unexpected WTA): {families[k]:.1f}"
    assert abs(families["ctrl_pos_local_excit"]) > min_wta_gap, \
        f"positive control is not bistable -> the detector is not validated: {families['ctrl_pos_local_excit']:.1f}"
    assert abs(families["ctrl_pos_local_excit_realpsi"]) > min_wta_gap, \
        f"real-bearings positive control not bistable -> detector misses WTA on real geometry: {families['ctrl_pos_local_excit_realpsi']:.1f}"
    assert vd_latch_gain is None, \
        f"vDelta-as-local-excitation latches at gain {vd_latch_gain} -> 'no local excitation' is not gain-robust"
    for a, gap in div_vs_asym.items():
        assert abs(gap) < max_gap, f"divisive commits at asym={a} (C5 fails): {gap:.1f}"
    return out


def _render_figure(out: dict) -> None:
    """One figure: basin_gap per family — connectome + negative control ~0, positive control tall."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_no_wta] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    g = out["basin_gap_deg"]
    labels = ["real_subtractive", "real_divisive", "real_divisive_noise", "real_hdelta_antilocal", "real_vdelta_local_excit", "ctrl_neg_global_only", "ctrl_pos_local_excit", "ctrl_pos_local_excit_realpsi"]
    vals = [abs(g[k]) for k in labels]
    colors = ["#1b6ca8", "#1b6ca8", "#1b6ca8", "#1b6ca8", "#1b6ca8", "#7a7a7a", "#b8432b", "#b8432b"]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(out["thresholds"]["max_gap_deg"], color="k", ls=":", lw=0.8, label=f"no-latch bound ({out['thresholds']['max_gap_deg']:.0f})")
    ax.axhline(out["thresholds"]["min_wta_gap_deg"], color="#b8432b", ls="--", lw=0.8, label=f"WTA bound ({out['thresholds']['min_wta_gap_deg']:.0f})")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}°" if v >= 0.5 else "≈0°", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=6.5, xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["subtractive\n(real)", "divisive\n(real)", "divisive+noise\n(real)", "hΔ anti-local\n(real)", "vΔ local-excit\n(real)", "global-only\n(ctrl−)", "+local-excit\n(ctrl+ ideal)", "+local-excit\n(ctrl+ real ψ)"], fontsize=6.5)
    ax.set_ylabel("|basin_gap| (deg) — bistability")
    ax.set_title("C4/C5: no ring-attractor winner-take-all in the FC2 connectome\n(bistable only with local excitation, absent per C3)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(_FIG / "c4c5_no_wta_bistability.png", dpi=150); plt.close(fig)
    print(f"[fc2_no_wta] figure -> {_FIG}/c4c5_no_wta_bistability.png")


def main() -> None:
    prove_fc2_no_wta(_cfg())


if __name__ == "__main__":
    main()
