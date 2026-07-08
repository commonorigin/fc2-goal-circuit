"""Prediction robustness + detectability for the FB5A-silencing experiment.

A sensitivity analysis over the model's free gains (power, g_inh, sigma) that establishes WHICH FB5A-silencing
prediction is robust and which is a tuned-operating-point artifact. Key result, and the reason the paper's
prediction is stated the way it is:

  SHAPE / "mush" effect (FB5A off -> more bumps for competing candidates): FRAGILE. It holds only near the
      committed gains (~30% of a 3x3x3 grid) and flips/vanishes elsewhere. This is forced by C2: FB5A's
      inhibition is UNIFORM across bearing, so it cannot reshape the bump; any bump-count change is a threshold
      artifact. We therefore do NOT headline a shape/mush prediction.

  AMPLITUDE / disinhibition effect (FB5A off -> total FC2 activity rises, bump shape preserved): ROBUST. Removing
      a uniform inhibitor uniformly disinhibits the population -> total activity rises in 100% of the grid, and
      the normalized bump shape (concentration, peak bearing) is ~preserved (uniform scaling). This IS the
      normalizer signature and the paper's stated prediction. NOTE: the raw magnitude of the amplitude rise is a
      model artifact (the divisive model is not meant to run un-inhibited), so the prediction is DIRECTIONAL
      (amplitude up + shape preserved), not a magnitude claim.

Distinguishing test: a normalizer predicts disinhibition-without-reshaping; a spatial selector predicts goal
selection degrades/abolished. Cleanly separable by FC2 imaging.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_prediction.json"
_SEL = _D / "fc2_selection.json"
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def analyze_fc2_prediction(cfg: dict, write: bool = True) -> dict:
    """Sweep gains; report shape-effect fragility vs disinhibition robustness."""
    g = lambda k: require(cfg, f"our_ai.goal_pred_{k}")
    power_mult = [float(m) for m in g("power_mult")]
    ginh_mult = [float(m) for m in g("ginh_mult")]
    sigma_mult = [float(m) for m in g("sigma_mult")]
    sep_deg = float(g("sep_deg")); bump_kappa = float(g("bump_kappa"))
    mush_max_robust = float(g("mush_max_robust")); disinhib_min_robust = float(g("disinhib_min_robust"))
    shape_max_dconc = float(g("shape_preserved_max_dconc"))

    sel = json.loads(_SEL.read_text())
    inh_w = np.asarray(sel["inh_w"]); pool = np.asarray(sel["pool"])
    P0 = float(sel["power"]); G0 = float(sel["g_inh"]); S0 = float(sel["sigma"])
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    n = len(psi)

    def vm(c):
        b = np.exp(bump_kappa * np.cos(c - psi)); return b / b.sum() * n

    def dn(drive, p, gi, s, iters=60):  # divisive-normalization runtime form (matches fc2_selection)
        x = np.maximum(drive, 0.0); x = x / (x.max() + 1e-9)
        for _ in range(iters):
            x = np.maximum(drive, 0.0) ** p / (s + gi * inh_w * float(pool @ x) + 1e-9)
            x = x / (x.max() + 1e-9)
        return x

    def raw(drive, p, gi, s, iters=200):  # un-renormalized settled activity (amplitude/disinhibition test)
        x = np.maximum(drive, 0.0).copy()
        for _ in range(iters):
            x = np.maximum(drive, 0.0) ** p / (s + gi * inh_w * float(pool @ x) + 1e-9)
        return x

    def nbumps(x, rel=0.5):
        xa = x[np.argsort(psi)]; a = xa > rel * xa.max(); idx = np.where(a)[0]
        if idx.size == 0: return 0
        r = 1 + int((np.diff(idx) > 1).sum()); return r - 1 if (a[0] and a[-1] and r > 1) else r

    def conc(x):
        return float(abs((x * np.exp(1j * psi)).sum()) / x.sum()) if x.sum() > 1e-12 else 0.0

    c = np.deg2rad(sep_deg) / 2
    drive_comp = vm(-c) + vm(c)
    drive_single = vm(0.6)

    mush_hits = amp_hits = total = 0
    amp_ratios, shape_dconcs = [], []
    for pm in power_mult:
        for gm in ginh_mult:
            for sm in sigma_mult:
                p, gi, s = P0 * pm, G0 * gm, S0 * sm
                total += 1
                # shape/mush effect (competing candidates)
                if nbumps(dn(drive_comp, p, 0.0, s)) > nbumps(dn(drive_comp, p, gi, s)):
                    mush_hits += 1
                # disinhibition/amplitude (single goal, raw settled activity)
                a_on = raw(drive_single, p, gi, s).sum(); a_off = raw(drive_single, p, 0.0, s).sum()
                amp_ratios.append(a_off / a_on)
                if a_off > a_on:
                    amp_hits += 1
                # shape preservation: normalized-shape concentration change under silencing
                shape_dconcs.append(abs(conc(raw(drive_single, p, gi, s)) - conc(raw(drive_single, p, 0.0, s))))

    mush_robust = mush_hits / total
    disinhib_robust = amp_hits / total
    amp_ratios = np.array(amp_ratios); shape_dconcs = np.array(shape_dconcs)

    out = dict(
        source="FB5A-silencing prediction: sensitivity over gains (power/g_inh/sigma)",
        n_gain_combos=total, committed_gains=dict(power=P0, g_inh=G0, sigma=S0),
        mush_shape_robust_frac=mush_robust,            # ~0.30 -> FRAGILE (documented, not headlined)
        disinhibition_robust_frac=disinhib_robust,     # ~1.00 -> ROBUST (the headline prediction)
        amplitude_ratio_off_on=dict(min=float(amp_ratios.min()), median=float(np.median(amp_ratios)),
                                    max=float(amp_ratios.max()),
                                    note="direction robust (all>1); MAGNITUDE is a model artifact, not a claim"),
        shape_preserved_dconc=dict(median=float(np.median(shape_dconcs)), max=float(shape_dconcs.max())),
        thresholds=dict(mush_max_robust=mush_max_robust, disinhib_min_robust=disinhib_min_robust,
                        shape_preserved_max_dconc=shape_max_dconc),
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_prediction.json\n")
        _render_figure(out, amp_ratios)

    print(f"[fc2_prediction] mush/shape effect robust in {mush_robust*100:.0f}% of gains (FRAGILE, <{mush_max_robust*100:.0f}%)")
    print(f"[fc2_prediction] disinhibition robust in {disinhib_robust*100:.0f}% (amp off/on median {np.median(amp_ratios):.0f}x; magnitude=artifact)")
    print(f"[fc2_prediction] shape preserved under silencing: median Δconc {np.median(shape_dconcs):.3f} (<{shape_max_dconc})")
    # self-verify: the shape effect is NOT robust; the disinhibition IS; the shape is preserved under silencing
    assert mush_robust < mush_max_robust, f"mush effect unexpectedly robust ({mush_robust}); re-check the claim"
    assert disinhib_robust >= disinhib_min_robust, f"disinhibition not robust ({disinhib_robust})"
    assert np.median(shape_dconcs) < shape_max_dconc, "shape not preserved under silencing"
    return out


def _render_figure(out: dict, amp_ratios: np.ndarray) -> None:
    """Two panels: robustness bars (mush vs disinhibition); amplitude-ratio distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_prediction] figure skipped ({exc})"); return
    _FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax[0].bar([0, 1], [out["mush_shape_robust_frac"] * 100, out["disinhibition_robust_frac"] * 100],
              color=["#b8432b", "#1b6ca8"])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["shape / mush\n(fragile)", "disinhibition\n(robust)"], fontsize=8)
    ax[0].set_ylabel("% of gain combos effect holds"); ax[0].set_ylim(0, 105)
    ax[0].set_title("Which FB5A-silencing prediction is robust?")
    ax[1].hist(np.log10(amp_ratios), bins=12, color="#1b6ca8")
    ax[1].set_xlabel("log10(FC2 activity off/on)"); ax[1].set_ylabel("gain combos")
    ax[1].set_title("Disinhibition: amplitude always rises\n(direction robust; magnitude = artifact)")
    fig.tight_layout(); fig.savefig(_FIG / "u1_prediction_robustness.png", dpi=150); plt.close(fig)
    print(f"[fc2_prediction] figure -> {_FIG}/u1_prediction_robustness.png")


def main() -> None:
    analyze_fc2_prediction(_cfg())


if __name__ == "__main__":
    main()
