"""Two-cue competition-structure test: distinguishing a UNIFORM/global FB5A from a within-FC2 LOCAL selector.

Present two competing goals at a fixed input strength ratio and measure the RELATIVE activation of the two
columns (ratio A/B) before vs after FB5A silencing. The experiment distinguishes:

  UNIFORM / GLOBAL FB5A (what C2 measured): a spatially-uniform effect changes both columns nearly in proportion,
      so silencing PRESERVES the relative competition structure (the ratio is nearly unchanged).

  WITHIN-FC2 LOCAL SPATIAL SELECTOR: a ring-attractor-like circuit resolves competition by suppressing the
      loser; silencing its inhibition CHANGES which column dominates -> the ratio distorts by orders of magnitude.

HONEST SCOPE (scope-corrected after adversarial review):
  * The discriminator is UNIFORM-vs-LOCAL, not "normalizer-vs-selector". Preservation shows FB5A acts globally
    (consistent with a normalizer, uniform gating, or even a bystander); it does NOT by itself prove a normalizer.
  * It is form- and transmitter-ROBUST for the right reason: ratio preservation requires only that FB5A be
    spatially UNIFORM (C2, a measured fact). We verify this empirically across three functional forms --- uniform
    DIVISIVE, uniform SUBTRACTIVE, and uniform EXCITATORY-additive (the jones2025 predicted-GABAergic-to-
    cholinergic failure mode) --- all of which preserve the ratio to |log-dev| < ~0.05, versus ~20 for a local
    selector (a >100x separation). (The earlier "cancels in the ratio -> transmitter-independent" argument was
    exact only for the divisive form; the empirical form-sweep is the honest justification.)
  * The meaningful robustness axis is spatial NON-UNIFORMITY of the inhibition (the p/g_inh/sigma scalar sweep
    cancels and is uninformative). The ratio survives non-uniformity up to alpha~0.5; the REAL inh_w bearing-
    modulation is ~0.04 (C2), well within margin.
  * Wet-lab confounds (stated in the paper): the fly commits to one goal rather than holding a stable two-column
    ratio; disinhibition-driven saturation compresses ratios; real FB5A silencing leaves the anti-local hDelta
    term intact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.config_access import require
from experiments.our_ai.fc2_selection import divisive_normalize

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_competition.json"
_SEL = _D / "fc2_selection.json"
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def analyze_fc2_competition(cfg: dict, write: bool = True) -> dict:
    """Uniform (divisive/subtractive/excitatory) FB5A preserves the ratio; a local selector distorts it."""
    g = lambda k: require(cfg, f"our_ai.goal_comp_{k}")
    sep = float(g("cand_sep_deg")); rin = float(g("input_ratio")); kap = float(g("bump_kappa"))
    win = np.deg2rad(float(g("window_deg")))
    preserve_max = float(g("preserve_max_logdev")); selector_min = float(g("selector_min_logdev"))
    exc = float(g("selector_exc")); ekap = float(g("selector_kappa")); eginh = float(g("selector_ginh"))
    nonunif_alphas = [float(a) for a in g("nonunif_alphas")]; excit_e = float(g("excit_e")); sub_g = float(g("subtractive_g"))

    sel = json.loads(_SEL.read_text())
    inh_w = np.asarray(sel["inh_w"]); pool = np.asarray(sel["pool"])
    P0 = float(sel["power"]); G0 = float(sel["g_inh"]); S0 = float(sel["sigma"])
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"]); n = len(psi)
    psi_u = np.linspace(-np.pi, np.pi, n, endpoint=False)
    cA, cB = -np.deg2rad(sep) / 2, np.deg2rad(sep) / 2

    def vm(p, c):
        b = np.exp(kap * np.cos(c - p)); return b / b.sum() * n

    def ratio(x, pgrid):
        dA = np.abs(np.angle(np.exp(1j * (pgrid - cA)))) < win
        dB = np.abs(np.angle(np.exp(1j * (pgrid - cB)))) < win
        return float(x[dA].sum() / max(x[dB].sum(), 1e-9))

    def logdev(r_on, r_off):
        return float(abs(np.log(max(r_on, 1e-9) / max(r_off, 1e-9))))

    drive = 1.0 * vm(psi, cA) + rin * vm(psi, cB)

    # (1) three UNIFORM functional forms: does silencing preserve the ratio for each?
    def f_divisive(active):
        return divisive_normalize(drive, inh_w, pool, P0, G0 if active else 0.0, S0)

    def f_subtractive(active, it=80):
        x = np.maximum(drive, 0.0)
        gg = sub_g if active else 0.0
        for _ in range(it):
            x = np.maximum(np.maximum(drive, 0.0) ** P0 - gg * inh_w * float(pool @ x), 0.0)
            m = x.max(); x = x / m if m > 0 else x
        return x

    def f_excit(active):   # EXCITATORY-additive FB5A (jones2025 mode): adds to the drive when active
        return np.maximum(drive + (excit_e if active else 0.0), 0.0) ** P0 / (S0 + 1e-9)

    forms = {
        "divisive":    logdev(ratio(f_divisive(True), psi), ratio(f_divisive(False), psi)),
        "subtractive": logdev(ratio(f_subtractive(True), psi), ratio(f_subtractive(False), psi)),
        "excitatory":  logdev(ratio(f_excit(True), psi), ratio(f_excit(False), psi)),
    }
    forms_max = float(max(forms.values()))

    # (2) the axis that MATTERS: spatial non-uniformity of the inhibition
    base = float(inh_w.mean()); nonunif = {}
    for a in nonunif_alphas:
        iw = base * (1 + a * np.cos(psi))
        nonunif[f"{a:.2f}"] = logdev(ratio(divisive_normalize(drive, iw, pool, P0, G0, S0), psi),
                                     ratio(divisive_normalize(drive, iw, pool, P0, 0.0, S0), psi))
    real_mod = float(np.hypot(np.corrcoef(inh_w, np.cos(psi))[0, 1], np.corrcoef(inh_w, np.sin(psi))[0, 1]))
    break_alpha = next((a for a in nonunif_alphas if nonunif[f"{a:.2f}"] >= preserve_max), nonunif_alphas[-1])

    # (3) selector control (ring + local excitation): silencing distorts the ratio
    Wloc = np.exp(ekap * np.cos(psi_u[:, None] - psi_u[None, :])); Wloc /= Wloc.sum(1, keepdims=True)
    dU = 1.0 * vm(psi_u, cA) + rin * vm(psi_u, cB)

    def ring(drive_u, ginh, dt=0.2, it=400, xmax=5.0):
        x = np.maximum(drive_u, 0.0).copy()
        for _ in range(it):
            x = x + dt * (-x + np.clip(0.4 * drive_u + exc * (Wloc @ x) - ginh * float(x.sum()), 0.0, xmax))
        return x
    sel_logdev = logdev(ratio(ring(dU, eginh), psi_u), ratio(ring(dU, 0.0), psi_u))

    out = dict(
        source="Two-cue competition test: uniform (any form) preserves the ratio, a local selector distorts it",
        input_ratio=rin, cand_sep_deg=sep,
        uniform_form_logdev=forms, uniform_forms_max_logdev=forms_max,
        nonuniformity_logdev=nonunif, nonuniformity_break_alpha=break_alpha,
        real_inh_w_bearing_modulation=real_mod,
        selector_control_logdev=sel_logdev,
        separation_selector_over_uniform=float(sel_logdev / max(forms_max, 1e-9)),
        thresholds=dict(preserve_max_logdev=preserve_max, selector_min_logdev=selector_min),
        note="discriminator is UNIFORM-vs-LOCAL: preservation shows FB5A is global (C2), robust to sign/form; it does NOT alone prove a normalizer",
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_competition.json\n")
        _render_figure(out)

    print(f"[fc2_competition] UNIFORM forms preserve the ratio: " + " ".join(f"{k}={v:.3f}" for k, v in forms.items())
          + f" (all <{preserve_max})")
    print(f"[fc2_competition] NON-UNIFORMITY (real axis): breaks at alpha~{break_alpha}; real inh_w {real_mod:.3f} (safe)")
    print(f"[fc2_competition] SELECTOR control: |log-dev| {sel_logdev:.1f} -> separation {sel_logdev/max(forms_max,1e-9):.0f}x")
    for k, v in forms.items():
        assert v < preserve_max, f"uniform {k} form did not preserve the ratio ({v})"
    assert real_mod < break_alpha, f"real inh_w non-uniformity ({real_mod}) exceeds the break point ({break_alpha})"
    assert sel_logdev > selector_min, "selector control did not distort the ratio"
    return out


def _render_figure(out: dict) -> None:
    """Uniform forms preserve (bars near 0) vs selector distorts; + non-uniformity margin."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_competition] figure skipped ({exc})"); return
    _FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    forms = out["uniform_form_logdev"]
    fvals = list(forms.values())
    fbars = ax[0].bar(range(len(forms)), fvals, color="#1b6ca8", zorder=3)
    ax[0].axhline(out["selector_control_logdev"], color="#b8432b", lw=2,
                  label=f"local selector ({out['selector_control_logdev']:.0f})")
    # symlog with a linear zone around the tiny uniform values so they are actually visible
    ax[0].set_yscale("symlog", linthresh=0.01)
    ax[0].set_ylim(0, 40)
    for b, v in zip(fbars, fvals):
        ax[0].annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                       ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax[0].set_xticks(range(len(forms)))
    ax[0].set_xticklabels([k + "\n(uniform)" for k in forms], fontsize=7)
    ax[0].set_ylabel("|log(ratio$_{on}$/ratio$_{off}$)|  (0 = ratio preserved)")
    ax[0].set_title("Any UNIFORM FB5A preserves the ratio ($\\approx$0);\na LOCAL selector distorts it ($\\approx$21)"); ax[0].legend(fontsize=7)
    al = sorted(float(k) for k in out["nonuniformity_logdev"]); ld = [out["nonuniformity_logdev"][f"{a:.2f}"] for a in al]
    ax[1].plot(al, ld, "o-", color="#1b6ca8")
    ax[1].axhline(out["thresholds"]["preserve_max_logdev"], color="k", ls=":", lw=0.8, label="bound")
    ax[1].axvline(out["real_inh_w_bearing_modulation"], color="#2a9d3a", lw=2, label=f"real inh_w ({out['real_inh_w_bearing_modulation']:.2f})")
    ax[1].set_xlabel("inhibition bearing non-uniformity $\\alpha$"); ax[1].set_ylabel("ratio |log-dev|")
    ax[1].set_title("Robust at the measured non-uniformity"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(_FIG / "u6_competition_structure.png", dpi=150); plt.close(fig)
    print(f"[fc2_competition] figure -> {_FIG}/u6_competition_structure.png")


def main() -> None:
    analyze_fc2_competition(_cfg())


if __name__ == "__main__":
    main()
