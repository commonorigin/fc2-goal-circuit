"""Cross-connectome replication of the no-WTA measurements (C2/C3) in the HEMIBRAIN.

`analyze_fc2_scoping.py` measured C2 (FB5A uniform across bearing) + C3 (no local recurrent excitation) in
the FlyWire connectome. This builder repeats the SAME measurements in the hemibrain (Scheffer 2020, Janelia) — an
independently reconstructed brain, different team — so the negative result cannot be a single-dataset artifact.
It writes a committed cache (`fc2_scoping_hemibrain.json`) + a FlyWire-vs-hemibrain comparison figure.

Data: `traced-neurons.csv` (bodyId → type/instance) + `traced-total-connections.csv` (weighted edges). FC2
bearing is the REAL anatomical column index parsed from the instance (`_C<k>_`, columns 1..N) — a measured
position, not the spectral estimate FlyWire needs.

HONEST hemibrain caveat (investigated, resolved): the hemibrain is a PARTIAL volume, so edge columns are
under-reconstructed. The raw per-cell FB5A→FC2 weight therefore carries a monotonic completeness gradient across
columns (per-column FB5A weight correlates with per-column TOTAL synapses at r≈0.93) that mimics a bearing signal
(raw sin-corr ≈0.36). Dividing each cell's FB5A weight by its total synapses removes it (corr →≈0.14, matching
FlyWire's ~0). So the C2 uniformity check uses the completeness-NORMALIZED weight; the raw value + the 0.93
completeness correlation are reported so the artifact is visible, not hidden. The disynaptic FC2→FB5A→FC2 loop and
all C3 metrics are robust to this (they reproduce raw).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_scoping_hemibrain.json"
_FLYWIRE = _D / "fc2_scoping.json"        # the FlyWire result, for the comparison figure
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _angdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def analyze_fc2_scoping_hemibrain(cfg: dict, write: bool = True) -> dict:
    """Replicate C2/C3 in the hemibrain (bearing = real FB column index)."""
    hb = Path(require(cfg, "our_ai.goal_scope_hemibrain_dir"))
    n_bins = int(require(cfg, "our_ai.goal_scope_n_bearing_bins"))
    thr_corr = float(require(cfg, "our_ai.goal_scope_bearing_corr_max"))
    thr_unif = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    thr_ff = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    thr_anti = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    thr_vd = float(require(cfg, "our_ai.goal_scope_vdelta_max_frac"))
    thr_compl = float(require(cfg, "our_ai.goal_scope_completeness_corr_min"))

    neu = pd.read_csv(hb / "traced-neurons.csv")
    conn = pd.read_csv(hb / "traced-total-connections.csv")   # bodyId_pre, bodyId_post, weight
    t = neu["type"].astype(str); inst = neu["instance"].astype(str)

    fc2_mask = t.str.match("FC2", na=False)
    FC2 = neu[fc2_mask]["bodyId"].tolist()
    col = inst[fc2_mask].str.extract(r"_C(\d+)")[0].astype(int).to_numpy()
    n_col = int(col.max())
    psi = 2 * np.pi * (col - 1) / n_col                        # FB column -> bearing on the circle
    n = len(FC2)
    FB5A = neu[t == "FB5A"]["bodyId"].tolist()
    hD = neu[t.str.match("hDelta", na=False)]["bodyId"].tolist()
    vD = neu[t.str.match("vDelta", na=False)]["bodyId"].tolist()

    allids = set(FC2) | set(FB5A) | set(hD) | set(vD)
    c = conn[conn["bodyId_pre"].isin(allids) & conn["bodyId_post"].isin(allids)]

    def wmat(pre, post):
        e = c[c["bodyId_pre"].isin(set(pre)) & c["bodyId_post"].isin(set(post))]
        agg = e.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum()
        pi = {p: i for i, p in enumerate(pre)}; qi = {q: j for j, q in enumerate(post)}
        M = np.zeros((len(pre), len(post)))
        for (p, q), w in agg.items():
            M[pi[p], qi[q]] = w
        return M

    Dm = _angdist(psi[:, None], psi[None, :])
    bins = np.linspace(0, np.pi, n_bins + 1)
    bctr = np.rad2deg(0.5 * (bins[:-1] + bins[1:]))

    def profile(M):
        out = []
        for k in range(n_bins):
            m = (Dm >= bins[k]) & (Dm < bins[k + 1]); np.fill_diagonal(m, False)
            out.append(float(M[m].mean()) if m.any() else float("nan"))
        return np.array(out)

    # ---- C2: direct (raw + completeness-normalized) + disynaptic loop ----
    inh = wmat(FB5A, FC2).sum(0)
    # completeness proxy: total in+out synapses per FC2 cell (any partner)
    tin = conn[conn["bodyId_post"].isin(set(FC2))].groupby("bodyId_post")["weight"].sum()
    tout = conn[conn["bodyId_pre"].isin(set(FC2))].groupby("bodyId_pre")["weight"].sum()
    tot = np.array([float(tin.get(b, 0.0) + tout.get(b, 0.0)) for b in FC2])
    inh_norm = inh / (tot + 1e-9)
    raw_corr = max(abs(np.corrcoef(inh, np.cos(psi))[0, 1]), abs(np.corrcoef(inh, np.sin(psi))[0, 1]))
    norm_corr = max(abs(np.corrcoef(inh_norm, np.cos(psi))[0, 1]), abs(np.corrcoef(inh_norm, np.sin(psi))[0, 1]))
    # artifact evidence: per-column FB5A weight vs per-column total synapses
    dfc = pd.DataFrame({"col": col, "inh": inh, "tot": tot}).groupby("col").mean()
    compl_corr = float(np.corrcoef(dfc["inh"], dfc["tot"])[0, 1])
    loop_prof = profile(wmat(FC2, FB5A) @ wmat(FB5A, FC2))
    loop_mod = float((np.nanmax(loop_prof) - np.nanmin(loop_prof)) / np.nanmean(loop_prof))

    # ---- C3 ----
    ff = wmat(FC2, FC2)
    ff_mean = float(ff[~np.eye(n, dtype=bool)].mean())
    hD_prof = profile(wmat(FC2, hD) @ wmat(hD, FC2))
    vD_prof = profile(wmat(FC2, vD) @ wmat(vD, FC2))
    # near = smallest populated bin, far = largest populated bin (mid bins can be empty at 9 columns)
    hd_near = float(hD_prof[np.where(~np.isnan(hD_prof))[0][0]])
    hd_far = float(hD_prof[np.where(~np.isnan(hD_prof))[0][-1]])
    hd_ratio = hd_far / max(hd_near, 1e-9)
    vd_peak_frac = float(np.nanmax(vD_prof) / max(np.nanmax(hD_prof), 1e-9))

    out = dict(
        source="Hemibrain v1.2 (Scheffer 2020): cross-connectome replication of C2/C3",
        brain="hemibrain", n_fc2=n, n_fb5a=len(FB5A), n_hdelta=len(hD), n_vdelta=len(vD), n_columns=n_col,
        bearing_bin_centers_deg=bctr.tolist(),
        fb5a_reaches=int((inh > 0).sum()),
        inh_corr_raw=float(raw_corr), inh_corr_completeness_normalized=float(norm_corr),
        completeness_corr=compl_corr,           # per-column FB5A weight vs total synapses (the artifact)
        loop_profile=loop_prof.tolist(), loop_modulation=loop_mod,
        ff_offdiag_mean=ff_mean,
        hdelta_profile=hD_prof.tolist(), hdelta_near=hd_near, hdelta_far=hd_far, hdelta_far_near_ratio=hd_ratio,
        vdelta_profile=vD_prof.tolist(), vdelta_peak_frac_of_hdelta=vd_peak_frac,
        thresholds=dict(bearing_corr_max=thr_corr, uniform_max_modulation=thr_unif, ff_negligible_max=thr_ff,
                        antilocal_min_ratio=thr_anti, vdelta_max_frac=thr_vd, completeness_corr_min=thr_compl),
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_scoping_hemibrain.json\n")
        _render_figure(out)

    print(f"[hemibrain] C2: FB5A {out['fb5a_reaches']}/{n} | direct corr raw={raw_corr:.3f} -> completeness-norm={norm_corr:.3f} "
          f"(<{thr_corr}) | completeness_corr={compl_corr:.2f} | loop modulation={loop_mod:.3f} (<{thr_unif})")
    print(f"[hemibrain] C3: FC2->FC2 mean={ff_mean:.2f} (<{thr_ff}) | hDelta near={hd_near:.0f} far={hd_far:.0f} "
          f"ratio={hd_ratio:.2f} (>{thr_anti}) | vDelta {vd_peak_frac*100:.0f}% of hDelta (<{thr_vd*100:.0f}%)")
    # self-verify: the no-WTA structure replicates (completeness-normalized for the direct check)
    assert out["fb5a_reaches"] == n, "FB5A does not reach all FC2"
    assert norm_corr < thr_corr, f"FB5A inhibition bearing-organized after completeness norm (C2 fails): {norm_corr:.3f}"
    assert compl_corr > thr_compl, f"raw gradient not completeness-explained (unexpected): {compl_corr:.2f}"
    assert loop_mod < thr_unif, f"disynaptic loop not flat (C2 fails): {loop_mod:.3f}"
    assert ff_mean < thr_ff, f"FC2->FC2 not negligible (C3 fails): {ff_mean:.3f}"
    assert hd_ratio > thr_anti, f"hDelta not anti-local (C3 fails): {hd_ratio:.2f}"
    assert vd_peak_frac < thr_vd, f"vDelta not weak (C3 fails): {vd_peak_frac:.2f}"
    return out


def _render_figure(out: dict) -> None:
    """FlyWire vs hemibrain: the two brains agree (C2 loop flat, C3 hDelta anti-local)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[hemibrain] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    fw = json.loads(_FLYWIRE.read_text()) if _FLYWIRE.exists() else None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    # C2 loop flatness
    ax[0].plot(out["bearing_bin_centers_deg"], out["loop_profile"], "o-", color="#b8432b", label=f"hemibrain (mod {out['loop_modulation']*100:.1f}%; 105deg bin empty)")
    if fw:
        ax[0].plot(fw["bearing_bin_centers_deg"], fw["loop_profile"], "s--", color="#1b6ca8", label=f"FlyWire (mod {fw['loop_modulation']*100:.0f}%)")
    ax[0].set_title("C2: FB5A uniform across bearing\n(both brains: flat disynaptic loop)")
    ax[0].set_xlabel("bearing distance (deg)"); ax[0].set_ylabel("FC2->FB5A->FC2 (normalized)")
    # normalize each to its own max for shape comparison
    ax[0].set_ylim(bottom=0); ax[0].legend(fontsize=7)
    # C3 hDelta anti-local ratio bar
    labels, vals, cols = ["hemibrain"], [out["hdelta_far_near_ratio"]], ["#b8432b"]
    if fw:
        labels.append("FlyWire"); vals.append(fw["hdelta_far_near_ratio"]); cols.append("#1b6ca8")
    ax[1].bar(range(len(labels)), vals, color=cols)
    ax[1].axhline(out["thresholds"]["antilocal_min_ratio"], color="k", ls=":", lw=0.8, label="anti-local bound")
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("FC2->hDelta->FC2 far/near ratio")
    ax[1].set_title("C3: no local recurrence\n(both brains: hDelta anti-local)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(_FIG / "w3_flywire_vs_hemibrain.png", dpi=150); plt.close(fig)
    print(f"[hemibrain] figure -> {_FIG}/w3_flywire_vs_hemibrain.png")


def main() -> None:
    analyze_fc2_scoping_hemibrain(_cfg())


if __name__ == "__main__":
    main()
