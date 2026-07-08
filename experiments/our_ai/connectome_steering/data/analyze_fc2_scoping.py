"""No-within-FC2-WTA scoping measurements (C2/C3), reproducible from FlyWire.

The FC2 goal-selection paper (Paper C) rests on two *structural* connectome facts, previously recorded only
as prose (CLAIMS_LEDGER C2/C3, [TO-ATTACH]). This builder re-derives them from the committed FlyWire data and
writes a committed results cache (`fc2_scoping.json`) + two figures, upgrading them to [REPRO]:

  C2  FB5A inhibition is UNIFORM across bearing. The 4 FB5A cells reach all 85 FC2; the per-cell FB5A->FC2
      weight carries no bearing signal (corr with cos/sin of the FC2 preferred bearing ~0), and the full
      disynaptic FC2->FB5A->FC2 loop is flat across bearing distance. FB5A scales the population; it does not
      sculpt it spatially. (Per-cell weights DO vary in magnitude -- that variation is simply not organized
      by bearing.)

  C3  The FC2 ring has NO local recurrent excitation. Direct FC2->FC2 is negligible; the FC2->hDelta->FC2
      disynaptic path is ANTI-local (rises toward ~180 deg, a vector-summation motif, not neighbour coupling);
      the FC2->vDelta->FC2 path is weak. Global inhibition without local excitation is a normalizer, not a WTA.

These are pure connectivity-geometry measurements (real synapse counts) -- no neurotransmitter sign is used,
so they are independent of the (ML-predicted) FB5A transmitter label. The FC2 preferred bearings `psi_fc2`
are the committed spectral estimate from `steering_wires.json` (synapse positions are absent for a positional
map; declared). Data provenance: FlyWire `proofread_connections_783.feather` + `flywire_neuron_annotations.tsv`
(Zenodo 10676866), the same materialization used by `build_fc2_selection.py`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_scoping.json"
_REC = _D / "fc2_recurrence.json"          # FC2->hDelta->FC2 matrix (for the hDelta-bistability family)
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ann(raw: Path) -> pd.DataFrame:
    return pd.read_csv(raw / "flywire_neuron_annotations.tsv", sep="\t", low_memory=False)


def _ids(ann: pd.DataFrame, *types: str) -> list[int]:
    """root_ids whose cell_type exactly matches one of `types` (case-insensitive)."""
    m = ann["cell_type"].astype(str).str.fullmatch("|".join(types), case=False)
    return ann[m]["root_id"].astype("int64").tolist()


def _ids_prefix(ann: pd.DataFrame, prefix: str) -> list[int]:
    """root_ids whose cell_type starts with `prefix` (the hDelta* / vDelta* families)."""
    m = ann["cell_type"].astype(str).str.match(prefix, case=False, na=False)
    return ann[m]["root_id"].astype("int64").tolist()


def _angdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular distance in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def analyze_fc2_scoping(cfg: dict, write: bool = True) -> dict:
    """Measure C2 (FB5A uniform-across-bearing) + C3 (no local recurrence) from FlyWire."""
    raw = Path(require(cfg, "our_ai.avp_cache_dir"))
    n_bins = int(require(cfg, "our_ai.goal_scope_n_bearing_bins"))
    thr_unif = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    thr_ff = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    thr_anti = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    thr_corr = float(require(cfg, "our_ai.goal_scope_bearing_corr_max"))

    ann = _ann(raw)
    FC2 = _ids(ann, "FC2A", "FC2B", "FC2C")
    FB5A = _ids(ann, "FB5A")
    hD = _ids_prefix(ann, r"hDelta")
    vD = _ids_prefix(ann, r"vDelta")
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    n = len(FC2)
    assert len(psi) == n, f"psi_fc2 ({len(psi)}) != n_fc2 ({n})"

    feather = pd.read_feather(raw / "proofread_connections_783.feather")

    def wmat(pre: list[int], post: list[int]) -> np.ndarray:
        e = feather[feather["pre_pt_root_id"].isin(set(pre)) & feather["post_pt_root_id"].isin(set(post))]
        agg = e.groupby(["pre_pt_root_id", "post_pt_root_id"])["syn_count"].sum()
        pi = {p: i for i, p in enumerate(pre)}
        qi = {q: j for j, q in enumerate(post)}
        M = np.zeros((len(pre), len(post)))
        for (p, q), w in agg.items():
            M[pi[p], qi[q]] = w
        return M

    # bearing-distance bins over FC2 pairs (0..180 deg)
    Dm = _angdist(psi[:, None], psi[None, :])
    bins = np.linspace(0, np.pi, n_bins + 1)
    bctr = np.rad2deg(0.5 * (bins[:-1] + bins[1:]))

    def profile(M: np.ndarray) -> np.ndarray:
        prof = []
        for k in range(n_bins):
            mask = (Dm >= bins[k]) & (Dm < bins[k + 1])
            np.fill_diagonal(mask, False)
            prof.append(float(M[mask].mean()) if mask.any() else float("nan"))
        return np.array(prof)

    # ---- C2: FB5A -> FC2 direct + disynaptic loop ----
    inh = wmat(FB5A, FC2).sum(0)
    corr_cos = float(np.corrcoef(inh, np.cos(psi))[0, 1])
    corr_sin = float(np.corrcoef(inh, np.sin(psi))[0, 1])
    loop = wmat(FC2, FB5A) @ wmat(FB5A, FC2)
    loop_prof = profile(loop)
    loop_mod = float((np.nanmax(loop_prof) - np.nanmin(loop_prof)) / np.nanmean(loop_prof))

    # ---- C3: recurrence ----
    ff = wmat(FC2, FC2)
    ff_mean = float(ff[~np.eye(n, dtype=bool)].mean())

    def disyn(inter: list[int]) -> np.ndarray:
        return profile(wmat(FC2, inter) @ wmat(inter, FC2))

    hD_prof = disyn(hD)
    vD_prof = disyn(vD)
    Dhd = wmat(FC2, hD) @ wmat(hD, FC2)      # the raw FC2->hDelta->FC2 anti-local recurrence matrix (85x85)
    Dvd = wmat(FC2, vD) @ wmat(vD, FC2)      # the raw FC2->vDelta->FC2 LOCAL recurrence matrix (85x85), for the vD bistability family
    hD_ratio = float(hD_prof[-1] / max(hD_prof[0], 1e-9))     # far / near (>1 = anti-local)
    vD_peak_frac = float(np.nanmax(vD_prof) / max(np.nanmax(hD_prof), 1e-9))  # vD weak vs hD

    out = dict(
        source="FlyWire proofread_connections_783: C2 (FB5A uniform-across-bearing) + C3 (no local recurrence)",
        n_fc2=n, n_fb5a=len(FB5A), n_hdelta=len(hD), n_vdelta=len(vD),
        bearing_bin_centers_deg=bctr.tolist(),
        # C2
        fb5a_reaches=int((inh > 0).sum()),
        inh_mean=float(inh.mean()), inh_std=float(inh.std()), inh_cv=float(inh.std() / inh.mean()),
        inh_corr_cos_bearing=corr_cos, inh_corr_sin_bearing=corr_sin,
        loop_profile=loop_prof.tolist(), loop_modulation=loop_mod,
        # C3
        ff_total_syn=float(ff.sum()), ff_offdiag_mean=ff_mean,
        hdelta_profile=hD_prof.tolist(), hdelta_near=float(hD_prof[0]), hdelta_far=float(hD_prof[-1]),
        hdelta_far_near_ratio=hD_ratio,
        vdelta_profile=vD_prof.tolist(), vdelta_peak_frac_of_hdelta=vD_peak_frac,
        # gate thresholds (config, echoed for provenance)
        thresholds=dict(uniform_max_modulation=thr_unif, ff_negligible_max=thr_ff,
                        antilocal_min_ratio=thr_anti, bearing_corr_max=thr_corr),
    )

    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        digest = hashlib.sha256(_OUT.read_bytes()).hexdigest()
        _OUT.with_suffix(".sha256").write_text(f"{digest}  fc2_scoping.json\n")
        # companion cache: the FC2->hDelta->FC2 matrix, for the hDelta-bistability family in prove_fc2_no_wta.py
        _REC.write_text(json.dumps(dict(source="FC2->hDelta->FC2 (anti-local) + FC2->vDelta->FC2 (local) recurrence matrices (85x85)",
                                        n_fc2=n, hdelta_matrix=Dhd.tolist(), vdelta_matrix=Dvd.tolist())))
        _REC.with_suffix(".sha256").write_text(f"{hashlib.sha256(_REC.read_bytes()).hexdigest()}  fc2_recurrence.json\n")
        _render_figures(out)

    # ---- self-verify: the two claims must hold (a STOP if the connectome ever stops supporting them) ----
    print(f"[fc2_scoping] C2: FB5A reaches {out['fb5a_reaches']}/{n} | bearing corr (cos,sin)="
          f"({corr_cos:+.3f},{corr_sin:+.3f}) | disyn-loop modulation={loop_mod:.3f} (<{thr_unif})")
    print(f"[fc2_scoping] C3: FC2->FC2 offdiag mean={ff_mean:.3f} (<{thr_ff}) | "
          f"hDelta near={hD_prof[0]:.1f} far={hD_prof[-1]:.1f} ratio={hD_ratio:.2f} (>{thr_anti}) | "
          f"vDelta peak={np.nanmax(vD_prof):.1f} ({vD_peak_frac*100:.0f}% of hDelta)")
    assert out["fb5a_reaches"] == n, "FB5A does not reach all FC2 (C1)"
    assert abs(corr_cos) < thr_corr and abs(corr_sin) < thr_corr, "FB5A inhibition is bearing-organized (C2 fails)"
    assert loop_mod < thr_unif, f"disynaptic FB5A loop not flat across bearing (C2 fails): {loop_mod}"
    assert ff_mean < thr_ff, f"FC2->FC2 not negligible (C3 fails): {ff_mean}"
    assert hD_ratio > thr_anti, f"FC2->hDelta->FC2 not anti-local (C3 fails): {hD_ratio}"
    return out


def _render_figures(out: dict) -> None:
    """Two paper figures: (1) FB5A uniform vs bearing; (2) FC2 recurrence anti-local."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional; the JSON is the source of truth
        print(f"[fc2_scoping] figures skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    bc = out["bearing_bin_centers_deg"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(bc, out["loop_profile"], "o-", color="#1b6ca8")
    ax.set_ylim(0, max(out["loop_profile"]) * 1.25)
    ax.set_xlabel("bearing distance between FC2 cells (deg)")
    ax.set_ylabel("disynaptic FC2->FB5A->FC2 weight")
    ax.set_title(f"C2: FB5A inhibition is uniform across bearing\n(modulation {out['loop_modulation']*100:.0f}%, flat)")
    fig.tight_layout(); fig.savefig(_FIG / "c2_fb5a_uniform_vs_bearing.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(bc, out["hdelta_profile"], "o-", color="#b8432b", label="FC2->hDelta->FC2 (anti-local)")
    ax.plot(bc, out["vdelta_profile"], "s--", color="#7a7a7a", label="FC2->vDelta->FC2 (weak)")
    ax.axhline(out["ff_offdiag_mean"], color="#2a9d3a", ls=":", label=f"direct FC2->FC2 (negligible, {out['ff_offdiag_mean']:.2f})")
    ax.set_xlabel("bearing distance between FC2 cells (deg)")
    ax.set_ylabel("disynaptic recurrent weight")
    ax.set_title(f"C3: no local recurrent excitation\n(hDelta far/near = {out['hdelta_far_near_ratio']:.1f}, rises toward 180 deg)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(_FIG / "c3_recurrence_antilocal.png", dpi=150); plt.close(fig)
    print(f"[fc2_scoping] figures -> {_FIG}/c2_fb5a_uniform_vs_bearing.png, c3_recurrence_antilocal.png")


def main() -> None:
    analyze_fc2_scoping(_cfg())


if __name__ == "__main__":
    main()
