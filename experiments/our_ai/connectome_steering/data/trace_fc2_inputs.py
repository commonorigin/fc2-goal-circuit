"""Trace FC2's upstream inputs — the distributed goal-candidate drive.

Paper C argues goal selection is DISTRIBUTED (candidates arrive upstream, FB5A normalizes). Earlier steps established
the negative (no within-FC2 WTA) in two brains. This builder measures the POSITIVE side: what actually feeds
FC2, from the FlyWire connectome, and separates the inputs by their single-cell targeting geometry.

Ranked input map (all edges INTO the 85 FC2), then a PER-CELL analysis: for each presynaptic cell, the bearing
concentration of its projection onto FC2 (over the committed `psi_fc2`) and how many FC2 it contacts. This
cleanly splits two roles that a type-summed view hides:

  NARROW (per-cell concentration high, few FC2 targets)  -> a DIRECTIONAL carrier (can inject a goal bearing).
      PFN (columnar heading / self-motion drive, ~10 FC2 each, conc ~0.96) — the feedforward directional input.
      hDelta/vDelta are also narrow, but they are RECURRENCE (anti-local vector-summation, C3), not upstream.
  BROAD (per-cell concentration low, many FC2 targets)   -> a GLOBAL gate/normalizer (no direction).
      The whole FB-tangential family, incl. FB5A (84/85, conc ~0.20 — the C2 normalizer at single-cell level).

Findings the map refines vs the initial expectation: FB-tangential dominates FC2 input (~55%); MBON input is ~0 DIRECT (MB
valence reaches FC2 indirectly, via FB-tangential gating), correcting the "MB valence upstream (direct)" reading.

Honest limits: the streams are MEASURED; which carries the *committed* goal is a reasoned nomination, not proof;
NT sign is the ML-predicted label (as everywhere); the picture is distributed (no single localized goal-setter).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_inputs.json"
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _category(t: str) -> str:
    if t.startswith(("hDelta", "vDelta")) or t.startswith("FC"):
        return "recurrence"
    if t.startswith("PFN"):
        return "PFN_directional"
    if t.startswith("FB"):
        return "FB_tangential_broad"
    if t.startswith("MBON"):
        return "MBON"
    if t.startswith("ER"):
        return "ER_ring"
    return "other"


def trace_fc2_inputs(cfg: dict, write: bool = True) -> dict:
    """Rank FC2's inputs + per-cell narrow(directional)/broad(normalizer) split."""
    raw = Path(require(cfg, "our_ai.avp_cache_dir"))
    top_n = int(require(cfg, "our_ai.goal_upstream_top_n"))
    min_cell_syn = int(require(cfg, "our_ai.goal_upstream_min_cell_syn"))
    narrow_thr = float(require(cfg, "our_ai.goal_upstream_narrow_conc"))
    broad_thr = float(require(cfg, "our_ai.goal_upstream_broad_conc"))

    ann = pd.read_csv(raw / "flywire_neuron_annotations.tsv", sep="\t", low_memory=False)
    feather = pd.read_feather(raw / "proofread_connections_783.feather")
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    type_of = dict(zip(ann["root_id"].astype("int64"), ann["cell_type"].astype(str)))

    fc2_mask = ann["cell_type"].astype(str).str.fullmatch("FC2A|FC2B|FC2C", case=False)
    FC2 = list(ann[fc2_mask]["root_id"].astype("int64"))
    fc2_idx = {b: i for i, b in enumerate(FC2)}
    n = len(FC2)
    assert len(psi) == n

    inc = feather[feather["post_pt_root_id"].isin(set(FC2))].copy()
    inc["pre_type"] = inc["pre_pt_root_id"].map(type_of).fillna("<untyped>")
    total = float(inc["syn_count"].sum())
    by_type = inc.groupby("pre_type")["syn_count"].sum().sort_values(ascending=False)
    typed_frac = float(1.0 - by_type.get("<untyped>", 0.0) / total)

    # category composition
    cats: dict[str, float] = {}
    for ty, s in by_type.items():
        cats[_category(ty)] = cats.get(_category(ty), 0.0) + float(s)
    cats = {k: v / total for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}

    # MBON naming sanity: MBONs DO exist in the annotation -> "~0 into FC2" is meaningful, not a name miss
    n_mbon_total = int(ann["cell_type"].astype(str).str.match("MBON", na=False).sum())

    # per-cell bearing concentration for the top-N presynaptic types
    def percell(type_name: str) -> dict:
        e = inc[inc["pre_type"] == type_name]
        concs, ntgt = [], []
        for _, g in e.groupby("pre_pt_root_id"):
            w = np.zeros(n)
            for b, s in g.groupby("post_pt_root_id")["syn_count"].sum().items():
                if b in fc2_idx:
                    w[fc2_idx[b]] = s
            if w.sum() < min_cell_syn:
                continue
            concs.append(float(abs((w * np.exp(1j * psi)).sum()) / w.sum()))
            ntgt.append(int((w > 0).sum()))
        if not concs:
            return {}
        return dict(n_cells=len(concs), conc_mean=float(np.mean(concs)),
                    conc_med=float(np.median(concs)), ntgt_med=float(np.median(ntgt)))

    per_type = {}
    for ty in list(by_type.head(top_n).index):
        if ty == "<untyped>":
            continue
        pc = percell(ty)
        if pc:
            pc.update(syn=float(by_type[ty]), pct=float(100 * by_type[ty] / total), category=_category(ty))
            per_type[ty] = pc

    # classify NARROW (directional) vs BROAD (normalizer) at the single-cell level
    narrow = {t: v for t, v in per_type.items() if v["conc_med"] >= narrow_thr}
    broad = {t: v for t, v in per_type.items() if v["conc_med"] <= broad_thr}
    fb5a = per_type.get("FB5A", {})
    pfn = {t: v for t, v in per_type.items() if v["category"] == "PFN_directional"}
    pfn_conc = float(np.mean([v["conc_med"] for v in pfn.values()])) if pfn else float("nan")

    out = dict(
        source="FlyWire proofread_connections_783: FC2 upstream input map + per-cell targeting",
        n_fc2=n, total_input_syn=total, typed_coverage=typed_frac, n_presynaptic_types=int(by_type.size),
        category_fractions=cats, n_mbon_in_brain=n_mbon_total, mbon_direct_frac=cats.get("MBON", 0.0),
        top_inputs={t: dict(syn=float(by_type[t]), pct=float(100 * by_type[t] / total), category=_category(t))
                    for t in list(by_type.head(15).index) if t != "<untyped>"},
        per_type_targeting=per_type,
        fb5a_percell=fb5a, pfn_percell_conc_mean=pfn_conc,
        narrow_directional_types=sorted(narrow), broad_normalizer_types=sorted(broad),
        thresholds=dict(narrow_conc=narrow_thr, broad_conc=broad_thr, min_cell_syn=min_cell_syn),
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_inputs.json\n")
        _render_figure(out)

    print(f"[fc2_inputs] typed {typed_frac*100:.0f}% | categories: " +
          " ".join(f"{k}={v*100:.0f}%" for k, v in cats.items()))
    print(f"[fc2_inputs] MBON in brain={n_mbon_total} but direct->FC2={cats.get('MBON',0)*100:.2f}% (indirect)")
    print(f"[fc2_inputs] PFN per-cell conc={pfn_conc:.2f} (NARROW/directional, {len(pfn)} types) vs "
          f"FB5A per-cell conc={fb5a.get('conc_med',float('nan')):.2f} contacts {fb5a.get('ntgt_med',0):.0f}/{n} (BROAD/normalizer)")
    # self-verify: the input map + the directional-carrier vs normalizer split
    assert typed_frac > 0.95, f"typed coverage low: {typed_frac}"
    assert cats.get("FB_tangential_broad", 0) > 0.4, "FB-tangential not the dominant input"
    assert n_mbon_total > 0 and cats.get("MBON", 0.0) < 0.01, "MBON-direct check invalid or non-negligible"
    assert pfn_conc >= narrow_thr, f"PFN not narrow/directional: {pfn_conc}"
    assert fb5a and fb5a["conc_med"] <= broad_thr, f"FB5A not broad/normalizer: {fb5a.get('conc_med')}"
    assert fb5a["ntgt_med"] >= 0.9 * n, f"FB5A does not blanket FC2: {fb5a['ntgt_med']}/{n}"
    return out


def _render_figure(out: dict) -> None:
    """Two panels: input composition; per-cell concentration (PFN narrow/directional vs FB-tangential broad)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_inputs] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cats = out["category_fractions"]
    _disp = {"recurrence": "hΔ/vΔ (recurrence)", "PFN_directional": "PFN (directional)",
             "FB_tangential_broad": "FB-tangential (broad)"}
    _clabels = [_disp.get(k, k) for k in list(cats)[::-1]]
    ax[0].barh(_clabels, [v * 100 for v in list(cats.values())[::-1]], color="#1b6ca8")
    ax[0].set_xlabel("% of FC2 input synapses"); ax[0].set_title("FC2 input composition\n(MBON ~0 direct)")
    # per-cell concentration by category
    pt = out["per_type_targeting"]
    colc = {"PFN_directional": "#b8432b", "recurrence": "#e0a52b", "FB_tangential_broad": "#1b6ca8"}
    for cat_name, color in colc.items():
        xs = [v["ntgt_med"] for v in pt.values() if v["category"] == cat_name]
        ys = [v["conc_med"] for v in pt.values() if v["category"] == cat_name]
        ax[1].scatter(xs, ys, c=color, label=cat_name.replace("_", " "), s=28, alpha=0.8)
    if out["fb5a_percell"]:
        ax[1].scatter([out["fb5a_percell"]["ntgt_med"]], [out["fb5a_percell"]["conc_med"]],
                      c="k", marker="*", s=160, label="FB5A (normalizer)")
    ax[1].axhline(out["thresholds"]["narrow_conc"], color="gray", ls=":", lw=0.8)
    ax[1].set_xlabel("FC2 cells contacted per input cell"); ax[1].set_ylabel("per-cell bearing concentration")
    ax[1].set_title("Directional carriers (PFN) vs\nglobal normalizers (FB-tangential/FB5A)")
    ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(_FIG / "w4_fc2_input_map.png", dpi=150); plt.close(fig)
    print(f"[fc2_inputs] figure -> {_FIG}/w4_fc2_input_map.png")


def main() -> None:
    trace_fc2_inputs(_cfg())


if __name__ == "__main__":
    main()
