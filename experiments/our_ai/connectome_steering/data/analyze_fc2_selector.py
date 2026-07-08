"""Where is FC2's goal set? A connectome nomination of the upstream goal-substrate + a decisive exclusion.

Paper C shows FC2 does not select the goal internally (no within-FC2 ring-attractor WTA). Selection is
upstream. This script names the most likely substrate from the wiring, and — the load-bearing positive
result — EXCLUDES the best-modelled alternative:

  * The Lanz et al. 2025 hDeltaK-PFG disinhibition-gated persistent-goal attractor does NOT feed FC2
    (hDeltaK -> FC2 = 0.069%, PFGs -> FC2 = 0.113%; both < 0.2%). Its readout is PFL/PFR, not FC2 --
    a parallel goal module. So FC2 reads a DIFFERENT substrate.
  * FC2's structured (directional) input is the recurrent hDelta columnar network (~28%; hDeltaJ 6.1%,
    hDeltaC 3.7%; per-cell bearing concentration ~0.9), not the broad, non-directional FB-tangential
    stream (~55%, concentration ~0.2, which cannot specify a bearing).
  * The only route from mushroom-body (learned valence) into FC2's directional substrate is
    MBON -> FB5AB -> {hDeltaC (6.9%, top named input), FC2 (1.7%)}; MBON -> FC2 direct = 0%.

Nomination: the hDelta recurrent network (hDeltaC-led, hDeltaJ the PFN-vector-injection partner) holds
the committed goal FC2 reads; PFN injects the bearing (self-motion vector), FB5AB gates by valence,
FB5A normalizes. Falsifiable double-dissociation: setting the FC2 goal needs hDeltaC/FB5AB and is
insensitive to hDeltaK/PFG (whose manipulation instead affects PFL-side persistence).

Honest limits: recurrence is necessary-not-sufficient for a holding attractor; transmitter signs are
predicted; the disinhibitory-gate timing and hDeltaC-vs-hDeltaK driver-line identity are unresolved
from static wiring. Numbers are FlyWire synapse counts (`proofread_connections_783`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_selector.json"
_WIRES = _D / "steering_wires.json"
_FIG = _D / "scoping_figures"

_EXCLUDE_MAX_FRAC = 0.003   # hDeltaK/PFG each < 0.3% of FC2 input => excluded as FC2's source


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _category(t: str) -> str:
    if t.startswith(("hDelta", "vDelta")) or t.startswith("FC"):
        return "hDelta_recurrent_directional"
    if t.startswith("PFN"):
        return "PFN_directional"
    if t.startswith("FB"):
        return "FB_tangential_broad"
    if t.startswith("MBON"):
        return "MBON"
    return "other"


def analyze_fc2_selector(cfg: dict, write: bool = True) -> dict:
    """Nominate FC2's upstream goal-substrate; exclude the Lanz hDeltaK-PFG attractor."""
    raw = Path(require(cfg, "our_ai.avp_cache_dir"))
    min_cell_syn = int(require(cfg, "our_ai.goal_upstream_min_cell_syn"))
    ann = pd.read_csv(raw / "flywire_neuron_annotations.tsv", sep="\t", low_memory=False)
    feather = pd.read_feather(raw / "proofread_connections_783.feather")
    psi = np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])
    type_of = dict(zip(ann["root_id"].astype("int64"), ann["cell_type"].astype(str)))

    FC2 = list(ann[ann["cell_type"].astype(str).str.fullmatch("FC2A|FC2B|FC2C", case=False)]["root_id"].astype("int64"))
    fc2_idx = {b: i for i, b in enumerate(FC2)}; n = len(FC2)

    inc = feather[feather["post_pt_root_id"].isin(set(FC2))].copy()
    inc["pre_type"] = inc["pre_pt_root_id"].map(type_of).fillna("<untyped>")
    total = float(inc["syn_count"].sum())
    by_type = inc.groupby("pre_type")["syn_count"].sum().sort_values(ascending=False)

    # input composition by category (directional vs broad)
    cats: dict[str, float] = {}
    for ty, s in by_type.items():
        cats[_category(ty)] = cats.get(_category(ty), 0.0) + float(s)
    cats = {k: v / total for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}

    def frac(pre_types) -> float:
        m = inc["pre_type"].isin(pre_types) if isinstance(pre_types, (list, set)) \
            else inc["pre_type"].astype(str).str.startswith(pre_types)
        return float(inc[m]["syn_count"].sum() / total)

    def route_frac(pre_type: str, post_ids: set) -> float:
        e = feather[feather["post_pt_root_id"].isin(post_ids)]
        t = e["pre_pt_root_id"].astype("int64").map(type_of)
        return float(e[t == pre_type]["syn_count"].sum() / max(e["syn_count"].sum(), 1))

    def percell_conc(type_name: str) -> float:
        e = inc[inc["pre_type"] == type_name]; concs = []
        for _, g in e.groupby("pre_pt_root_id"):
            w = np.zeros(n)
            for b, s in g.groupby("post_pt_root_id")["syn_count"].sum().items():
                if b in fc2_idx:
                    w[fc2_idx[b]] = s
            if w.sum() < min_cell_syn:
                continue
            concs.append(float(abs((w * np.exp(1j * psi)).sum()) / w.sum()))
        return float(np.median(concs)) if concs else float("nan")

    hDC = set(ann[ann["cell_type"] == "hDeltaC"]["root_id"].astype("int64"))
    fb5ab = set(ann[ann["cell_type"] == "FB5AB"]["root_id"].astype("int64"))

    # hDelta within-class recurrence (hDelta* -> hDelta*, fraction of all hDelta input) -- the "vector memory" motif
    hD_all = set(ann[ann["cell_type"].astype(str).str.startswith("hDelta")]["root_id"].astype("int64"))
    hd_in = feather[feather["post_pt_root_id"].isin(hD_all)]
    hd_in_pretypes = hd_in["pre_pt_root_id"].astype("int64").map(type_of).astype(str)
    hdelta_recurrence = float(hd_in[hd_in_pretypes.str.startswith("hDelta")]["syn_count"].sum() / max(hd_in["syn_count"].sum(), 1))

    # hDeltaK's OUTPUT targets (where the Lanz attractor projects) -- confirms it is PFL/FB6-facing, not FC2
    hdK = set(ann[ann["cell_type"] == "hDeltaK"]["root_id"].astype("int64"))
    hdK_out = feather[feather["pre_pt_root_id"].isin(hdK)]
    hdK_out_tot = max(hdK_out["syn_count"].sum(), 1)
    hdK_out_types = hdK_out.assign(t=hdK_out["post_pt_root_id"].astype("int64").map(type_of).astype(str)) \
        .groupby("t")["syn_count"].sum().sort_values(ascending=False)
    hdeltaK_output_top = {str(t): float(s / hdK_out_tot) for t, s in hdK_out_types.head(6).items()}
    hdeltaK_output_to_fc2 = float(hdK_out_types.reindex(["FC2A", "FC2B", "FC2C"]).fillna(0).sum() / hdK_out_tot)

    out = dict(
        source="FC2 upstream goal-substrate nomination + Lanz hDeltaK-PFG exclusion",
        n_fc2=n, total_input_syn=total,
        input_composition=cats,
        directional_hdelta_frac=frac("hDelta"),
        pfn_frac=frac("PFN"),
        fb_tangential_broad_frac=cats.get("FB_tangential_broad", 0.0),
        hdeltaJ_frac=frac(["hDeltaJ"]), hdeltaC_frac=frac(["hDeltaC"]),
        percell_conc=dict(hDeltaC=percell_conc("hDeltaC"), hDeltaJ=percell_conc("hDeltaJ"),
                          PFNm=percell_conc("PFNm"), FB5A=percell_conc("FB5A")),
        # THE EXCLUSION
        hdeltaK_into_fc2_frac=frac(["hDeltaK"]), pfg_into_fc2_frac=frac(["PFGs"]),
        # THE VALENCE ROUTE
        mbon_into_fc2_frac=frac("MBON"),
        fb5ab_into_fc2_frac=frac(["FB5AB"]),
        fb5ab_into_hdeltaC_frac=route_frac("FB5AB", hDC),
        mbon_into_fb5ab_frac=route_frac_mbon(feather, type_of, fb5ab),
        mbon_into_hdeltaC_frac=route_frac_mbon(feather, type_of, hDC),
        hdelta_within_class_recurrence_frac=hdelta_recurrence,
        hdeltaK_output_top_targets=hdeltaK_output_top,
        hdeltaK_output_to_fc2_frac=hdeltaK_output_to_fc2,
        thresholds=dict(exclude_max_frac=_EXCLUDE_MAX_FRAC),
        note="hDelta recurrent network (hDeltaC-led, hDeltaJ vector-injection) is FC2's directional goal "
             "substrate; the Lanz hDeltaK-PFG attractor is excluded (<0.2% into FC2, PFL-facing). Valence "
             "enters via MBON->FB5AB->{hDeltaC,FC2}. Nomination, not proof: persistence/gating need physiology.",
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_selector.json\n")
        _render_figure(out)

    print(f"[fc2_selector] directional hDelta={100*out['directional_hdelta_frac']:.1f}% (conc hDeltaC={out['percell_conc']['hDeltaC']:.2f}) "
          f"| PFN={100*out['pfn_frac']:.1f}% | FB-tangential broad={100*out['fb_tangential_broad_frac']:.1f}% (conc FB5A={out['percell_conc']['FB5A']:.2f})")
    print(f"[fc2_selector] EXCLUSION: hDeltaK->FC2={100*out['hdeltaK_into_fc2_frac']:.3f}%  PFGs->FC2={100*out['pfg_into_fc2_frac']:.3f}%  (both < {100*_EXCLUDE_MAX_FRAC:.1f}%)")
    print(f"[fc2_selector] VALENCE: MBON->FC2={100*out['mbon_into_fc2_frac']:.2f}% (direct); MBON->FB5AB={100*out['mbon_into_fb5ab_frac']:.1f}%; FB5AB->hDeltaC={100*out['fb5ab_into_hdeltaC_frac']:.1f}%; FB5AB->FC2={100*out['fb5ab_into_fc2_frac']:.1f}%")

    # gates
    assert out["hdeltaK_into_fc2_frac"] < _EXCLUDE_MAX_FRAC, "hDeltaK is not excluded from FC2 input"
    assert out["pfg_into_fc2_frac"] < _EXCLUDE_MAX_FRAC, "PFGs are not excluded from FC2 input"
    assert out["directional_hdelta_frac"] > out["pfn_frac"], "hDelta is not the dominant directional input"
    assert out["mbon_into_fc2_frac"] < 1e-6, "MBON reaches FC2 directly (valence route claim broken)"
    assert out["fb5ab_into_hdeltaC_frac"] > 0.05 and out["mbon_into_fb5ab_frac"] > 0.03, "FB5AB valence bridge broken"
    assert out["percell_conc"]["hDeltaC"] > 0.6 and out["percell_conc"]["FB5A"] < 0.4, "directional/broad concentration split broken"
    return out


def route_frac_mbon(feather, type_of, post_ids: set) -> float:
    e = feather[feather["post_pt_root_id"].isin(post_ids)]
    t = e["pre_pt_root_id"].astype("int64").map(type_of).astype(str)
    return float(e[t.str.startswith("MBON")]["syn_count"].sum() / max(e["syn_count"].sum(), 1))


def _render_figure(out: dict) -> None:
    """FC2 input directional-vs-broad; the hDeltaK/PFG exclusion; the MBON->FB5AB valence route."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_selector] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    # left: input composition, directional (hDelta+PFN) vs broad (FB-tangential)
    comp = out["input_composition"]
    disp = {"FB_tangential_broad": "FB-tangential\n(broad, non-directional)",
            "hDelta_recurrent_directional": "hΔ/vΔ recurrent\n(37%; 28% directional hΔ)", "PFN_directional": "PFN\n(directional)",
            "MBON": "MBON", "other": "other"}
    keys = [k for k in comp]
    colors = {"hDelta_recurrent_directional": "#1b6ca8", "PFN_directional": "#2a9d3a",
              "FB_tangential_broad": "#b8b0a0", "MBON": "#b8432b", "other": "#ddd"}
    ax[0].barh([disp.get(k, k) for k in keys][::-1], [100 * comp[k] for k in keys][::-1],
               color=[colors.get(k, "#ccc") for k in keys][::-1])
    ax[0].set_xlabel("% of FC2 input synapses")
    ax[0].set_title("FC2's input: directional carriers vs broad gate")
    # right: the exclusion + valence route as a labelled bar of key edges
    labels = ["hΔK→FC2\n(Lanz attractor)", "PFGs→FC2\n(Lanz attractor)", "MBON→FC2\n(direct)",
              "FB5AB→FC2", "FB5AB→hΔC", "MBON→FB5AB"]
    vals = [100 * out["hdeltaK_into_fc2_frac"], 100 * out["pfg_into_fc2_frac"], 100 * out["mbon_into_fc2_frac"],
            100 * out["fb5ab_into_fc2_frac"], 100 * out["fb5ab_into_hdeltaC_frac"], 100 * out["mbon_into_fb5ab_frac"]]
    cols = ["#b8432b", "#b8432b", "#b8432b", "#1b6ca8", "#1b6ca8", "#2a9d3a"]
    bars = ax[1].barh(labels[::-1], vals[::-1], color=cols[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax[1].annotate(f"{v:.2f}%", (v, b.get_y() + b.get_height() / 2), va="center", fontsize=7, xytext=(2, 0), textcoords="offset points")
    ax[1].axvline(0.2, color="k", ls=":", lw=0.8, label="0.2% (both Lanz edges below)")
    ax[1].set_xlabel("% of target's input synapses")
    ax[1].set_title("Lanz hΔK–PFG excluded from FC2;\nvalence via MBON→FB5AB")
    ax[1].legend(fontsize=6.5)
    fig.savefig(_FIG / "csel_fc2_selector.png", dpi=150); plt.close(fig)
    print(f"[fc2_selector] figure -> {_FIG}/csel_fc2_selector.png")


def main() -> None:
    analyze_fc2_selector(_cfg(), write=True)


if __name__ == "__main__":
    main()
