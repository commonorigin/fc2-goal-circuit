"""Who supplies FC2's uniform (global) inhibition? A connectome scan of FC2's broad inputs by transmitter.

FB5A's inhibitory identity is a low-confidence prediction (~0.79 GABA, unverified; its siblings are
experimentally gaba-negative -- FB5AB cholinergic (Matheson 2022), FB5AA predicted-glutamatergic /
gaba-negative (Ito 2013)). Rather than bet the normalizer story on one cell, we ask the connectome:
of all cells that contact FC2 broadly (uniformly), which are inhibitory-capable, and how large is FB5A among
them? This makes the global-inhibition substrate a *distributed, evidence-ranked* claim, not a single-cell bet.

Sign convention (Drosophila): GABA -> inhibitory; glutamate -> inhibitory-capable via GluClα chloride channels
(Liu & Wilson 2013); acetylcholine -> excitatory. "Inhibitory-capable" = GABA or glutamate.

Findings (FlyWire): FB5A is the LARGEST single uniform input to FC2 (7,741 syn, 85/85, the only GABA-predicted
type) but only ~11% of the uniform input mass; a large glutamatergic FB-tangential family supplies most of the
inhibitory-capable uniform drive, so the normalizer computation is robust to FB5A's specific transmitter. Honest
caveat: the uniform input is ~half inhibitory-capable / ~half cholinergic (excitatory), so the NET uniform sign
is not resolvable from wiring alone (needs FC2 receptor data).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config_access import require

_D = Path(__file__).resolve().parent
_OUT = _D / "fc2_uniform_inhibitors.json"
_FIG = _D / "scoping_figures"

_COVERAGE_MIN = 0.5     # a type is "uniform/broad" if it reaches >=50% of FC2 cells
_MIN_SYN = 200          # ignore trivial inputs
_INHIB = ("gaba", "glutamate")   # inhibitory-capable transmitters (glutamate via GluClα)


def _cfg() -> dict:
    import yaml
    with open(Path(__file__).resolve().parents[4] / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ids(ann: pd.DataFrame, *types: str) -> list[int]:
    m = ann["cell_type"].astype(str).str.fullmatch("|".join(types), case=False)
    return ann[m]["root_id"].astype("int64").tolist()


def analyze_fc2_uniform_inhibitors(cfg: dict, write: bool = True) -> dict:
    """Rank FC2's uniform inputs by transmitter; quantify FB5A's share of the global-inhibition substrate."""
    raw = Path(require(cfg, "our_ai.avp_cache_dir"))
    ann = pd.read_csv(raw / "flywire_neuron_annotations.tsv", sep="\t", low_memory=False)
    fc2 = set(_ids(ann, "FC2A", "FC2B", "FC2C")); n_fc2 = len(fc2)
    feather = pd.read_feather(raw / "proofread_connections_783.feather")
    into = feather[feather["post_pt_root_id"].astype("int64").isin(fc2)].copy()
    into["pre_pt_root_id"] = into["pre_pt_root_id"].astype("int64")
    meta = ann.set_index("root_id")[["cell_type", "top_nt", "top_nt_conf"]]
    into = into.join(meta, on="pre_pt_root_id")

    # per presynaptic TYPE
    g = into.groupby("cell_type").agg(
        n_pre=("pre_pt_root_id", "nunique"),
        total_syn=("syn_count", "sum"),
        n_fc2_reached=("post_pt_root_id", lambda s: s.astype("int64").nunique()),
        top_nt=("top_nt", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        nt_conf=("top_nt_conf", "mean"),
    ).reset_index()
    g["fc2_coverage"] = g["n_fc2_reached"] / n_fc2
    g = g[g["total_syn"] >= _MIN_SYN]
    g["inhibitory_capable"] = g["top_nt"].isin(_INHIB)
    g["uniform_x_inhib"] = g["fc2_coverage"] * g["nt_conf"] * g["inhibitory_capable"].astype(float)
    g = g.sort_values(["uniform_x_inhib", "total_syn"], ascending=False)

    # the UNIFORM subset (broad inputs) and its transmitter mass breakdown
    uni = g[g["fc2_coverage"] >= _COVERAGE_MIN]
    uni_mass = float(uni["total_syn"].sum())
    by_nt_mass = uni.groupby("top_nt")["total_syn"].sum().to_dict()
    inhib_mass = sum(by_nt_mass.get(k, 0) for k in _INHIB)
    ach_mass = by_nt_mass.get("acetylcholine", 0)

    fb5a = g[g["cell_type"] == "FB5A"].iloc[0]
    fb5a_rank = int(g.reset_index(drop=True).index[g.reset_index(drop=True)["cell_type"] == "FB5A"][0]) + 1
    top10 = g.head(10)[["cell_type", "n_pre", "total_syn", "fc2_coverage", "top_nt", "nt_conf"]].to_dict("records")

    # GABAergic base rate among EXPERIMENTALLY-typed FB-tangential cells (known_nt field = wet-lab, not prediction).
    # Motivates the "FB5A predicted-GABAergic is against the base rate" caution (traceable number for §8).
    fbt = ann[ann["cell_type"].astype(str).str.startswith("FB")].copy()
    known = fbt[fbt["known_nt"].notna()]
    kn = known["known_nt"].astype(str).str.lower()
    n_known_gaba = int((kn.str.contains("gaba") & ~kn.str.contains("gaba-negative")).sum())
    n_known_gaba_neg = int(kn.str.contains("gaba-negative").sum())

    out = dict(
        source="FC2 uniform-input scan by transmitter: distributed global-inhibition substrate, FB5A the largest GABA member",
        n_fc2=n_fc2, coverage_min=_COVERAGE_MIN, inhibitory_capable_nts=list(_INHIB),
        n_uniform_types=int(len(uni)),
        uniform_mass_by_nt=by_nt_mass,
        uniform_inhib_capable_frac=float(inhib_mass / uni_mass),
        uniform_excitatory_ach_frac=float(ach_mass / uni_mass),
        uniform_type_counts_by_nt={k: int(v) for k, v in uni.groupby("top_nt")["cell_type"].nunique().items()},
        fb5a=dict(rank_by_uniform_x_inhib=fb5a_rank, total_syn=int(fb5a["total_syn"]),
                  fc2_coverage=float(fb5a["fc2_coverage"]), top_nt=str(fb5a["top_nt"]),
                  nt_conf=float(fb5a["nt_conf"]),
                  share_of_uniform_mass=float(fb5a["total_syn"] / uni_mass),
                  is_only_gaba_uniform=bool((uni["top_nt"] == "gaba").sum() == 1 and fb5a["top_nt"] == "gaba"),
                  is_largest_uniform=bool(fb5a["total_syn"] == uni["total_syn"].max())),
        top10_uniform_inhibitory=top10,
        fb_tangential_known_gaba=n_known_gaba,
        fb_tangential_known_gaba_negative=n_known_gaba_neg,
        note="FB5A is the largest single uniform input to FC2 and the only GABA-predicted one, but a large "
             "glutamatergic FB-tangential family supplies most inhibitory-capable uniform drive -> the normalizer "
             "computation is distributed and robust to FB5A's transmitter. The uniform input is ~half "
             "inhibitory-capable / ~half cholinergic, so the NET uniform sign needs FC2 receptor data.",
    )
    if write:
        _OUT.write_text(json.dumps(out, indent=2))
        _OUT.with_suffix(".sha256").write_text(f"{hashlib.sha256(_OUT.read_bytes()).hexdigest()}  fc2_uniform_inhibitors.json\n")
        _render_figure(out)

    print(f"[fc2_uniform_inhib] FB5A: rank {fb5a_rank}, {int(fb5a['total_syn'])} syn, "
          f"{100*out['fb5a']['share_of_uniform_mass']:.0f}% of uniform mass, only-GABA={out['fb5a']['is_only_gaba_uniform']}, largest={out['fb5a']['is_largest_uniform']}")
    print(f"[fc2_uniform_inhib] uniform input mass: {100*out['uniform_inhib_capable_frac']:.0f}% inhibitory-capable "
          f"(GABA+glutamate), {100*out['uniform_excitatory_ach_frac']:.0f}% cholinergic (excit)")
    print(f"[fc2_uniform_inhib] uniform TYPES by NT: {out['uniform_type_counts_by_nt']}")
    print(f"[fc2_uniform_inhib] FB-tangential GABA base rate (known_nt): {n_known_gaba} GABAergic vs {n_known_gaba_neg} GABA-negative")

    # gates
    assert out["fb5a"]["is_largest_uniform"], "FB5A is not the largest uniform input (claim broken)"
    assert out["fb5a"]["is_only_gaba_uniform"], "FB5A is not the unique GABA uniform input"
    assert out["uniform_inhib_capable_frac"] > 0.4, "uniform input is not substantially inhibitory-capable"
    assert n_known_gaba_neg > n_known_gaba, "GABAergic is not the minority among experimentally-typed FB-tangentials"
    return out


def _render_figure(out: dict) -> None:
    """FC2's uniform input by transmitter (inhibitory-capable vs excitatory); FB5A the largest / only GABA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fc2_uniform_inhib] figure skipped (matplotlib unavailable: {exc})")
        return
    _FIG.mkdir(exist_ok=True)
    m = out["uniform_mass_by_nt"]
    order = ["gaba", "glutamate", "acetylcholine"] + [k for k in m if k not in ("gaba", "glutamate", "acetylcholine")]
    order = [k for k in order if k in m]
    tot = sum(m.values())
    labels = {"gaba": "GABA\n(inhib)", "glutamate": "glutamate\n(inhib via GluClα)", "acetylcholine": "ACh\n(excit)"}
    colors = {"gaba": "#1b6ca8", "glutamate": "#2a9d3a", "acetylcholine": "#b8432b"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    vals = [100 * m[k] / tot for k in order]
    bars = ax[0].bar([labels.get(k, k) for k in order], vals, color=[colors.get(k, "#888") for k in order])
    for b, v in zip(bars, vals):
        ax[0].annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")
    ax[0].set_ylabel("% of FC2's uniform-input synapses")
    ax[0].set_title(f"FC2's uniform input by transmitter\n({100*out['uniform_inhib_capable_frac']:.0f}% inhibitory-capable, {100*(1-out['uniform_inhib_capable_frac']):.0f}% excit/other)")
    # FB5A in context: top uniform inhibitory-capable types by synapse count
    top = out["top10_uniform_inhibitory"]
    names = [t["cell_type"] for t in top][::-1]
    syns = [t["total_syn"] for t in top][::-1]
    cols = ["#1b6ca8" if t["top_nt"] == "gaba" else ("#2a9d3a" if t["top_nt"] == "glutamate" else "#b8432b") for t in top][::-1]
    ax[1].barh(names, syns, color=cols)
    ax[1].set_xlabel("synapses onto FC2")
    ax[1].set_title("Top uniform inhibitory-capable inputs\n(FB5A = largest, only GABA [blue])")
    fig.tight_layout(); fig.savefig(_FIG / "cui_fc2_uniform_inhibitors.png", dpi=150); plt.close(fig)
    print(f"[fc2_uniform_inhib] figure -> {_FIG}/cui_fc2_uniform_inhibitors.png")


def main() -> None:
    analyze_fc2_uniform_inhibitors(_cfg(), write=True)


if __name__ == "__main__":
    main()
