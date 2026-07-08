"""cross-connectome gate: C2/C3 (no within-FC2 WTA) replicate in the HEMIBRAIN.

Runs off the committed cache `fc2_scoping_hemibrain.json` (built by `analyze_fc2_scoping_hemibrain.py`) + the
shared `goal_scope_*` thresholds. Asserts the no-WTA structure holds in a second, independently reconstructed
brain — killing the "single-connectome" objection. The C2 direct-uniformity check uses the completeness-
normalized FB5A weight (the raw value carries a partial-volume gradient, asserted to be completeness-driven).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_scoping_hemibrain.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def hb():
    if not _CACHE.exists():
        pytest.skip("fc2_scoping_hemibrain.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_scoping_hemibrain)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


def test_hemibrain_has_the_populations(hb):
    """The hemibrain independently contains FC2 (88), FB5A (4), and the hDelta/vDelta interneurons."""
    assert hb["n_fb5a"] == 4
    assert hb["n_fc2"] >= 80 and hb["fb5a_reaches"] == hb["n_fc2"]
    assert hb["n_hdelta"] > 100 and hb["n_vdelta"] > 100


def test_c2_fb5a_uniform_replicates(hb, cfg):
    """C2 in the hemibrain: the disynaptic FC2->FB5A->FC2 loop is flat, AND the completeness-normalized direct
    weight is bearing-uncorrelated (the raw gradient is a partial-volume artifact, asserted below)."""
    corr_max = float(require(cfg, "our_ai.goal_scope_bearing_corr_max"))
    mod_max = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    assert hb["loop_modulation"] < mod_max, f"loop not flat: {hb['loop_modulation']:.3f}"
    assert hb["inh_corr_completeness_normalized"] < corr_max, \
        f"FB5A bearing-organized after completeness norm: {hb['inh_corr_completeness_normalized']:.3f}"


def test_raw_gradient_is_a_completeness_artifact(hb, cfg):
    """The honest caveat, asserted: the RAW per-cell FB5A gradient tracks per-column reconstruction completeness
    (so it is a partial-volume artifact, not bearing tuning)."""
    compl_min = float(require(cfg, "our_ai.goal_scope_completeness_corr_min"))
    assert hb["completeness_corr"] > compl_min, \
        f"raw FB5A gradient not explained by completeness: {hb['completeness_corr']:.2f}"


def test_c3_no_local_recurrence_replicates(hb, cfg):
    """C3 in the hemibrain: FC2->FC2 negligible, FC2->hDelta->FC2 anti-local, vDelta weak."""
    ff_max = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    anti_min = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    vd_max = float(require(cfg, "our_ai.goal_scope_vdelta_max_frac"))
    assert hb["ff_offdiag_mean"] < ff_max, f"FC2->FC2 not negligible: {hb['ff_offdiag_mean']:.3f}"
    assert hb["hdelta_far_near_ratio"] > anti_min, f"hDelta not anti-local: {hb['hdelta_far_near_ratio']:.2f}"
    assert hb["vdelta_peak_frac_of_hdelta"] < vd_max, "vDelta not weak"


def test_two_brains_agree_conclusion(hb, cfg):
    """Synthesis: the no-WTA structure (uniform FB5A pathway + no local excitation) holds in BOTH FlyWire
    and the hemibrain — not a single-dataset artifact."""
    mod_max = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    ff_max = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    anti_min = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    assert hb["loop_modulation"] < mod_max             # FB5A pathway uniform
    assert hb["ff_offdiag_mean"] < ff_max              # no direct local excitation
    assert hb["hdelta_far_near_ratio"] > anti_min      # disynaptic path anti-local
