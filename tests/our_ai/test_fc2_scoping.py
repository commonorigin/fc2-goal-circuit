"""C2/C3 reproducibility gate: NO within-FC2 winner-take-all in the FlyWire connectome.

Paper C's core negative result rests on two structural connectome facts, re-derived by `analyze_fc2_scoping.py`
into the committed cache `fc2_scoping.json`. These gates assert the cache still supports them (upgrading
CLAIMS_LEDGER C2/C3 from [TO-ATTACH] to [REPRO]). Runs off the committed JSON + config thresholds, so it needs
no GPU and no (gitignored) feather. Thresholds are margins around the measured values, read from config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_scoping.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def scope():
    if not _CACHE.exists():
        pytest.skip("fc2_scoping.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_scoping)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


def test_fb5a_reaches_all_fc2(scope):
    """C1: the 4 FB5A cells contact all 85 FC2 (the global-inhibition substrate)."""
    assert scope["n_fb5a"] == 4
    assert scope["fb5a_reaches"] == scope["n_fc2"] == 85


def test_c2_fb5a_inhibition_is_uniform_across_bearing(scope, cfg):
    """C2: FB5A inhibition carries no bearing signal (per-cell weight uncorrelated with bearing) AND the full
    disynaptic FC2->FB5A->FC2 loop is flat across bearing distance -> a global scaler, not a spatial selector."""
    corr_max = float(require(cfg, "our_ai.goal_scope_bearing_corr_max"))
    mod_max = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    assert abs(scope["inh_corr_cos_bearing"]) < corr_max
    assert abs(scope["inh_corr_sin_bearing"]) < corr_max
    assert scope["loop_modulation"] < mod_max, f"disynaptic loop not flat: {scope['loop_modulation']:.3f}"


def test_c3_no_direct_fc2_recurrence(scope, cfg):
    """C3a: direct FC2->FC2 connectivity is negligible."""
    ff_max = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    assert scope["ff_offdiag_mean"] < ff_max, f"FC2->FC2 not negligible: {scope['ff_offdiag_mean']:.3f}"


def test_c3_hdelta_recurrence_is_antilocal(scope, cfg):
    """C3b: FC2->hDelta->FC2 rises toward ~180 deg (vector-summation), it does NOT couple neighbours -> no
    local positive feedback for a WTA to use."""
    anti_min = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    assert scope["hdelta_far_near_ratio"] > anti_min, f"hDelta path not anti-local: {scope['hdelta_far_near_ratio']:.2f}"
    prof = scope["hdelta_profile"]
    assert prof[-1] > prof[0], "hDelta far weight not greater than near"


def test_c3_vdelta_recurrence_is_weak(scope, cfg):
    """C3c: the FC2->vDelta->FC2 path is weak relative to the (already non-local) hDelta path."""
    vd_max = float(require(cfg, "our_ai.goal_scope_vdelta_max_frac"))
    assert scope["vdelta_peak_frac_of_hdelta"] < vd_max, "vDelta recurrence not weak vs hDelta"


def test_no_wta_structural_conclusion(scope, cfg):
    """The synthesis: global inhibition (C2) + no local excitation (C3) => a normalizer, not a winner-take-all.
    Asserts the JSON jointly satisfies both premises so the paper's negative result is one reproducible object."""
    mod_max = float(require(cfg, "our_ai.goal_scope_uniform_max_modulation"))
    ff_max = float(require(cfg, "our_ai.goal_scope_ff_negligible_max"))
    anti_min = float(require(cfg, "our_ai.goal_scope_antilocal_min_ratio"))
    assert scope["fb5a_reaches"] == scope["n_fc2"]              # global inhibition present
    assert scope["loop_modulation"] < mod_max                   # ...and uniform (not spatial)
    assert scope["ff_offdiag_mean"] < ff_max                    # no direct local excitation
    assert scope["hdelta_far_near_ratio"] > anti_min            # ...and the disynaptic path is anti-local
