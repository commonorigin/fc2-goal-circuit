"""C4/C5 gate: no connectome-parameterized rate model does a within-FC2 winner-take-all.

Runs off the committed cache `fc2_no_wta.json` (built by `prove_fc2_no_wta.py`) + config thresholds. The metric
is bistability: basin_gap ~0 = one fixed point = NO WTA; basin_gap ~sep = two basins = a WTA. Asserts the three
connectome mechanism families (subtractive / divisive / divisive+noise) and the negative control do NOT commit,
while the positive control DOES (so the detector is validated, not vacuous).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_no_wta.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def nowta():
    if not _CACHE.exists():
        pytest.skip("fc2_no_wta.json not built (run: python -m experiments.our_ai.connectome_steering.data.prove_fc2_no_wta)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


@pytest.mark.parametrize("family", ["real_subtractive", "real_divisive", "real_divisive_noise", "real_hdelta_antilocal", "real_vdelta_local_excit"])
def test_c4_connectome_families_do_not_commit(nowta, cfg, family):
    """C4: no mechanism family parameterized by the real wiring is bistable (no WTA) --- including the anti-local
    hDelta recurrence, which is a between-column (mutual) inhibition that could in principle latch a choice but
    does not (this tests, rather than asserts, that the anti-local term is not a WTA)."""
    max_gap = float(require(cfg, "our_ai.goal_nowta_max_gap_deg"))
    assert abs(nowta["basin_gap_deg"][family]) < max_gap, \
        f"{family} shows bistability (unexpected WTA): {nowta['basin_gap_deg'][family]:.1f} deg"


def test_negative_control_global_only_no_wta(nowta, cfg):
    """Negative control: an idealized ring with global inhibition but NO local excitation is monostable —
    isolating local excitation (absent per C3) as the missing WTA ingredient."""
    max_gap = float(require(cfg, "our_ai.goal_nowta_max_gap_deg"))
    assert abs(nowta["basin_gap_deg"]["ctrl_neg_global_only"]) < max_gap


def test_positive_control_is_bistable_detector_valid(nowta, cfg):
    """Positive control: the SAME ring WITH local excitation IS bistable — proving the seeded-basin detector
    can see a WTA when one exists (so the negative result is not an artifact of a blind test)."""
    min_wta = float(require(cfg, "our_ai.goal_nowta_min_wta_gap_deg"))
    assert abs(nowta["basin_gap_deg"]["ctrl_pos_local_excit"]) > min_wta, \
        "positive control not bistable -> detector not validated"
    # and on the REAL irregular bearings (not just the idealized ring): a WTA is still detected (>bound),
    # so the negative results are not an artifact of the real geometry suppressing latch detection.
    assert abs(nowta["basin_gap_deg"]["ctrl_pos_local_excit_realpsi"]) > min_wta, \
        "real-bearings positive control not bistable -> detector could miss a WTA on real geometry"


def test_c5_divisive_no_commit_across_asymmetry(nowta, cfg):
    """C5: the divisive family does not become bistable even at 20% amplitude asymmetry (it shifts by salience
    but never commits via a winner-take-all)."""
    max_gap = float(require(cfg, "our_ai.goal_nowta_max_gap_deg"))
    vs = nowta["divisive_basin_gap_vs_asym"]
    assert "0.20" in vs, "the 20% asymmetry point (C5) is missing"
    for asym, gap in vs.items():
        assert abs(gap) < max_gap, f"divisive commits at asym={asym} (C5 fails): {gap:.1f} deg"


def test_no_wta_conclusion_needs_local_excitation(nowta, cfg):
    """The synthesis: a WTA appears ONLY with local excitation (positive control), and it is absent from every
    connectome-parameterized family — so the FC2 connectome (no local excitation, C3) cannot do a WTA."""
    max_gap = float(require(cfg, "our_ai.goal_nowta_max_gap_deg"))
    min_wta = float(require(cfg, "our_ai.goal_nowta_min_wta_gap_deg"))
    g = nowta["basin_gap_deg"]
    connectome = [g["real_subtractive"], g["real_divisive"], g["real_divisive_noise"], g["ctrl_neg_global_only"]]
    assert all(abs(v) < max_gap for v in connectome)     # no WTA without local excitation
    assert abs(g["ctrl_pos_local_excit"]) > min_wta       # WTA only with it
