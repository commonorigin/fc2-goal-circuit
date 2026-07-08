"""gate: FC2's upstream input map + the directional-carrier vs normalizer split.

Runs off the committed cache `fc2_inputs.json` (built by `trace_fc2_inputs.py`) + config thresholds. Asserts the
measured, distributed goal-drive picture: FB-tangential dominates FC2 input, MBON is ~0 direct (indirect via
FB-tangential), PFN cells are NARROW directional carriers while FB5A/FB-tangential cells are BROAD normalizers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_inputs.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def inp():
    if not _CACHE.exists():
        pytest.skip("fc2_inputs.json not built (run: python -m experiments.our_ai.connectome_steering.data.trace_fc2_inputs)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


def test_input_map_is_complete(inp):
    """The FC2 input map is fully typed (every presynaptic partner has a cell type)."""
    assert inp["typed_coverage"] > 0.95
    assert inp["n_presynaptic_types"] > 100


def test_fb_tangential_dominates_input(inp):
    """FB-tangential neurons are the dominant FC2 input stream (the broad contextual/gating drive)."""
    assert inp["category_fractions"]["FB_tangential_broad"] > 0.4


def test_mbon_is_not_a_direct_input(inp):
    """A real refinement of the initial expectation: MBONs exist in the brain but do NOT directly feed FC2 (MB influence is
    indirect). Guarded so '~0' is meaningful, not a naming miss."""
    assert inp["n_mbon_in_brain"] > 0, "no MBONs found by name -> the direct-input check would be vacuous"
    assert inp["mbon_direct_frac"] < 0.01


def test_pfn_are_narrow_directional_carriers(inp, cfg):
    """PFN cells each target a NARROW bearing band of FC2 -> they can inject a goal direction (the directional
    feedforward drive)."""
    narrow = float(require(cfg, "our_ai.goal_upstream_narrow_conc"))
    assert inp["pfn_percell_conc_mean"] >= narrow
    assert any(t.startswith("PFN") for t in inp["narrow_directional_types"])


def test_fb5a_is_a_broad_normalizer(inp, cfg):
    """FB5A blankets nearly all FC2 with a bearing-flat projection -> the global normalizer, at single-cell
    resolution (confirming C2). It is BROAD, not directional."""
    broad = float(require(cfg, "our_ai.goal_upstream_broad_conc"))
    fb5a = inp["fb5a_percell"]
    assert fb5a["conc_med"] <= broad
    assert fb5a["ntgt_med"] >= 0.9 * inp["n_fc2"]
    assert "FB5A" in inp["broad_normalizer_types"]


def test_directional_vs_normalizer_separation(inp, cfg):
    """Synthesis: the two roles are cleanly separated at single-cell resolution — PFN (directional) sits far
    above FB5A (normalizer) in per-cell bearing concentration (gap >= the narrow-broad threshold span)."""
    narrow = float(require(cfg, "our_ai.goal_upstream_narrow_conc"))
    broad = float(require(cfg, "our_ai.goal_upstream_broad_conc"))
    assert inp["pfn_percell_conc_mean"] - inp["fb5a_percell"]["conc_med"] >= (narrow - broad)
