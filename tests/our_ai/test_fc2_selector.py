"""Gate: FC2's upstream goal-substrate nomination + the decisive Lanz hDeltaK-PFG exclusion.

Runs off the committed cache `fc2_selector.json`. Asserts the load-bearing, connectome-verifiable claims of
the "Where the selector is" subsection: (1) the Lanz et al. 2025 hDeltaK-PFG attractor does NOT feed FC2
(<0.2% each), so FC2 reads a different substrate; (2) FC2's directional input is the recurrent hDelta network,
not the broad FB-tangential gate; (3) learned valence enters only via MBON->FB5AB->{hDeltaC, FC2}, never
MBON->FC2 directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_selector.json")


@pytest.fixture(scope="module")
def sel():
    if not _CACHE.exists():
        pytest.skip("fc2_selector.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_selector)")
    return json.loads(_CACHE.read_text())


def test_lanz_hdeltaK_pfg_attractor_is_excluded_from_fc2(sel):
    """The best-modelled persistent-goal circuit (Lanz 2025 hDeltaK-PFG) contributes <0.3% of FC2 input each
    -> it is PFL-facing, not FC2's source. This is the subsection's headline positive result."""
    b = sel["thresholds"]["exclude_max_frac"]
    assert sel["hdeltaK_into_fc2_frac"] < b
    assert sel["pfg_into_fc2_frac"] < b
    # and the magnitudes are the reported ones (54 + 88 of 78,193 synapses)
    assert sel["hdeltaK_into_fc2_frac"] < 0.001 and sel["pfg_into_fc2_frac"] < 0.002
    # hDeltaK's OUTPUT is PFL/FB6-facing (PFGs top target), negligibly to FC2 -> confirms the parallel-module claim
    assert "PFGs" in sel["hdeltaK_output_top_targets"]
    assert sel["hdeltaK_output_to_fc2_frac"] < 0.01                     # <1% of hDeltaK output reaches FC2


def test_hdelta_is_the_directional_substrate_not_the_broad_gate(sel):
    """FC2's directional input is the recurrent hDelta network (~28%, per-cell conc ~0.9), dominating the narrow
    PFN stream; the ~55% FB-tangential input is broad (conc ~0.2) and cannot specify a bearing."""
    assert sel["directional_hdelta_frac"] > sel["pfn_frac"]
    assert sel["directional_hdelta_frac"] > 0.2
    assert sel["percell_conc"]["hDeltaC"] > 0.6      # directional
    assert sel["percell_conc"]["FB5A"] < 0.4         # broad
    assert sel["hdelta_within_class_recurrence_frac"] > 0.05   # recurrent (the vector-memory motif)


def test_valence_enters_only_via_fb5ab(sel):
    """Learned valence reaches FC2's substrate through MBON->FB5AB->{hDeltaC, FC2}, never MBON->FC2 directly."""
    assert sel["mbon_into_fc2_frac"] < 1e-6                 # no direct MBON->FC2
    assert sel["mbon_into_fb5ab_frac"] > 0.03               # MBON drives FB5AB
    assert sel["fb5ab_into_hdeltaC_frac"] > 0.05            # FB5AB gates hDeltaC (top named input)
    assert sel["fb5ab_into_fc2_frac"] > 0.01               # and touches FC2
