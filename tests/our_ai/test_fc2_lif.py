"""Gate: a committed LIF spiking model of FC2+FB5A does not latch a winner-take-all.

Runs off the committed cache `fc2_lif.json`. Turns the paper's "a full spiking model would fail too" from
an assertion into a gated result: a real leaky integrate-and-fire network (threshold/reset/leak) with
connectome FB5A global-inhibition feedback, under the same seeded-basin test as the rate families, has
basin gap ~0deg -- spike timing does not rescue a within-FC2 WTA.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_lif.json")


@pytest.fixture(scope="module")
def lif():
    if not _CACHE.exists():
        pytest.skip("fc2_lif.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_lif)")
    return json.loads(_CACHE.read_text())


def test_lif_network_actually_spikes(lif):
    """The LIF run is non-degenerate (the network fires) -- so the no-latch result is meaningful."""
    assert lif["mean_settled_rate"] > 1e-3


def test_lif_spiking_does_not_latch_a_wta(lif):
    """Seeded-basin gap ~0deg: the settled spike-rate bump is input-determined, not seed-determined ->
    spike timing does not implement a within-FC2 winner-take-all (confirms the rate result)."""
    assert abs(lif["basin_gap_deg"]) < lif["thresholds"]["max_gap_deg"]
