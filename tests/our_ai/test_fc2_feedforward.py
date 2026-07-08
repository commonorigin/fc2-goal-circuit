"""gate: no within-FC2 WTA via the FEEDFORWARD max-selection route either.

Runs off the committed cache `fc2_feedforward.json`. Closes the reviewer route the bistability test does
not: a high-exponent divisive normalization could approach a hard max-selector without recurrence. Asserts
(pure model geometry, no transmitter): (1) at the task-agnostic operating point p0 the circuit CO-REPRESENTS
two competing goals; (2) the high-p collapse is INIT-INDEPENDENT (a fixed anatomical bias, not a bistable
choice); (3) at high p the collapse does NOT track the stronger goal (bias beats input) -> not a selector.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_feedforward.json")


@pytest.fixture(scope="module")
def ff():
    if not _CACHE.exists():
        pytest.skip("fc2_feedforward.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_feedforward)")
    return json.loads(_CACHE.read_text())


def test_operating_point_corepresents_competitors(ff):
    """At the task-agnostic exponent p0, two equal competing goals are BOTH represented (no feedforward WTA)."""
    assert ff["corepresentation_loser_frac_at_P0"] > ff["thresholds"]["corep_frac"]


def test_collapse_is_only_supraphysiological(ff):
    """Any collapse to a single column appears only well above the operating point (not at p0)."""
    assert ff["first_collapse_power"] is None or ff["first_collapse_power"] > ff["operating_point_power"] * 2


def test_high_p_collapse_is_init_independent(ff):
    """Seeding toward goal A vs goal B yields the identical winner at every p -> a fixed bias, NOT a
    history-dependent (bistable) choice. This is the WTA-defining property, and it is absent."""
    assert ff["init_spread_max_pct"] < ff["thresholds"]["init_eps_pct"]


def test_high_p_collapse_does_not_track_the_stronger_goal(ff):
    """A functional goal-WTA must select the stronger competitor. At high p the fixed bias wins instead
    (the stronger input loses), so the collapse is not goal-selection at any exponent."""
    assert ff["stronger_goal_winnerA_at_P0"] > 0.55        # picks the stronger goal at p0
    assert ff["stronger_goal_winnerA_at_maxp"] < 0.55      # fails to, at supra-physiological p
