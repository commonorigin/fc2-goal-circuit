"""Gate: FC2's global-inhibition substrate is a distributed FB-tangential family, FB5A the largest / only GABA member.

Runs off the committed cache `fc2_uniform_inhibitors.json`. Supports the reframed §4 claim: the normalizer
computation does not hinge on FB5A's (unverified) transmitter — FB5A is the largest single uniform input and the
only GABA-predicted one, but a large glutamatergic FB-tangential family supplies most inhibitory-capable uniform
drive. Honest caveat asserted too: the uniform input is only ~half inhibitory-capable (the rest cholinergic).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_uniform_inhibitors.json")


@pytest.fixture(scope="module")
def ui():
    if not _CACHE.exists():
        pytest.skip("fc2_uniform_inhibitors.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_uniform_inhibitors)")
    return json.loads(_CACHE.read_text())


def test_fb5a_is_the_largest_and_only_gaba_uniform_input(ui):
    """FB5A is the single largest uniform input to FC2 and the only GABA-predicted one -> naming it is not arbitrary."""
    assert ui["fb5a"]["is_largest_uniform"]
    assert ui["fb5a"]["is_only_gaba_uniform"]


def test_uniform_inhibition_substrate_is_distributed(ui):
    """FB5A is only a minority (~12%) of the uniform-input mass; a broad glutamatergic family supplies most of the
    inhibitory-capable uniform drive -> the normalizer computation is robust to FB5A's specific transmitter."""
    assert ui["fb5a"]["share_of_uniform_mass"] < 0.25
    assert ui["uniform_type_counts_by_nt"].get("glutamate", 0) >= 10   # a large glutamatergic family, not one cell


def test_uniform_input_is_substantially_inhibitory_capable(ui):
    """>=40% of the uniform-input mass is inhibitory-capable (GABA or glutamate/GluClα) -> a real global-inhibition
    substrate exists independent of FB5A."""
    assert ui["uniform_inhib_capable_frac"] >= 0.4


def test_honest_caveat_competing_excitation(ui):
    """Honest limit: the uniform input is NOT purely inhibitory — a large cholinergic (excitatory) fraction exists,
    so the NET uniform sign is not resolvable from wiring alone. Assert the caveat's factual basis holds."""
    assert ui["uniform_excitatory_ach_frac"] > 0.2


def test_fb_tangential_gaba_base_rate_is_hostile(ui):
    """Traceable base rate (§8): among experimentally-typed FB-tangential cells, GABAergic is the minority
    (14 vs 371 GABA-negative) — so FB5A's predicted-GABAergic label sits against an unfavourable prior."""
    assert ui["fb_tangential_known_gaba_negative"] > ui["fb_tangential_known_gaba"]
    assert ui["fb_tangential_known_gaba"] == 14
    assert ui["fb_tangential_known_gaba_negative"] == 371
