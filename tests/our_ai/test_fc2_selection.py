"""E.FC2 gate for the FB5A divisive-normalization / soft-k-WTA module.

Scoping (tested) found NO within-FC2 winner-take-all in the connectome; FB5A's real role is divisive
normalization / soft k-WTA (the same global-inhibition motif as the mushroom body's APL) that ENFORCES a
single clean goal bump. These gates validate that role + the falsifiable prediction (silence FB5A -> the
single-bump cleanup fails). Runs off the committed cache (fc2_selection.json); rate, not spiking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.our_ai.fc2_selection import GlobalInhibitionNormalizer, divisive_normalize

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_selection.json")
_WIRES = Path("experiments/our_ai/connectome_steering/data/steering_wires.json")


@pytest.fixture(scope="module")
def norm():
    return GlobalInhibitionNormalizer()


@pytest.fixture(scope="module")
def psi():
    return np.asarray(json.loads(_WIRES.read_text())["psi_fc2"])


def _vm(c, psi, k=2.0):
    b = np.exp(k * np.cos(c - psi))
    return b / b.sum() * len(psi)


def _n_bumps(x, psi, rel=0.5):
    xa = x[np.argsort(psi)]
    if xa.max() <= 0:
        return 0
    a = xa > rel * xa.max(); idx = np.where(a)[0]
    r = 1 + int((np.diff(idx) > 1).sum())
    return r - 1 if (a[0] and a[-1] and r > 1) else r


def test_cache_is_real_fb5a_weights(norm):
    """The module is parameterized by the REAL FB5A wiring (4 GABA cells -> all 85 FC2), not hand values."""
    assert norm.n == 85 and norm.inh_w.shape == (85,) and norm.pool.shape == (85,)
    assert (norm.inh_w > 0).all(), "FB5A reaches every FC2 (measured: 85/85)"


def test_single_goal_is_one_clean_bump(norm, psi):
    """A single committed goal -> ONE clean, bounded bump (the goal->PFL message)."""
    x = norm(_vm(0.6, psi))
    assert np.isfinite(x).all() and (x >= 0).all()
    assert abs(x.sum() - 1.0) < 1e-6                       # L1-normalized (drop-in for the goal bump)
    assert _n_bumps(x, psi) == 1
    conc = abs((x * np.exp(1j * psi)).sum()) / x.sum()
    assert conc > 0.6, f"single-goal bump not concentrated: {conc:.2f}"


def test_fb5a_enforces_single_bump_prediction(norm, psi):
    """THE falsifiable prediction, in-model: competing candidates -> FB5A ON enforces ONE bump; silence FB5A
    (g_inh=0) -> the cleanup fails and multiple bumps coexist (the goal 'mushes'). Mussells-Pires' single-bump
    role, as a testable prediction (silence FB5A in vivo -> selection degrades)."""
    drive = _vm(-1.0, psi) + _vm(1.1, psi)                 # two competing candidates ~120deg apart
    on = norm(drive, fb5a_on=True)
    off = norm(drive, fb5a_on=False)
    assert _n_bumps(on, psi) == 1, "FB5A ON did not enforce a single bump"
    assert _n_bumps(off, psi) > _n_bumps(on, psi), "silencing FB5A did not degrade the single-bump cleanup"


def test_reusable_normalizer_is_deterministic(norm, psi):
    """The GlobalInhibitionNormalizer is a reusable global-inhibition soft-k-WTA (also serves MB/APL); output
    is deterministic + matches the low-level function."""
    drive = _vm(0.3, psi)
    a = norm(drive)
    b = divisive_normalize(drive, norm.inh_w, norm.pool, norm.power, norm.g_inh, norm.sigma)
    assert np.allclose(a, b / b.sum())
    assert np.allclose(a, norm(drive))                    # deterministic
