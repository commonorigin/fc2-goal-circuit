"""gate: two-cue competition test — a UNIFORM FB5A (any form) preserves the ratio; a LOCAL selector distorts it.

Runs off the committed cache `fc2_competition.json`. Asserts the scope-corrected claims: silencing a spatially-
uniform FB5A preserves the competing-column ratio across divisive / subtractive / excitatory forms (so the
discriminator depends on UNIFORMITY = C2, measured, not on sign/form/transmitter); it is robust to the measured
inhibition non-uniformity; and a local-selector control distorts the ratio by orders of magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_competition.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def comp():
    if not _CACHE.exists():
        pytest.skip("fc2_competition.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_competition)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


def test_uniform_fb5a_preserves_ratio_any_form(comp, cfg):
    """Silencing a spatially-uniform FB5A preserves the ratio across divisive/subtractive/excitatory forms ->
    the prediction depends on UNIFORMITY (C2), not on the transmitter sign or the divisive-vs-subtractive form."""
    preserve = float(require(cfg, "our_ai.goal_comp_preserve_max_logdev"))
    for form, ld in comp["uniform_form_logdev"].items():
        assert ld < preserve, f"uniform {form} form did not preserve the ratio ({ld})"


def test_robust_at_the_measured_nonuniformity(comp):
    """The REAL robustness axis: ratio preservation survives the MEASURED inh_w non-uniformity with margin
    (this, not the cancelling scalar sweep, is the empirical content)."""
    assert comp["real_inh_w_bearing_modulation"] < comp["nonuniformity_break_alpha"]


def test_selector_control_distorts_ratio(comp, cfg):
    """A within-FC2 local spatial selector changes which column dominates when silenced -> the ratio distorts,
    with a large separation from any uniform form. Confirms the experiment distinguishes uniform from local."""
    selector_min = float(require(cfg, "our_ai.goal_comp_selector_min_logdev"))
    assert comp["selector_control_logdev"] > selector_min
    assert comp["separation_selector_over_uniform"] > 50    # uniform-vs-local is a clean, large-margin discriminator
