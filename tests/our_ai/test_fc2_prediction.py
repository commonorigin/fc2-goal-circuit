"""gate: the FB5A-silencing prediction is DISINHIBITION (robust), not shape/mush (fragile).

Runs off the committed cache `fc2_prediction.json` (built by `analyze_fc2_prediction.py`) + config thresholds.
Asserts the honest sensitivity result: the shape/mush effect is a tuned-operating-point artifact (fragile across
gains), while the disinhibition/amplitude effect is robust across all gains and preserves the bump shape --- so the
paper headlines disinhibition-without-reshaping, not the fragile mush.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.config_access import require

_CACHE = Path("experiments/our_ai/connectome_steering/data/fc2_prediction.json")
_CONFIG = Path("configs/config.yaml")


@pytest.fixture(scope="module")
def pred():
    if not _CACHE.exists():
        pytest.skip("fc2_prediction.json not built (run: python -m experiments.our_ai.connectome_steering.data.analyze_fc2_prediction)")
    return json.loads(_CACHE.read_text())


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(_CONFIG.read_text())


def test_mush_shape_effect_is_fragile(pred, cfg):
    """The shape/mush prediction (FB5A off -> more bumps) holds only near the tuned gains -> fragile, so the paper
    must NOT headline it. Uniform inhibition (C2) cannot reshape a bump; any bump-count change is a threshold artifact."""
    mx = float(require(cfg, "our_ai.goal_pred_mush_max_robust"))
    assert pred["mush_shape_robust_frac"] < mx, \
        f"shape/mush effect is more robust than expected ({pred['mush_shape_robust_frac']}); re-examine the claim"


def test_disinhibition_is_robust(pred, cfg):
    """The disinhibition prediction (FB5A off -> total FC2 activity rises) holds across ALL gains -> the robust,
    headline prediction (direction only; magnitude is a model artifact)."""
    mn = float(require(cfg, "our_ai.goal_pred_disinhib_min_robust"))
    assert pred["disinhibition_robust_frac"] >= mn, \
        f"disinhibition not robust ({pred['disinhibition_robust_frac']})"
    assert pred["amplitude_ratio_off_on"]["min"] > 1.0, "amplitude did not rise in some gain combo"


def test_bump_shape_preserved_under_silencing(pred, cfg):
    """Silencing FB5A rescales the population without reshaping the bump (uniform inhibition) -> the concentration
    is ~preserved. This is the normalizer signature that distinguishes it from a spatial selector."""
    mx = float(require(cfg, "our_ai.goal_pred_shape_preserved_max_dconc"))
    assert pred["shape_preserved_dconc"]["median"] < mx


def test_prediction_is_disinhibition_not_mush(pred):
    """Synthesis: the robust, headline prediction is disinhibition-without-reshaping, NOT the fragile mush ---
    the disinhibition effect is far more robust across gains than the shape effect."""
    assert pred["disinhibition_robust_frac"] > pred["mush_shape_robust_frac"] + 0.4
