"""FB5A divisive-normalization / soft-k-WTA for FC2 goal selection — runtime.

Scoping (tested 2026-07-05) found NO within-FC2 winner-take-all in the connectome (FB5A inhibition is uniform;
the ring has no local recurrent excitation). FB5A's real computation is **divisive normalization / soft
k-WTA** — the same global-inhibition motif as the mushroom body's APL. It ENFORCES a single clean goal bump
(Mussells-Pires' single-bump result) and salience-weights; it does not originate the choice (that is
distributed: few candidates upstream + commit-and-hold). A RATE computation. Declared assumption: the FB5A
inhibition is DIVISIVE (shunting) — a receptor property the connectome does not resolve.

`GlobalInhibitionNormalizer` is reusable (FC2/FB5A here; MB/APL later): one inhibitory pool normalizes a ring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE = Path(__file__).resolve().parent / "connectome_steering" / "data" / "fc2_selection.json"


def divisive_normalize(drive: np.ndarray, inh_w: np.ndarray, pool: np.ndarray,
                       power: float, g_inh: float, sigma: float, iters: int = 60) -> np.ndarray:
    """Global-inhibition soft-k-WTA (divisive normalization). Reusable (FC2/FB5A, MB/APL).

    x_i = relu(drive_i)^power / (sigma + g_inh * inh_w_i * (pool . x)).  One inhibitory pool normalizes the
    whole ring -> the strongest survive best (soft k-WTA) + a single clean bump is enforced; no local
    recurrence -> it sharpens/salience-weights but does not force a single winner (matches the tested behavior).
    `g_inh=0` = FB5A silenced. Robust prediction (analyze_fc2_prediction.py): silencing DISINHIBITS FC2 (activity
    rises, bump tuning preserved) -- NOT a shape/"mush" change (that effect is fragile/gain-dependent)."""
    x = np.maximum(drive, 0.0)
    x = x / (x.max() + 1e-9)
    for _ in range(iters):
        num = np.maximum(drive, 0.0) ** power
        x = num / (sigma + g_inh * inh_w * float(pool @ x) + 1e-9)
        x = x / (x.max() + 1e-9)
    return x


class GlobalInhibitionNormalizer:
    """A connectome-parameterized global-inhibition soft-k-WTA (divisive normalization).

    Loads real per-cell inhibition weights + pool weights + gains from a committed cache. Reusable: FC2 uses
    the FB5A cache here; the mushroom body's APL sparsification is the same motif (a later reuse)."""

    def __init__(self, cache_path: Path = _CACHE):
        d = json.loads(Path(cache_path).read_text())
        self.inh_w = np.asarray(d["inh_w"], dtype=float)
        self.pool = np.asarray(d["pool"], dtype=float)
        self.power = float(d["power"]); self.g_inh = float(d["g_inh"]); self.sigma = float(d["sigma"])
        self.n = int(d["n_fc2"])

    def __call__(self, drive: np.ndarray, fb5a_on: bool = True) -> np.ndarray:
        """Normalize a drive over the ring -> the (soft-k-WTA) population, L1-normalized to sum=1 (so it is a
        drop-in for the goal bump that feeds W_FC2->PFL). `fb5a_on=False` silences FB5A (the prediction control)."""
        x = divisive_normalize(drive, self.inh_w, self.pool, self.power,
                               self.g_inh if fb5a_on else 0.0, self.sigma)
        s = x.sum()
        return x / s if s > 1e-9 else x
