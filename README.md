# fc2-goal-circuit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21252417.svg)](https://doi.org/10.5281/zenodo.21252417)

Analysis code, committed result caches, and tests for the paper:

**How the fly holds a single goal: normalization, not selection, in *Drosophila* FC2**
Gioele Nanni, Christopher Lee (Common Origin).

Preprint: *bioRxiv* (DOI to be added on posting).
Archived release: *Zenodo*, https://doi.org/10.5281/zenodo.21252417

The paper asks, from the connectome, what circuit produces the distant feedback inhibition
between FC2 goal neurons, and whether FC2 *selects* one goal (a winner-take-all) or *normalizes*
a goal set elsewhere. Every figure and every reported number in the paper is produced by a
re-runnable script here that writes a committed JSON cache; a test suite checks each number against
its cache; and every load-bearing number is traced in [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).

## Layout

```
experiments/our_ai/
 fc2_selection.py divisive-normalization model (Eq. 1)
 connectome_steering/data/
 analyze_fc2_scoping.py -> c2, c3 figures (FB5A uniformity; anti-local recurrence)
 analyze_fc2_scoping_hemibrain.py -> w3 (second-connectome replication)
 trace_fc2_inputs.py -> w4 (FC2 input map)
 analyze_fc2_selector.py -> csel (upstream selector + Lanz exclusion)
 analyze_fc2_uniform_inhibitors.py -> cui (distributed uniform-inhibition substrate)
 prove_fc2_no_wta.py -> c4c5 (seeded-basin no-WTA test + controls)
 analyze_fc2_lif.py -> (fc2_lif.json) (committed LIF spiking run)
 analyze_fc2_feedforward.py -> cff (feedforward max-selection sweep)
 analyze_fc2_prediction.py -> u1 (FB5A-silencing robustness)
 analyze_fc2_competition.py -> u6 (two-cue competition test)
 build_fc2_selection.py builds fc2_selection.json from the raw FlyWire feather
 *.json committed result caches (each with a .sha256)
 scoping_figures/*.png the 10 paper figures
tests/our_ai/test_fc2_*.py one gate per claim (checks each number against its cache)
configs/config.yaml the parameters the scripts read (no value is hardcoded)
CLAIMS_LEDGER.md every number -> its script / cache / test
paper.pdf the manuscript
```

## Install

```sh
pip install -r requirements.txt # numpy, pandas, matplotlib, pyyaml, pytest (+ pyarrow for raw data)
```

Python 3.11+.

## Run the tests (reproduce every number from the committed caches)

From the repository root:

```sh
PYTHONPATH=. pytest tests/ -q
```

All gates run off the committed JSON caches, so they pass without any connectome download.

## Regenerate figures / re-run the analyses

Two tiers, by data need:

**Runs out of the box** (reads only committed caches): the dynamical / model scripts

```sh
PYTHONPATH=. python experiments/our_ai/connectome_steering/data/prove_fc2_no_wta.py
PYTHONPATH=. python experiments/our_ai/connectome_steering/data/analyze_fc2_lif.py
PYTHONPATH=. python experiments/our_ai/connectome_steering/data/analyze_fc2_prediction.py
PYTHONPATH=. python experiments/our_ai/connectome_steering/data/analyze_fc2_feedforward.py
PYTHONPATH=. python experiments/our_ai/connectome_steering/data/analyze_fc2_competition.py
```

Each writes its JSON cache and its figure into `scoping_figures/`.

**Requires the raw connectome** (not shipped — see below): the wiring-measurement scripts
`analyze_fc2_scoping.py`, `analyze_fc2_scoping_hemibrain.py`, `analyze_fc2_selector.py`,
`analyze_fc2_uniform_inhibitors.py`, `trace_fc2_inputs.py`. Their committed output caches are
included, so all figures and tests downstream of them reproduce without the raw data.

## Reproducing from raw data (optional)

The wiring is sourced from two public connectomes, which are too large to ship here:

- **FlyWire** (Dorkenwald et al. 2024; Schlegel et al. 2024) — download the proofread connections
 and neuron annotations from the FlyWire Codex (https://codex.flywire.ai), and place
 `proofread_connections_783.feather` and `flywire_neuron_annotations.tsv` under the directory named
 by `our_ai.avp_cache_dir` in `configs/config.yaml`.
- **Hemibrain** (Scheffer et al. 2020) — the traced-adjacency export from neuPrint
 (https://neuprint.janelia.org), under `our_ai.goal_scope_hemibrain_dir`.

Transmitter predictions are the FlyWire classifier of Eckstein et al. 2024. With the raw data in
place, the five wiring scripts above regenerate their caches from scratch.

## Notes

- No parameter is fit to any downstream task; all live in `configs/config.yaml`.
- Result caches are content-addressed (`*.sha256`) so drift is detectable.
- `CLAIMS_LEDGER.md` is the source of truth mapping each reported number to its artifact.
