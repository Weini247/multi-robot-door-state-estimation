# Multi-Robot Door-State Estimation

**Repairing stale beliefs in a robot team that explores a building whose doors open and close, over a lossy, bandwidth-limited channel.**

A preregistered simulation study. The code is the study's simulator, estimators, methods and experiment pipeline (internal code name `argus`).

> **Status (2026-11-09).** Three frozen batches have been run, each analysed exactly once:
>
> 1. **Confirmatory test (X1/X2):** 17 920 episodes on 160 held-out buildings. **H2 is not supported**: `repair` lowers the decision loss against the preregistered control by 5.6 % [3.4, 7.9], below the 10 % threshold, and loses 8.9 % coverage, above the 5 % limit. [Report](docs/results/X1_X2_TEST_v1_REPORT.md).
> 2. **Robustness batch (X3, descriptive):** 16 800 episodes on 120 fresh buildings. The gain is stable at 4.6–5.9 %, also under a misspecified sensor model. [Report](docs/results/X3_v1_REPORT.md).
> 3. **Study 2 (separately preregistered, interventional):** 12 000 episodes on 100 new buildings. Targeted re-observation alone helps (+2.2 %), but its gain does not fall with faster doors, which refutes our own explanation from the first study. [Report](docs/results/S2_v1_REPORT.md).

## Results at a glance

- **H2 (primary): not supported.** The gain against `cross_rank` is +0.056 [+0.034, +0.079] (required: lower bound ≥ 0.10). The relative coverage loss is +0.089 [+0.038, +0.135] (allowed: upper bound ≤ 0.05).
- **Where repair helps:** about 11 % gain with static doors and independent sensor errors. The gain falls to about 0 with switching doors, correlated errors and relay.
- **H1:** all three interaction contrasts are about 0. The factors act additively.
- **Robustness:**
  - The result is unchanged at cost ratios 3:1 and 2:1.
  - Out-of-distribution families show the same pattern: a smaller gain and a larger coverage cost.
- **Exploratory, not preregistered:**
  - `repair` matches the communication-heavy `dynamics_ids` (difference +0.001 [−0.014, +0.015]) while sending about a third of the bytes. This compares the configurations selected for lowest loss. `dynamics_ids` was not tuned for bytes: on validation its 20 s send period nearly matched its loss (0.2800 vs 0.2761) with about half the bytes of the busiest robot, so the ratio does not measure efficiency.
  - The control chosen on validation was not the strongest on test, a winner's-curse effect from selecting on 10 buildings.

![H2: both preregistered conditions are missed](docs/figures/h2_decision.svg)

![Loss gain of repair per factorial condition](docs/figures/gain_per_cell.svg)

![Decision loss against bytes sent, per method](docs/figures/loss_vs_bytes.svg)

The figures are generated from `docs/results/X1_X2_TEST_v1.json` by `tools/make_figures.py`, which uses the standard library only.

### Robustness batch (X3)

A second frozen, purely descriptive batch (amendment 04) re-tests the four correlated cells on 60 fresh buildings per list. It adds a stronger shared error (ρ = 0.1) and a second sensor family, a soft thresholded Gaussian field, while the robots keep their cell model.

- The gain against `cross_rank` stays at 4.6–5.9 % in all five conditions, and no shift changes it detectably.
- The misspecified sensor model does no measurable harm.
- Against `dynamics_ids`, `repair` is worse in the reference condition (−2.7 % [−5.4, −0.4]) and indistinguishable elsewhere.

An exploratory [regime map](docs/results/REGIME_MAP.md) of the X1/X2 batch shows `repair` against each control per cell; against `dynamics_ids` no cell differs detectably.

### Study 2: intervening on door tempo and costs

The first study suggested why gains are small with switching doors: under 4:1 costs even certain knowledge of an open door decays below the decision threshold within about 12 s. Study 2 tests this by changing its inputs, door tempo $s$ (both rates scaled, open fraction fixed) and cost ratio, and by splitting `repair` into re-observation only and sending only.

| Hypothesis (re-observation only vs `no_comm`) | Estimate [family-wise interval] | Outcome |
|---|---|---|
| S2-H1: targeted re-observation helps | +0.022 [+0.014, +0.030] | **supported** |
| S2-H2: the gain falls with door tempo | +0.003 [−0.014, +0.021] | **against** |
| S2-H3: window-matched conditions are equal | −0.008 [−0.019, +0.002] | consistent, uninformative given S2-H2 |

- **The window account is refuted for re-observation.** The policy responds to tempo instead: at the fastest doors it decides to look again 326 instead of 145 times and pays with coverage (0.206 vs 0.345 for `no_comm`). The gain stays constant.
- **The window does limit sending.** At the fastest doors sending adds exactly nothing, and with static doors only sending helps (+5–6 %).
- **Targeting matters.** Untargeted return by age adds nothing in any condition.
- **Replication.** The condition matching a study 1 cell reproduces its losses on new buildings (for example `repair` 0.292 vs 0.293).

![Study 2: gain of re-observation only and sending only per condition](docs/figures/s2_gain_by_condition.svg)

## Problem

A small team of robots explores an unknown building. Doors (more generally, passages) change state over time. At pre-planned moments each robot must decide whether a given passage is open. A wrong "open" costs four times as much as a wrong "closed". The robots share evidence over a lossy channel with a byte budget.

Three ways information goes wrong are kept apart on purpose:

| Failure | Example | Remedy |
|---|---|---|
| **Stale** evidence | the door was open when seen, then closed | a time model, and looking again |
| **Duplicated** evidence | the same reading arrives twice via relays | provenance-based deduplication |
| **Dependent** evidence | two readings share a common error (bad visibility) | joint error model; only independent looks help |

The question: *when does a new, as-independent-as-possible observation improve the team's decisions more than further communication of what it already knows?*

## Hypotheses (preregistered)

- **H2, primary.** The candidate method `repair` reduces the mean decision loss relative to the strongest control, under equal resource limits. The claim requires the lower 95 % CI bound of the gain `1 − L_repair / L_control` to be ≥ 0.10, **and** the upper bound of the relative coverage loss to be ≤ 0.05. Otherwise H2 is reported as not supported.
- **H1.** Interaction contrasts between the three factors below show where repair matters.
- **OOD.** The same estimands on building families never used for development, reported separately.

## Design

- **Factorial cells (2 × 2 × 2):**
  - dynamics: static vs. doors switching at 0.01 / 0.02 s⁻¹;
  - sensor error: independent vs. spatially correlated, with the same marginal error of 0.14;
  - circulation: direct only vs. multi-hop relay.
- **Methods:** the candidate `repair` (a decision-relevant observation and repair policy) against six controls. The controls are no communication, naive gossip, TTL with identities, dynamics-aware identities, an age-based revisit policy, and an adaptation of a published low-bandwidth exploration baseline (Cross-rank, Bayer & Faigl 2026; see [`docs/baselines/PORT_CONTRACT.md`](docs/baselines/PORT_CONTRACT.md)).
- **Buildings:** procedurally generated floor plans in a hash-frozen split manifest. There are four lists: validation, calibration, in-distribution test and out-of-distribution test (unseen families). Development only ever touched validation and calibration.
- **Pairing:** all methods see the same buildings, seeds, passage timelines, sensor noise and probe requests. Channel identifiers are method-neutral, so message sizes and loss draws do not depend on the method.
- **Inference:** the unit is the building, analysed with a paired cluster bootstrap (10 000 replicates, stratified by family). The number of test buildings B comes from a calibration pilot (B = 80) and is capped by an approved compute budget. It is never topped up after the fact.

The machine-readable preregistration and its amendments are in [`configs/experiments/`](configs/experiments/):
- `preregistration_v1.json`;
- amendment 01: channel fairness, coverage definition, infrastructure failures, sensitivity analyses;
- amendment 02: a documented deviation in validation tuning and a precise repair-lag definition;
- amendment 03: rules the analysis needs that the preregistration left open (replacements, failed episodes, the Holm family, zero-loss controls);
- amendment 04: the descriptive robustness batch X3 (`X3_frozen.json`), written after the X1/X2 analysis and changing no decision.

Amendments 01–03 were written before any test episode. Study 2 has its own preregistration (`preregistration_s2.json`), run set (`S2_frozen.json`) and building manifest (`configs/splits/splits_v2.json`, 100 buildings disjoint from v1), all frozen before its first test episode. `X1_X2_frozen.json` is the frozen run set. It is recomputed from the committed evidence, and the test runner refuses to start if anything differs. It also holds the hashes of the analysis code (`src/argus/evaluation/statistics.py`, `tools/analyze_x1x2.py`). The analysis refuses to run if its code differs, if the batch is unfinished, or if the source tree changed. A dry run on calibration data is in `docs/results/ANALYSIS_DRY_RUN_calib_v3.md`; it tests the pipeline and is not a result.

## What exists

| Component | Content | Checked by |
|---|---|---|
| `argus.dynamics`, `argus.beliefs` | exact continuous-time Markov door model; exact Bayes micro case; joint (state, shared error) filter; fixed-lag filter with identity deduplication and out-of-order evidence | exact rational enumeration, brute-force path enumeration, arrival-order permutations |
| `argus.world`, `argus.sensors` | 2D floor plans, SE(2) motion, interval collision checks, line of sight; readings with a hidden shared error field at a fixed marginal error | tunnelling and boundary tests, matched-marginal tests |
| `argus.communication` | canonical wire format with byte accounting, rolling sender budget, FIFO links, Gilbert–Elliott burst loss | window-boundary and stress tests |
| `argus.policies` | candidate, controls and diagnostics behind one interface; information boundary enforced (agents cannot import ground truth) | AST import check, hidden-world canary |
| `argus.experiments` | episode runner, preregistered tuning, calibration pilot, frozen run set, batch runner with hard timeouts, pause/resume, crash isolation and budget caps across sessions | about 500 tests |

Calibration and validation data are included with full per-episode raw records: `docs/results/val_tuning_v2.jsonl`, `calib_pilot_v2/` and the calibration probe `test_probe_calib_v3/`. They were used for configuration selection and sample-size planning, **not as evidence**.

The test results are included as aggregated estimates (`docs/results/X1_X2_TEST_v1.*`, `X3_v1.*`, `S2_v1.*`); the study 2 spread pilot on calibration buildings is in `docs/results/s2_pilot_calib_v1/`. The frozen run sets bind source-tree hashes from the private development history. Later code changes alter those hashes, so the guards refuse to re-run an earlier batch on the current tree; the matching commits are available on request. The per-episode raw test data (131 MB) are hash-bound in the private research repository and available on request.

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Verified on Windows 11; Linux has not been tested yet.

```bash
python -m pip install --user uv
python -m uv sync --locked
python -m uv run --locked pytest -q
python -m uv run --locked python tools/test_batch.py freeze --check   # frozen run set matches the evidence
python -m uv run --locked python tools/test_batch.py probe --out runs/probe   # small calibration-only batch
```

**Reproduction check.** Study 2 was rerun in full from a fresh clone of this repository: all 12 000 episodes and the analysis report came out identical. Samples of X1/X2 and X3 at their original commits matched as well (32/32 each). [docs/REPRODUCING.md](docs/REPRODUCING.md) has the steps and the limits; the comparison reports are `docs/results/REPRO_*.json`.

Some tests check internal audit reports that live only in the private research repository (written in German). Those tests are skipped here and say why.

## Limitations

- A simulation study with a simplified 2D world, registered poses (no SLAM), idealised passage detection with a noisy state, and a two-state loss channel rather than radio physics.
- The sample size rests on a 20-building calibration pilot, and the cost model on a small probe.
- Per-decision policy runtime is not recorded, only per-episode runtime.

## Project

Developed as an independent research project. The process follows staged gates: a preregistration before any test data, independent pre-run audits with GO/NO-GO decisions, redacted evidence logs for every task, and amendments instead of silent changes. The full audit trail is kept in a private research repository and is available on request.

## License

- **Code** (`src/`, `tools/`, `tests/`): [MIT](LICENSE).
- **Documentation, figures and result data** (`docs/`, `configs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may reuse them with attribution.

Copyright (c) 2026 Jonas Weingart. If you use this work, please cite the repository.
