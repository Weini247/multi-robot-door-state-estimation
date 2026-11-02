# Multi-Robot Door-State Estimation

**Repairing stale beliefs in a robot team that explores a building whose doors open and close, over a lossy, bandwidth-limited channel.**

A preregistered simulation study. The code is the study's simulator, estimators, methods and experiment pipeline (internal code name `argus`).

> **Status (2026-11-02).** The preregistration, validation tuning and calibration pilot are complete. The confirmatory test batch on held-out buildings is frozen and audited, but it has **not been run yet**. This repository contains **no test results** and makes no claim about the hypotheses.

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
- amendment 02: a documented deviation in validation tuning and a precise repair-lag definition.

Both amendments were written before any test episode. `X1_X2_frozen.json` is the frozen run set. It is recomputed from the committed evidence, and the test runner refuses to start if anything differs.

## What exists

| Component | Content | Checked by |
|---|---|---|
| `argus.dynamics`, `argus.beliefs` | exact continuous-time Markov door model; exact Bayes micro case; joint (state, shared error) filter; fixed-lag filter with identity deduplication and out-of-order evidence | exact rational enumeration, brute-force path enumeration, arrival-order permutations |
| `argus.world`, `argus.sensors` | 2D floor plans, SE(2) motion, interval collision checks, line of sight; readings with a hidden shared error field at a fixed marginal error | tunnelling and boundary tests, matched-marginal tests |
| `argus.communication` | canonical wire format with byte accounting, rolling sender budget, FIFO links, Gilbert–Elliott burst loss | window-boundary and stress tests |
| `argus.policies` | candidate, controls and diagnostics behind one interface; information boundary enforced (agents cannot import ground truth) | AST import check, hidden-world canary |
| `argus.experiments` | episode runner, preregistered tuning, calibration pilot, frozen run set, batch runner with hard timeouts, pause/resume, crash isolation and budget caps across sessions | about 500 tests |

Only calibration and validation data are included: `docs/results/val_tuning_v2.jsonl`, `calib_pilot_v2/` and the calibration probe `test_probe_calib_v3/` with full per-episode raw records. They were used for configuration selection and sample-size planning, **not as evidence**.

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Verified on Windows 11; Linux has not been tested yet.

```bash
python -m pip install --user uv
python -m uv sync --locked
python -m uv run --locked pytest -q
python -m uv run --locked python tools/test_batch.py freeze --check   # frozen run set matches the evidence
python -m uv run --locked python tools/test_batch.py probe --out runs/probe   # small calibration-only batch
```

Some tests check internal audit reports that live only in the private research repository (written in German). Those tests are skipped here and say why.

## Limitations

- A simulation study with a simplified 2D world, registered poses (no SLAM), idealised passage detection with a noisy state, and a two-state loss channel rather than radio physics.
- The sample size rests on a 20-building calibration pilot, and the cost model on a small probe.
- Per-decision policy runtime is not recorded, only per-episode runtime.

## Project

Developed as an independent research project. The process follows staged gates: a preregistration before any test data, independent pre-run audits with GO/NO-GO decisions, redacted evidence logs for every task, and amendments instead of silent changes. The full audit trail is kept in a private research repository and is available on request.

No license has been chosen yet. Until then all rights are reserved; please ask before reuse.
