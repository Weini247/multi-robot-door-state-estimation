# Reproducing the results

This guide shows how to recompute the study 2 batch from a fresh checkout and check it against the published results.

## What has been checked

| Batch | Check | Result |
|---|---|---|
| Study 2 (`S2_v1`) | full rerun of all 12 000 episodes from a fresh public checkout | 12 000/12 000 identical; analysis report identical |
| Main batch X1/X2 | systematic sample of 32 episodes at the commit that produced it | 32/32 identical |
| Robustness batch X3 | systematic sample of 32 episodes at the commit that produced it | 32/32 identical |

"Identical" means two things:
- Each result row matches on every field except timing, memory and the raw-file hash.
- Each raw episode record matches after removing run metadata: code commit, creation time, run id and platform string.

All checks ran on the original machine (Windows 11, Python 3.12). Other hosts and Linux are untested.

## Rerunning study 2

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

1. Clone the repository and install the locked environment:
   ```bash
   git clone https://github.com/Weini247/multi-robot-door-state-estimation
   cd multi-robot-door-state-estimation
   uv sync --locked
   ```
2. Check that the frozen run set matches the code:
   ```bash
   uv run python tools/s2_batch.py freeze --check
   ```
3. Run the batch. It takes about 9 h wall time and 40 CPU-h on the original machine. Interrupted runs resume and fill in only the missing episodes.
   ```bash
   uv run python tools/s2_batch.py run --out runs/s2_repro
   uv run python tools/s2_batch.py status --out runs/s2_repro
   ```
4. Run the frozen analysis into a separate file, so it never overwrites the published one:
   ```bash
   uv run python tools/analyze_s2.py --out runs/s2_repro --json runs/s2_repro/S2_repro.json --report runs/s2_repro/S2_repro.md
   ```
5. Compare `runs/s2_repro/S2_repro.json` with `docs/results/S2_v1.json`. The two should agree on every top-level key.

The batch refuses to start if the source tree differs from the one named in `configs/experiments/S2_approval.json`.

## Limits

- X1/X2 and X3 were produced by earlier versions of the code. That code is not in this repository in its original form. Those batches were checked by sampling at the original commits in the private development repository.
- The raw episode data of the original runs is not published. The episode-level comparison therefore cannot be repeated from this repository alone. Only the analysis-level comparison in step 5 can.

## Stages not run

The optional stages E10 (3D extension), E11 (learned components) and E13/E14 (viewer and arena) were not run. Each waits for a separate owner decision. No result in this repository depends on them.
