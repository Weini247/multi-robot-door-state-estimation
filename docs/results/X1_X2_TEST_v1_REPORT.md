# Results: X1/X2 test batch v1

As of 2026-11-04. Confirmatory test of H2 and H1 on the held-out test buildings.

Every number below comes from `docs/results/X1_X2_TEST_v1.json`. The frozen analysis (`tools/analyze_x1x2.py`) produced that file **exactly once**; see `X1_X2_TEST_v1.analysis_log.jsonl`. The machine-generated summary is in `X1_X2_TEST_v1.md`.

**Status of each section.** Section 2 is confirmatory. Section 3 is secondary, as preregistered. Section 4 is exploratory: those findings were not registered as hypotheses and should be read as motivation for follow-up studies, not as evidence.

## 1. Execution

- **Run set and approval:** frozen run set `X1_X2_frozen.json`, sha256 `b16b0593…a359c`; source tree `4f60a0a2…97111`. The approval in `configs/experiments/X1_X2_approval.json` is 73 CPU-h, 30 h wall, B = 80.
- **Sessions** (`X1_X2_TEST_v1.sessions.jsonl`):
  1. Paused by the owner after 37 groups.
  2. Resumed and finished regularly (`stop_reason: null`).

  Pausing is part of the design. On resume, the interrupted group was completed with its missing episodes only.
- **Scope:** 160 groups (80 positions × 2 lists), 17 920 episodes, **all `ok`**. There were no failures, no timeouts and no replacement buildings, so the analysed prefix equals the planned B = 80.
- **Cost:** 60.0 CPU-h and 14.9 h wall, both within the caps.
- **Code version:** One commit was used throughout. Commit hashes in the session and run-plan files refer to the private research history.

## 2. Confirmatory: H2

- **Candidate:** `repair` (h5.0).
- **Preregistered H2 control:** `cross_rank` (m1.5f1.0), chosen on validation as the strongest simple control.
- **Data:** in-distribution test list, 80 buildings (40 loop, 40 tree).
- **Inference:** paired cluster bootstrap, stratified by family, 10 000 draws, seed 20261103.

| Quantity | Estimate [95 % CI] | Preregistered condition |
|---|---|---|
| Loss gain 1 − L_repair / L_cross_rank | +0.056 [+0.034, +0.079] | lower bound ≥ 0.10 |
| Absolute difference L_repair − L_cross_rank | −0.016 [−0.023, −0.010] | (reported) |
| Relative coverage loss (closed, final) | +0.089 [+0.038, +0.135] | upper bound ≤ 0.05 |

**Decision: H2 is not supported.** Both conditions fail.

`repair` reliably lowers the decision loss relative to `cross_rank`; the interval excludes 0. The effect is smaller than the preregistered relevance threshold of 10 %, and `repair` answers noticeably fewer queries. p-values play no role in this decision.

## 3. Secondary (preregistered)

### 3.1 Gain per cell (Holm over 8 cells + 3 H1 contrasts)

| Cell (dynamics, sensor error, circulation) | Gain [95 % CI] | p (Holm) |
|---|---|---|
| static, independent, direct | +0.112 [+0.077, +0.149] | 0.002 |
| static, independent, relay | +0.112 [+0.071, +0.152] | 0.002 |
| static, correlated, direct | +0.084 [+0.030, +0.134] | 0.021 |
| static, correlated, relay | +0.058 [−0.015, +0.125] | 0.563 |
| dynamic, independent, direct | +0.043 [+0.023, +0.062] | 0.002 |
| dynamic, independent, relay | +0.035 [+0.013, +0.057] | 0.016 |
| dynamic, correlated, direct | +0.020 [+0.005, +0.036] | 0.065 |
| dynamic, correlated, relay | +0.006 [−0.011, +0.022] | 1.000 |

The gain is largest with static doors and independent sensor errors, about 11 % as a point estimate. It decreases monotonically with door dynamics and with correlated errors. In the hardest cell it cannot be distinguished from zero.

### 3.2 H1: interaction contrasts of the candidate

| Contrast | Estimate [95 % CI] | p (Holm) |
|---|---|---|
| dynamics × circulation | −0.000 [−0.006, +0.006] | 1.000 |
| dynamics × sensor | −0.005 [−0.018, +0.008] | 1.000 |
| sensor × circulation | +0.005 [−0.001, +0.011] | 0.563 |

None of the contrasts differs from zero. The factors act on the gain approximately **additively**. Each one reduces the gain, but none detectably amplifies or dampens another.

### 3.3 Sensitivity

| Variant | Gain [95 % CI] |
|---|---|
| loss as defined in the spec (M6) | +0.056 [+0.034, +0.079] |
| cost ratio 3:1 (re-scored) | +0.063 [+0.042, +0.084] |
| cost ratio 2:1 (re-scored) | +0.064 [+0.044, +0.085] |
| pairs without episode failures | +0.056 [+0.034, +0.079] |

The result does not depend on the 4:1 cost ratio. Policies ran at 4:1 (amendment 01, O5); only the decision was re-scored from the recorded probabilities. The last row equals the main result because no episode failed.

### 3.4 Out of distribution (descriptive, per family, 40 buildings each)

| Family | Gain vs. cross_rank | Coverage loss |
|---|---|---|
| dogleg | +0.028 [+0.004, +0.053] | +0.166 [+0.068, +0.246] |
| double_loop | +0.060 [+0.022, +0.101] | +0.107 [+0.036, +0.167] |

On building families never used during development the picture is the same: the gain is smaller or similar, and the coverage cost is higher.

### 3.5 Further quantities

- **Bytes sent per episode** (sum over robots):

  | Method | Bytes |
  |---|---|
  | cross_rank | 8.3 kB |
  | age_revisit | 44 kB |
  | no_comm | 50 kB (control messages only) |
  | repair | 268 kB |
  | naive_gossip | 367 kB |
  | ttl_ids | 637 kB |
  | dynamics_ids | 777 kB |

  All methods ran under the same budget limits; this is how much of the budget each one actually uses.
- **Calibration on answered queries:**

  | Method | Brier | NLL |
  |---|---|---|
  | dynamics_ids | 0.177 | 0.513 |
  | repair | 0.180 | 0.523 |
  | no_comm | 0.186 | 0.538 |
  | age_revisit | 0.186 | 0.538 |
  | cross_rank | 0.190 | 0.550 |
  | ttl_ids | 0.200 | 0.691 |
  | naive_gossip | 0.214 | 1.773 |

  Methods answer different subsets, so these values are only roughly comparable. The high NLL of naive gossip shows the expected overconfidence from double-counted evidence.
- **Repair lag** (amendment 02, D2; descriptive, horizon τ = 300 ticks): The restricted mean lag is 297.9 ticks for repair and 298.4 for cross_rank, per robot. Pooled over the team, it is 291.4 and 295.2.

  Most events are not repaired within the window under either method (censoring 93–99 %). In this form the metric discriminates poorly. That is a finding about the metric.
- **Runtime per episode** (median CPU): repair 14.0 s, dynamics_ids 15.4 s, all other methods 10–11 s.

## 4. Exploratory (not preregistered)

1. **repair matches dynamics_ids at about a third of the communication.**
   - The gain against dynamics_ids is +0.001 [−0.014, +0.015], so no difference is detectable.
   - repair sends 268 kB, dynamics_ids 777 kB.
   - The same holds out of distribution: +0.008 (dogleg) and +0.004 (double_loop), both intervals including 0.
   - This compares the configurations selected for lowest loss. `dynamics_ids` was not tuned for bytes: on validation its 20 s send period nearly matched its loss (0.2800 vs 0.2761) with about half the bytes of the busiest robot, so the ratio does not measure efficiency.

   This suggests that targeted re-observation can substitute for communication without losing decision quality. An efficiency claim would need its own study with a predefined efficiency metric and an equivalence margin.
2. **The control selected as strongest on validation is not the strongest on test.**
   - cross_rank was selected on 10 complete validation buildings by a very small margin: loss 0.2745, against 0.2761 for dynamics_ids, 0.2829 for age_revisit and 0.2830 for no_comm.
   - On test, cross_rank (0.290) ranks behind dynamics_ids (0.274), no_comm (0.285) and age_revisit (0.285). Selection among nearly equal candidates on a small sample is a typical winner's-curse effect.
   - repair also has a positive gain with CI above 0 against every control except dynamics_ids: age_revisit +0.040, no_comm +0.040, ttl_ids +0.115, naive_gossip +0.276.

   This does not change the confirmatory result, which is defined against cross_rank. It does mean H2 would not have been supported against any more plausible choice of control.
3. **The price of repair is coverage.** repair has the lowest coverage of all methods (0.262; the others range from 0.288 to 0.314). The policy sends robots back to re-observe and covers less new space in exchange. A follow-up study could make the re-observation budget an explicit trade-off against coverage.

## 5. Interpretation

Targeted, as-independent-as-possible re-observation measurably improves a robot team's passage decisions. The improvement is about 4–6 %, not the 10 % set as practically relevant. It comes at a cost in coverage. Its advantage shrinks additively with changing doors and with shared sensor errors. Compared with communication-heavy methods, it reaches the same decision quality with far fewer bytes.

Limitations:
- A simulation with a simplified 2D world, registered poses and a two-state loss channel.
- Two in-distribution and two out-of-distribution building families.
- A sample size based on a 20-building pilot.
- Section 4 is exploratory.

## 6. Files

| File | Content |
|---|---|
| `X1_X2_TEST_v1.json` | all estimates with confidence intervals |
| `X1_X2_TEST_v1.md` | machine-generated summary |
| `X1_X2_TEST_v1.analysis_log.jsonl` | log of the single analysis run |
| `X1_X2_TEST_v1.sessions.jsonl` | batch sessions |
| `X1_X2_TEST_v1.run_plan.json` | bound run plan |

The per-episode raw data (131 MB) are not included. They are bound by SHA-256 in the private research repository and are available on request.
