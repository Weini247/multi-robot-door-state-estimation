# Report: X3 robustness batch v1

Status 2026-11-05. Robustness batch under amendment 04 (`configs/experiments/preregistration_v1_amendment_04.json`), including the planned sensitivity analysis with ρ = 0.1.

All numbers come from `X3_v1.json`. The frozen analysis (`tools/analyze_x3.py`) produced that file exactly once; see `X3_v1.analysis_log.jsonl`. The batch is **purely descriptive**: intervals, no p-values, no new decision on H2 and no new claim. H2 remains decided by the X1/X2 test batch and is not supported. Section 4 is interpretation.

## 1 Execution

- **Scope:** 5 conditions, 60 fresh buildings per list (positions 80–139 of the frozen manifest), the four correlated cells only, 7 methods, 2 replicates. That is 16 800 episodes, **all valid**, with no replacement buildings.
- **Cost:** 54.2 CPU-h and 13.1 h counted wall time, against caps of 67 and 24.
- **Interruptions:** the host application that had launched the runner closed and ended it. A resume skipped the 4 274 finished episodes and ran only the missing ones. The result has no duplicate keys, and method, data and code did not change. `sessions.jsonl` lists only the final session, because ended sessions could not write an entry.
- **Code:** run set `2285696d…e2a7` (`configs/experiments/X3_frozen.json`).

**Conditions.**
- `ref_id`: the reference cell sensor, in-distribution test families.
- `rho10_id`: shared error ρ = 0.1 instead of 0.05 at the same marginal error.
- `gauss_id`: a second sensor family, a soft thresholded Gaussian random field (correlation length 2 m) with the same marginal error. The robots keep their cell model, a deliberate misspecification.
- `ref_ood` and `gauss_ood`: the same on the out-of-distribution families.

## 2 Results

Gain of `repair` and coverage loss over the four correlated cells, families 50/50, 95 % CI:

| Condition | Gain vs cross_rank | Gain vs dynamics_ids | Coverage loss vs cross_rank |
|---|---|---|---|
| `ref_id` | +0.050 [+0.023, +0.077] | −0.027 [−0.054, −0.004] | +0.086 [+0.027, +0.143] |
| `rho10_id` | +0.047 [+0.018, +0.075] | −0.002 [−0.036, +0.033] | −0.017 [−0.089, +0.049] |
| `gauss_id` | +0.059 [+0.030, +0.087] | +0.013 [−0.011, +0.033] | +0.079 [+0.026, +0.131] |
| `ref_ood` | +0.046 [+0.018, +0.075] | +0.014 [−0.011, +0.038] | +0.140 [+0.071, +0.202] |
| `gauss_ood` | +0.046 [+0.022, +0.072] | +0.009 [−0.013, +0.033] | +0.138 [+0.067, +0.203] |

Paired change against the reference on the same buildings:

| Shift | Change of gain | Change of repair loss | Change of cross_rank loss |
|---|---|---|---|
| ρ = 0.1 (ID) | −0.003 [−0.027, +0.021] | +0.007 [−0.000, +0.015] | +0.007 [+0.003, +0.012] |
| Gaussian field (ID) | +0.009 [−0.015, +0.034] | −0.009 [−0.019, +0.002] | −0.006 [−0.014, +0.001] |
| Gaussian field (OOD) | +0.000 [−0.022, +0.024] | +0.001 [−0.007, +0.008] | +0.001 [−0.003, +0.005] |

Gain per cell against cross_rank:

| Condition | sta-direct | sta-relay | dyn-direct | dyn-relay |
|---|---|---|---|---|
| `ref_id` | +0.093 [+0.040, +0.146] | +0.120 [+0.061, +0.178] | +0.001 [−0.015, +0.018] | −0.007 [−0.030, +0.016] |
| `rho10_id` | +0.083 [+0.017, +0.146] | +0.085 [+0.019, +0.148] | +0.012 [−0.000, +0.024] | +0.012 [−0.000, +0.024] |
| `gauss_id` | +0.090 [+0.034, +0.144] | +0.100 [+0.044, +0.154] | +0.028 [+0.008, +0.048] | +0.025 [+0.002, +0.048] |
| `ref_ood` | +0.081 [+0.030, +0.134] | +0.078 [+0.014, +0.141] | +0.018 [+0.003, +0.034] | +0.016 [−0.001, +0.033] |
| `gauss_ood` | +0.076 [+0.033, +0.123] | +0.080 [+0.022, +0.137] | +0.017 [+0.000, +0.033] | +0.021 [−0.001, +0.042] |

OOD per family: `ref_ood` dogleg +0.045 [+0.006, +0.083], double_loop +0.047 [+0.006, +0.090]; `gauss_ood` dogleg +0.037 [+0.007, +0.069], double_loop +0.056 [+0.017, +0.097]. The coverage loss there lies between +0.118 and +0.161.

Bytes sent per episode by `repair`: 230 kB (`ref_id`), 155 kB (`rho10_id`), 225 kB (`gauss_id`), 206 kB each in OOD. `dynamics_ids` sends 683–771 kB.

## 3 Findings

1. **The advantage over cross_rank is stable.** It lies at 4.6–5.9 % in all five conditions, every interval excludes 0, and no shift changes it detectably.
2. **The misspecification does no measurable harm.** The Gaussian field does not match the robots' cell model, yet gain and loss do not change detectably. The X1/X2 pattern is therefore not tied to hard cell boundaries. Exactly one misspecification, at matched marginal error, was tested.
3. **Against dynamics_ids the picture is mixed.** On `ref_id`, `repair` is worse here (−2.7 %, interval excludes 0); in the other conditions no difference is detectable. The X1/X2 statement "as good as dynamics_ids" held for the mean over all eight cells and must be qualified for the correlated cells.
4. **Coverage at ρ = 0.1.** The coverage loss disappears (−1.7 % [−8.9, +4.9]) and `repair` sends less (230 → 155 kB).
5. **The dynamic gain in `ref_id` is smaller than in X1/X2** (+0.001 and −0.007 against +0.020 and +0.006). The intervals overlap, and `ref_ood` is higher again; cell gains with switching doors are small and imprecise on 60 buildings.

## 4 Interpretation

Finding 4 fits the logic of the VOI rule: with a stronger shared error, another reading is worth less, so `repair` returns less often and sends less. This is a conjecture; action choices were not counted in this batch.

## 5 Limitations

- Correlated cells only; descriptive, not adjusted for multiplicity.
- The correlation length of 2 m is a setting, and 60 buildings is a smaller sample than in X1/X2.
- One sensor misspecification only; rates and door model stayed correct.
- `gauss_ood` combines two shifts; the channel was unchanged.

## 6 Files

`X3_v1.json`, `X3_v1.md`, `X3_v1.analysis_log.jsonl`, `X3_v1.sessions.jsonl`, `X3_v1.run_plan.json`. The per-episode raw data are hash-bound in the private research repository and available on request.
