# Report: Study 2 — when and why does targeted re-observation help?

Status 2026-11-08. All numbers come from `S2_v1.json`. The frozen analysis (`tools/analyze_s2.py`) produced that file exactly once; see `S2_v1.analysis_log.jsonl`.

**What is what.** Section 2 is confirmatory under `configs/experiments/preregistration_s2.json`. Section 3 is descriptive, section 4 interpretation. All statements concern the mechanism **in the simulator**. They are no causal claim about real robots, and there is no additive split of `repair`'s gain into re-observation and sending.

## 1 Question and design

The X1/X2 report suggested an explanation for the small gains with switching doors. With 4:1 costs the decision "open" needs a belief of at least 0.8, and without new evidence even certainty decays below that within

$$W=\frac1k\ln\frac{1-\pi}{0.8-\pi}\approx 11.9\ \mathrm{s}.$$

Study 2 tests this by intervening on its two inputs. It was preregistered after the X1/X2 analysis, with its own frozen run set, seed and 100 fresh buildings disjoint from all 560 buildings of the first study (`configs/splits/splits_v2.json`).

- **Interventions:** door tempo $s \in \{0, 0.25, 0.5, 1, 2\}$ scales both switching rates at fixed $\pi = 1/3$ ($s = 1$ is study 1, $s = 0$ static doors); cost ratio 2:1 and 4:1. The cost ratio changes the threshold, the policies' VOI and the loss scale together. At 4:1 the window is 47.6 / 23.8 / 11.9 / 5.9 s for $s$ = 0.25 / 0.5 / 1 / 2; at $s = 1$ and 2:1 it is 23.1 s.
- **Arms:** `repair` is split into `repair_observe` (may only re-observe) and `repair_send` (may only send), otherwise identical. Also run: `repair`, `no_comm`, `age_revisit` (untargeted return by age) and `dynamics_ids`. All configurations are reused from study 1; nothing was retuned.
- **Fixed:** independent sensor ($e = 0.14$), direct circulation, channel and budget as in study 1.
- **Scope:** 10 conditions × 6 arms × 2 replicates = 120 paired episodes per building, 12 000 in total.

**Hypotheses** (gain of `repair_observe` over `no_comm`; a set of conditions is the unweighted mean of per-condition gains):

| | Estimand | Outcome rule |
|---|---|---|
| **S2-H1** | mean over the 8 dynamic conditions | supported: lower bound > 0; against: upper bound < 0.03; else inconclusive |
| **S2-H2** | mean at $s = 0.25$ minus mean at $s = 2$ (both cost ratios) | as S2-H1 |
| **S2-H3** | mean difference over three window-matched pairs, e.g. ($s$ = 1, 2:1, 23.1 s) minus ($s$ = 0.5, 4:1, 23.8 s) | consistent: interval within [−0.03, 0.03]; against: interval excludes 0; else inconclusive |

Intervals are Bonferroni family-wise ($1 - 0.05/3$) from a paired cluster bootstrap over buildings, stratified by family, 10 000 draws, seed 20261107. A spread pilot on the 20 calibration buildings (`s2_pilot_calib_v1/`) gave minimum detectable effects of 0.009, 0.030 and 0.028 at 80 % power. S2-H2 is predicted by almost any "faster doors are harder" account; S2-H3 is the discriminating test of the window account.

## 2 Confirmatory results

All 12 000 episodes were valid, with no replacement buildings and no interruption (39.8 CPU-h and 9.0 h wall, against caps of 46 and 18). Run set `98f7261f…35d9`.

| Hypothesis | Estimate [family-wise interval] | Outcome |
|---|---|---|
| **S2-H1** targeted re-observation helps | +0.022 [+0.014, +0.030] | **supported** |
| **S2-H2** the gain falls with door tempo | +0.003 [−0.014, +0.021] | **against** |
| **S2-H3** window-matched pairs are equal | −0.008 [−0.019, +0.002] | **consistent** |

**Reading:**
- Targeted re-observation reliably lowers the loss, by about 2 %.
- The gain **does not detectably depend on door tempo**. The window account predicts a falling gain and is refuted for re-observation.
- S2-H3 being consistent does not support the window account: if the gain does not depend on the window at all, window-matched pairs are equal too. The rank correlation of the dynamic gains with the window is −0.21, if anything the wrong sign (descriptive).

## 3 Descriptive results

![Gain per condition](../figures/s2_gain_by_condition.svg)

### 3.1 How the policy responds to tempo (4:1)

| $s$ | Re-observation decisions | Coverage `repair_observe` | Coverage `no_comm` | Gain of re-observation |
|---|---|---|---|---|
| 0 (static) | 12 | 0.236 | 0.239 | −0.002 [−0.027, +0.019] |
| 0.25 | 145 | 0.277 | 0.281 | +0.034 [+0.021, +0.048] |
| 0.5 | 218 | 0.274 | 0.309 | +0.016 [+0.004, +0.029] |
| 1 | 199 | 0.260 | 0.338 | +0.028 [+0.018, +0.037] |
| 2 | 326 | 0.206 | 0.345 | +0.026 [+0.016, +0.035] |

The faster the doors, the more often `repair_observe` decides to go back, and the more coverage that costs: at $s = 2$, 0.206 instead of 0.345. The loss gain stays about the same. **The policy holds its benefit constant by investing more.**

### 3.2 Re-observation and sending apart

- **Static doors:** re-observation adds nothing (+0.019 / −0.002 at 2:1 / 4:1), sending adds a lot (+0.049 / +0.063).
- **Switching doors:** each arm adds about 2–3 %.
- **At $s = 2$ and 4:1** `repair_send` sends nothing useful any more; its gain is exactly 0, and `repair` sends only 30 kB there instead of about 200 kB.

For sending, the window logic fits: a relayed reading arrives too late to lift a decision over the threshold. For re-observation it does not, because the policy looks closer and more often.

### 3.3 Targeted versus untargeted

`age_revisit`, which returns without targeting, lies between −0.002 and +0.005 in all conditions, practically zero. **Targeting matters**, not returning as such.

### 3.4 Against the strongest simple control

`repair` beats `dynamics_ids` in 5 of the 8 dynamic conditions, by +0.014 to +0.024 with intervals above 0, and by +0.030 [+0.012, +0.051] with static doors at 4:1. In study 1 no difference was detectable in the comparable cell. This finding is descriptive and unadjusted.

### 3.5 Replication

Condition $s = 1$, 4:1 corresponds to the study 1 cell `dyn-ind-direct`:

| Method | Study 1 | Study 2 |
|---|---|---|
| `repair` | 0.293 | 0.292 |
| `no_comm` | 0.303 | 0.300 |
| `age_revisit` | 0.304 | 0.300 |
| `dynamics_ids` | 0.297 | 0.298 |

The values reproduce on new buildings.

## 4 Interpretation

The analytical window account predicted that targeted re-observation helps less with fast doors. That is not the case. Section 3.1 suggests why: the VOI rule responds to tempo, returns more often and closer, and so keeps its gain at the price of coverage. The window does limit the value of **sending**.

This shifts the explanation of the small gain in study 1. It is not that re-observation becomes useless under dynamics; it is the coverage that re-observation costs, and the low value of sending under dynamics. This interpretation rests on descriptive numbers and is not a separate test.

## 5 Negative findings

- S2-H2: the predicted decline with tempo does not occur; the window account is refuted in this form.
- Re-observation is worthless with static doors.
- `repair_send` adds nothing at $s = 2$ and 4:1.
- `age_revisit` adds nothing in any condition.

## 6 Limitations

- One sensor (independent), direct circulation, one parameter set.
- Configurations were tuned at $s = 1$, 4:1 and reused everywhere.
- The cost ratio acts on threshold and costs together.
- The condition comparisons in section 3 are not adjusted.

## 7 Files

`S2_v1.json`, `S2_v1.md`, `S2_v1.analysis_log.jsonl`, `S2_v1.sessions.jsonl`, `S2_v1.run_plan.json`, `s2_pilot_calib_v1/`, `../figures/s2_gain_by_condition.svg`. The per-episode raw test data are hash-bound in the private research repository and available on request.
