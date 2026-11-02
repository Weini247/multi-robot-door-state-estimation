# Port contract — Cross-rank as published low-bandwidth baseline (ARGUS-E07-T04)

As of 2026-11-02. Frozen contract for ARGUS-E07-T05. Machine-readable twin: [crossrank_contract.json](crossrank_contract.json).

## 1. Source and access

| Item | Status |
|---|---|
| Work | J. Bayer, J. Faigl: *Decentralized multi-robot exploration under low-bandwidth communications.* Autonomous Robots 50 (2026), DOI 10.1007/s10514-025-10234-3 (audit reference [A:13]) |
| Journal page | **not read**: the publisher page redirected to a login; no paywall bypass was attempted |
| Text actually read | Authors' preprint PDF hosted by the CTU ComRob lab (`comrob.fel.cvut.cz/papers/auro26mre.pdf`), fetched 2026-11-02, 8 325 366 bytes, SHA-256 `4608d74e3f1af242bc3b31a2a7cadb47c419fc020e3ec1946fcf5d6164b8e1ee`. Text extracted locally, not stored in this repository (copyright). The published version may differ from the preprint |
| Original code | **not found.** The preprint text contains no code or data availability statement and no repository link. Two web searches on 2026-11-02 (title and authors with "code/github") found no Cross-rank repository |
| License | not stated in the preprint text |

Consequence: ARGUS can only build an **adaptation from the published description**. No original implementation is run, and no result of the paper is claimed to be reproduced.

## 2. What the method does (paraphrased from the preprint, §6)

Two steps, executed locally by each robot r for the exploration waypoints W it knows.

1. **Cross-rank (Eqs. 6–8).**
   - Robot r keeps, for every teammate k, the ordered list H_k of positions it has received from k and connects them into a polyline.
   - A waypoint i gets one point per teammate whose received path passes closer than the threshold m_r. The test uses the nearest polyline segment if k has sent more than one position, else the single received position.
   - The rank G_i,cross is the number of such teammates, a non-negative integer.
   - m_r was 1.5 m in simulation and 5.0 m on the real robots. The authors advise more than half the robot diameter and less than the sensing range.
2. **Selection** from W_cr, the waypoints with the lowest Cross-rank:
   - **CR-ESR_p:** the lowest Spread-rank, with path lengths approximated by Euclidean distances. Spread-rank is the sum over teammates of C_r,i / C_k,i, using the teammates' latest positions. Ties are broken by path length from r.
   - **CR-TSP_p:** the first waypoint of a TSP tour over W_cr.
   - **CR-SR_m,p:** like CR-ESR_p but with true path lengths. This needs shared maps.

Only positions are broadcast in CR-ESR_p and CR-TSP_p; the motivating link budget is about 100 B/s per robot. The comparison baseline in the paper is MinPos (Bautin et al. 2012), which ranks a waypoint by how many teammates have a shorter path to it. Evaluation covered exploration time, travelled distance and coverage, in five simulated subterranean environments with 3 to 9 robots and with three real quadrupeds.

## 3. Frozen choice for ARGUS

| Decision | Value | Reason |
|---|---|---|
| Variant | **CR-ESR_p** | Positions only, so it fits the ARGUS sender budget. It is deterministic and needs no TSP solver. CR-TSP_p is noted, not ported (scope budget). CR-SR_m,p is excluded (needs map sharing) |
| m_r | 1.5 m (simulation value from the paper), tunable on validation only | ARGUS rooms are smaller than SubT tunnels; the paper's guidance applies |
| Waypoints W | ARGUS frontier cells of the local map | ARGUS has its own map and frontier definition; the paper's waypoint generation is not ported |
| Communication | New position frames through the ARGUS channel with full byte accounting; no evidence sends, no reservations | As in the original; positions are not free |
| Belief | ARGUS fixed-lag joint belief, identical to the other controls | Cross-rank only coordinates movement |
| Spatial filter | A position is sent only after moving at least 0.5 m since the last sent one | The paper names a spatial filter but gives no value here; the value is an ARGUS choice and tunable |

## 4. Mathematical and informational differences

- **Task.** The paper optimises exploration time and coverage in static environments. ARGUS scores decisions about dynamic doors at pre-planned requests. Cross-rank is therefore an exploration and coordination baseline, not a belief-repair method.
- **Geometry.** The paper uses 3D subterranean maps, LiDAR and a unicycle model with motion noise. ARGUS uses a 2D greybox, a registered pose and an idealised map scan.
- **Waypoints.** ARGUS frontier cells replace the paper's waypoints, so ranks are computed over a different candidate set.
- **Timing.** ARGUS decides on a 1 s cadence with integer ticks. Asynchrony in the paper comes from real communication.
- **Channel.** The ARGUS Gilbert-Elliott loss and rolling budget replace real 868/915 MHz radios. Lost positions simply leave gaps in H_k, as the paper's segment construction intends.
- **No results transfer.** Numbers from the paper are not comparable with ARGUS metrics and are never quoted as ARGUS results.

## 5. Acceptance for E07-T05

- Port only the parts in §3.
- Use a small "original-task" fixture built from the published description: the Fig. 4/5 situation, a waypoint near another robot's received path gets rank 1 and one away from it gets rank 0, plus equation-level cases.
- Mark everything as an ARGUS adaptation.
- Use the same byte accounting as all methods.
- Write tests that explain the changed task semantics.
- If the four-hour budget is not enough, split the task instead of cutting corners.
