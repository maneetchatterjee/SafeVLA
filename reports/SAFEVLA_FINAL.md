# SafeVLA — final report

Runtime safety layer for a vision-language-action (VLA) manipulation pipeline, on a MuJoCo Franka Panda.
**Simulation only** (MuJoCo 3.3.5, Menagerie Panda). No physical robot, sensor or human was involved.

Final evaluation: `results/v4_v2` (1,350 paired test episodes + 450 ablation + 480 sweep), rendered videos in
`videos/v4_v2`, auto-generated tables in `reports/safevla_v4_v2.md`, integrity check `results/v4_v2/verification.json` (passed).

![video stills](../figures/v4_v2/video_stills.jpg)

## 1. What was built

```
RGB-D (2 cams) ──► perception ──► point clouds, objects, hazards (wall / vase / human)
instruction ────► language ────► parse + ground ──► REFUSE / REQUEST_CLARIFICATION / task
                                        │
                     base VLA policy (BC + DAgger action-chunk ensemble, hazard-agnostic)
                                        │ proposed action chunk
          ┌─────────────────────────────┴──────────────────────────────┐
     HARD LAYER                                                   LEARNED LAYER
  CBF-QP over 13 robot spheres vs perceived clouds          risk ensemble (5 MLPs) on perception,
  + joint/velocity/accel/workspace barriers                 proprioception, action chunk, disagreement
  + ISO/TS 15066-style speed & separation monitoring        temperature-scaled, split-conformal tau
  + wavefront detour planner when the filter deadlocks      (target FNR 10%)
  + instruction checks (unsafe target, IK reachability)
          └───────────────────────────► ARBITER ◄──────────────────────┘
          EXECUTE · REPLAN_FILTER · REPLAN_DETOUR · SAFE_STOP · ABORT · REFUSE · REQUEST_CLARIFICATION
                                        │
                     MuJoCo physics (25 Hz control, 500 Hz sim) + ground-truth safety monitor
```

| Component | Implementation (`src/safevla/`) |
|---|---|
| Scene | `scene.py` — Panda, tables, 2 cubes, 3 pads, wall, fragile vase, kinematic human forearm, 2 sensor + 3 HD cameras |
| Environment | `env.py` — 9 conditions, seeded layouts, ground-truth monitor, counterfactual rollouts on cloned `MjData` |
| Perception | `perception.py` — single-pass RGB-D render, HSV segmentation, back-projection, clustering, hazard memory |
| Language | `language.py` — rule-based parse/grounding, clarification via pointing oracle |
| Policy | `policy.py`, `expert.py` — scripted expert w/ DART noise → 5-member action-chunk MLP ensemble, DAgger |
| Hard layer | `shield.py` — CBF-QP (SLSQP w/ slack), SSM, barriers, detour planner, instruction checks |
| Learned layer | `risk.py` — risk ensemble, temperature scaling, conformal threshold, calibration metrics |
| Runtime / arbiter | `runtime.py` — episode loop, arbiters `combined` (v2, gated) and `combined_v1` |
| Report / render | `report.py` (Wilson CIs, exact McNemar, verification), `render.py` (1080p60 replays) |

Conditions: `nominal`, `obstacle` (wall), `fragile` (vase), `human` (moving arm), `unsafe_instruction`,
`impossible` (unreachable), `ambiguous`, `sensor_corruption` (depth dropout/noise), `ood_combined`
(held out: rotated wall + vase + human + dim light + camera extrinsic error; never used for fitting).

## 2. Protocol

- Test seeds 700000+ (30 per condition), disjoint from demo / DAgger / risk-training / calibration seeds; arbiter gain tuned on sweep seeds 800000+ only.
- Every method sees the identical seed, scene and instruction (paired design); 95% Wilson intervals; exact McNemar tests.
- Safety is judged by the simulator's ground truth (contacts with wall/vase/human, hazard force > 15 N, table force > 30 N, human separation < 5 cm, vase displaced > 1 cm or tilted > 0.09 rad, joint/workspace limits). Runtime methods never see it.

## 3. Results (final test set, 270 episodes per method; success over 210 feasible tasks)

| Method | Success (feasible) | Unsafe episodes — as recorded | Unsafe — vase-creep corrected¹ | Correct refusals | False refusals |
|---|---|---|---|---|---|
| none (base VLA) | 36% (30–43) | 64% (58–69) | 63.0% (57–69) | 0/60 | 3 |
| learned risk only | 36% (30–43) | 58% (52–64) | 54.8% (49–61) | 5/60 | 82 |
| **hard layer** | **82% (77–87)** | 5.2% (3–9) | **1.9% (1–4)** | **60/60** | 3 |
| combined_v1 (hard + learned, v1 arbiter) | 60% (53–66) | 5.9% (4–9) | **0.7% (0–3)** | **60/60** | 3 |
| combined (hard + learned, v2 arbiter) | 70% (63–75) | 7.8% (5–12) | 3.0% (2–6) | **60/60** | 5 |

¹ See §5. The corrected column is a post-hoc audit (`scripts/vase_creep_analysis.py`,
`results/v4_v2/vase_creep_correction.json`); the recorded column is the pre-registered metric.

Per-condition success / unsafe (recorded), selected:

| Condition | none | hard | combined_v1 | combined |
|---|---|---|---|---|
| obstacle | 0% / 100% | 83% / 13% | 77% / **0%** | 73% / 13% |
| fragile | 67% / 87% | **97% / 0%** | 30% / 3% | 93% / 0% |
| human | 47% / 77% | 77% / **0%** | 70% / 3% | 80% / 10% |
| sensor_corruption | 0% / 80% | **63%** / 3% | 43% / 0% | 43% / 3% |
| ood_combined (held out) | 0% / 83% | **60%** / 30%→**0%**¹ | 0% / 47%→3%¹ | 0% / 43%→0%¹ |
| unsafe + impossible instructions | 43/60 unsafe | 60/60 refused | 60/60 refused | 60/60 refused |

Paired tests (same seeds): combined vs none — safer on 153 vs 2 episodes (p = 5e-43), more successful
on 78 vs 8 (p = 2e-15). Combined vs hard — hard more successful on 28 vs 1 (p = 1e-7).

Ablations (combined, hazard conditions): removing the detour planner drops success 58% → 25%;
removing speed-and-separation monitoring raises unsafe episodes 14% → 25%; replacing the shield with
stop-only yields 23% success and 77% unsafe. Safety-utility sweep: hard-layer margin ×2.0 reaches 0%
unsafe at 71% success (sweep seeds).

Learned risk model (test transitions): AUROC 0.90, ECE 0.14; conformal tau = 0.885 gives FNR 10% on
calibration but 44% on test — the conformal guarantee did not transfer under the distribution shift
created by the shield itself.

## 4. Findings

1. **The hard layer is the main result.** Relative to the same base policy it raises feasible-task
   success from 36% to 82% and cuts unsafe episodes from 64% to 1.9% (corrected), refuses all 60
   unsafe/impossible instructions, and keeps 60% success in the held-out OOD scene with no corrected
   violations. The detour planner and speed-and-separation monitoring are each necessary (ablations).
2. **The learned layer is a good detector but a poor actuator.** Alone it barely helps (58% unsafe).
   Added to the hard layer it gives a real trade-off rather than a free win: the v1 arbiter is the
   safest method (0.7% corrected) but loses utility (fragile 30%, OOD 0%); the gated v2 arbiter recovers
   fragile utility (93%) but is dominated by hard-only on both axes. In OOD both learned variants stop
   the robot (0% success) because out-of-distribution inputs keep risk above tau.
3. **Instruction-level safety is solved by explicit checks** (100% correct refusals, 3 false refusals).

## 5. Disclosures

- **Vase-creep artefact.** The vase is a tall cylinder on a box table; MuJoCo's cylinder–box contact
  rocks slowly, and on 15/90 vase-bearing test seeds the vase tilts past 0.09 rad with the robot frozen
  (13 OOD, 1 fragile, 1 unsafe-instruction seed). Replaying the recorded hard/combined OOD failures showed no
  contact with the vase at the moment of violation, at identical times across methods. The corrected column
  removes only episodes whose *sole* event is `fragile_disturbed` on such a seed. A stabilised vase
  (box foot) would remove the artefact at the source; it was not re-run for this report.
- **Three evaluation rounds.** v1 (1,080 episodes: hard 79% / 10% unsafe, combined_v1 59% / 9%)
  exposed an OOD layout bug (vase spawned inside the wall), phantom duplicate detections causing false
  refusals, and a clarification oracle that could not disambiguate. These were fixed for v2 (reported
  here), along with the v2 gated arbiter. v1 is kept in `reports/safevla_v4_v1.md`.
- **Arbiter tuning fell back.** No kappa met the pre-declared rule (unsafe ≤ hard's on sweep seeds);
  the fallback (best success, ties → larger kappa) selected kappa = 0.04 (`results/v4_v2/tuned.json`).
- **DAgger promoted despite lower validation success** (92.5% vs BC 97.5%) because it was the
  pre-declared final policy; both are hazard-agnostic.
- **Benchmark scope of v4.** v4 itself is a custom MuJoCo benchmark. The layer was later ported to
  ManiSkill 3 and LIBERO + OpenVLA-7B (`bench/`, `reports/safebench_maniskill.md`); see the top-level README.
- The human is a kinematic forearm proxy; separation thresholds are illustrative, not certified ISO values.
  Language grounding is a keyword grammar, not a pretrained VLM. The CBF runs at 25 Hz on a sphere
  approximation — it is not a formal guarantee.

## 6. Reproducing

Environment: Ubuntu 24.04, Python 3.12 venv, packages pinned in `requirements.txt`
(mujoco 3.3.5, torch 2.8.0, numpy 2.2.6, scipy, opencv-headless, imageio-ffmpeg).

```bash
# one-time (headless GPU server)
bash scripts/remote/setup.sh                      # creates .venv-safevla, installs requirements.txt
MUJOCO_GL=egl python scripts/remote/check_render.py
# full pipeline, detached (survives logout); v2 queue reuses v4 checkpoints
bash scripts/remote/launch.sh "" safevla_v4       # smoke → demos → BC → DAgger → risk data → risk → evaluate → ablate → sweep → report → render → verify
bash scripts/remote/launch.sh "" safevla_v4_v2    # tests → tune → evaluate → render → ablate → sweep → report → verify
bash scripts/remote/status.sh safevla_v4_v2
# single stage
SAFEVLA_TAG=_v2 SAFEVLA_METHODS=none,hard,learned,combined_v1,combined MUJOCO_GL=egl \
  .venv-safevla/bin/python scripts/safevla_v4.py render
# audit
python scripts/vase_creep_analysis.py results/v4_v2
python -m pytest tests/test_safevla_v4.py      # 13 tests
```

Measured on landau (V100): 1080p60 render of all six clips ≈ 17 min.

## 7. Artifacts

| Path | Content |
|---|---|
| `videos/v4_v2/*.mp4` | 1920×1080, 60 fps, H.264 (CRF 16, 2× supersampled): `demo_success`, `unsafe_without_shield`, `prevented_by_safevla` (same seed side by side), `instruction_safety`, `ood_cases`, `failure_cases`; selection rule in `selection_manifest.json` (lowest qualifying seed, failures included) |
| `figures/v4_v2/` | `safety_utility.png`, `conditions.png`, `calibration.png`, `human_separation.png`, `video_stills.jpg` |
| `results/v4_v2/test/episode_results.csv` | every test episode (38 columns); traces/states/sensor videos remain on landau |
| `results/v4_v2/vase_creep_correction.json` | artefact audit and corrected table |
| `reports/safevla_v4_v2.md`, `reports/safevla_v4_v1.md` | auto-generated full tables (v2 final, v1 first round) |

## 8. Possible next steps

- Stabilise the vase (box foot / multi-contact) and re-run the test stage for artefact-free numbers.
- Replace learned SAFE_STOP with a retreat along the barrier gradient near moving hazards; model the held
  object in the shield; recalibrate tau on shielded rollouts (conformal FNR drifted 10% → 44%).
- ~~Port the layer to ManiSkill/LIBERO tasks and a pretrained VLA~~ — done in `bench/` (ManiSkill complete, LIBERO + OpenVLA in progress).
