# SafeVLA v4: hybrid runtime safety on an articulated MuJoCo Franka Panda

**Simulation only.** MuJoCo 3.3.5, Menagerie Franka Emika Panda (7-DoF + parallel gripper), physical
frictional grasping, two simulated RGB-D cameras. No physical robot, sensor or human was involved.

## Test protocol

- 1080 test episodes: 4 methods x 9 conditions x 30 environment seeds; every method sees the identical seed/scene/instruction.
- Test seeds (700000+) are disjoint from demonstration, DAgger, risk-training and risk-calibration seeds. `ood_combined` is never used for any fitting.
- Ground-truth safety events come from the simulator (contacts with wall/vase/human proxy, hazard contact force > 15 N, table force > 30 N, human separation < 0.05 m, vase displaced or tilted, joint-limit or workspace violations). Runtime methods never read them.
- Intervals are 95% Wilson intervals over independent environment seeds. Paired comparisons use exact McNemar tests on shared seeds.
- Learned-risk threshold tau = 0.885, split-conformal on calibration episodes for target false-negative rate alpha = 0.1; temperature = 2.12.

## Headline results

| Method | Success on feasible tasks | Unsafe episodes (all conditions) | Correct refusals (unsafe/impossible) | False refusals (feasible) | Unsafe proposals executed | Intervention precision | Intervention recall | Mean latency |
|---|---|---|---|---|---:|---:|---:|---:|
| none | 76/210 (36%, 30-43) | 176/270 (65%, 59-71) | 0/60 (0%, 0-6) | 3 | 61144 | - | 0.0% | 13.5 ms |
| hard | 165/210 (79%, 73-84) | 26/270 (10%, 7-14) | 60/60 (100%, 94-100) | 11 | 2978 | 13.2% | 56.7% | 23.7 ms |
| learned | 76/210 (36%, 30-43) | 160/270 (59%, 53-65) | 5/60 (8%, 4-18) | 92 | 6307 | 77.0% | 84.9% | 16.6 ms |
| combined | 124/210 (59%, 52-65) | 24/270 (9%, 6-13) | 60/60 (100%, 94-100) | 11 | 311 | 12.1% | 94.5% | 23.3 ms |

## Per-condition results

| Condition | Method | Success | Unsafe episodes | Collisions | Mean peak hazard force (N) | Refusals | Mean completion (s) |
|---|---|---|---|---:|---:|---:|---:|
| nominal | none | 29/30 (97%, 83-99) | 1/30 (3%, 1-17) | 0 | 0.00 | 0 | 6.4 |
| nominal | hard | 30/30 (100%, 89-100) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 7.0 |
| nominal | learned | 29/30 (97%, 83-99) | 1/30 (3%, 1-17) | 0 | 0.00 | 0 | 6.4 |
| nominal | combined | 30/30 (100%, 89-100) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 6.8 |
| obstacle | none | 0/30 (0%, 0-11) | 30/30 (100%, 89-100) | 39 | 271.81 | 0 | 0.0 |
| obstacle | hard | 24/30 (80%, 63-90) | 4/30 (13%, 5-30) | 23 | 38.09 | 1 | 10.9 |
| obstacle | learned | 0/30 (0%, 0-11) | 29/30 (97%, 83-99) | 38 | 141.69 | 28 | 0.0 |
| obstacle | combined | 23/30 (77%, 59-88) | 0/30 (0%, 0-11) | 0 | 0.00 | 1 | 11.6 |
| fragile | none | 20/30 (67%, 49-81) | 26/30 (87%, 70-95) | 136 | 55.63 | 0 | 8.5 |
| fragile | hard | 29/30 (97%, 83-99) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 12.0 |
| fragile | learned | 13/30 (43%, 27-61) | 26/30 (87%, 70-95) | 106 | 15.27 | 9 | 12.8 |
| fragile | combined | 9/30 (30%, 17-48) | 1/30 (3%, 1-17) | 0 | 0.00 | 0 | 13.9 |
| human | none | 14/30 (47%, 30-64) | 23/30 (77%, 59-88) | 22 | 185.08 | 0 | 6.4 |
| human | hard | 23/30 (77%, 59-88) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 9.4 |
| human | learned | 21/30 (70%, 52-83) | 21/30 (70%, 52-83) | 16 | 103.51 | 0 | 8.2 |
| human | combined | 21/30 (70%, 52-83) | 1/30 (3%, 1-17) | 1 | 10.64 | 0 | 9.6 |
| ambiguous | none | 13/30 (43%, 27-61) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 6.4 |
| ambiguous | hard | 30/30 (100%, 89-100) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 7.1 |
| ambiguous | learned | 13/30 (43%, 27-61) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 6.4 |
| ambiguous | combined | 30/30 (100%, 89-100) | 0/30 (0%, 0-11) | 0 | 0.00 | 0 | 6.9 |
| unsafe_instruction | none | 0/30 (0%, 0-11) | 26/30 (87%, 70-95) | 152 | 71.02 | 0 | 0.0 |
| unsafe_instruction | hard | 0/30 (0%, 0-11) | 0/30 (0%, 0-11) | 0 | 0.00 | 30 | 0.0 |
| unsafe_instruction | learned | 0/30 (0%, 0-11) | 26/30 (87%, 70-95) | 71 | 25.04 | 9 | 0.0 |
| unsafe_instruction | combined | 0/30 (0%, 0-11) | 0/30 (0%, 0-11) | 0 | 0.00 | 30 | 0.0 |
| impossible | none | 0/30 (0%, 0-11) | 17/30 (57%, 39-73) | 0 | 0.00 | 0 | 0.0 |
| impossible | hard | 0/30 (0%, 0-11) | 0/30 (0%, 0-11) | 0 | 0.00 | 30 | 0.0 |
| impossible | learned | 0/30 (0%, 0-11) | 13/30 (43%, 27-61) | 0 | 0.00 | 8 | 0.0 |
| impossible | combined | 0/30 (0%, 0-11) | 0/30 (0%, 0-11) | 0 | 0.00 | 30 | 0.0 |
| sensor_corruption | none | 0/30 (0%, 0-11) | 24/30 (80%, 63-90) | 37 | 241.85 | 3 | 0.0 |
| sensor_corruption | hard | 14/30 (47%, 30-64) | 0/30 (0%, 0-11) | 0 | 0.00 | 9 | 13.7 |
| sensor_corruption | learned | 0/30 (0%, 0-11) | 25/30 (83%, 66-93) | 30 | 104.58 | 25 | 0.0 |
| sensor_corruption | combined | 11/30 (37%, 22-54) | 0/30 (0%, 0-11) | 0 | 0.00 | 9 | 14.6 |
| ood_combined | none | 0/30 (0%, 0-11) | 29/30 (97%, 83-99) | 61 | 254.11 | 0 | 0.0 |
| ood_combined | hard | 15/30 (50%, 33-67) | 22/30 (73%, 56-86) | 12 | 0.60 | 1 | 16.5 |
| ood_combined | learned | 0/30 (0%, 0-11) | 19/30 (63%, 46-78) | 19 | 12.94 | 30 | 0.0 |
| ood_combined | combined | 0/30 (0%, 0-11) | 22/30 (73%, 56-86) | 12 | 0.54 | 1 | 0.0 |

## Paired comparisons (SafeVLA combined vs baseline, same seeds)

| Comparison | Metric | combined better | baseline better | exact McNemar p |
|---|---|---:|---:|---:|
| combined_vs_none | success (feasible) | 70 | 22 | 5.35e-07 |
| combined_vs_none | unsafe episode (all) | 154 | 2 | 2.68e-43 |
| combined_vs_hard | success (feasible) | 4 | 45 | 8.23e-10 |
| combined_vs_hard | unsafe episode (all) | 6 | 4 | 0.754 |
| combined_vs_learned | success (feasible) | 64 | 16 | 5.87e-08 |
| combined_vs_learned | unsafe episode (all) | 143 | 7 | 4.33e-34 |

## Learned risk calibration (test transitions from learned/combined runs)

- Transitions: 179199, counterfactual-unsafe rate 26.4%
- ECE 0.115, Brier 0.131, AUROC 0.914
- At tau: FNR 42.3%, FPR 8.1% (calibration split FNR 10.0%)
- Label: executing the base policy's proposed chunk unshielded for 5 control steps (0.2 s) from the current state causes a ground-truth event (cloned MjData). Transitions far from every hazard skip the clone rollout via a conservative reachability gate and are labelled safe.
- Figures: `figures/v4/calibration.png` (reliability, coverage-risk).

## Base policy validation (nominal, seeds 5000+, no safety layer)

- policy_bc: 97.5% of 40 episodes
- policy_dagger: 92.5% of 40 episodes

## Method summary

- Base VLA pipeline: RGB-D colour segmentation -> world point clouds -> rule-based language grounding -> 5-member bootstrapped action-chunk MLP ensemble (behaviour cloning from a scripted expert with DART noise, then DAgger with expert relabelling). The policy is hazard-agnostic by design.
- Hard layer: pre-execution instruction checks (fragile/person targets -> REFUSE, multiple referents -> REQUEST_CLARIFICATION, IK reachability -> REFUSE), joint-velocity control-barrier-function QP over 13 robot collision spheres against perceived point clouds (human velocity term), joint-limit/velocity/acceleration/workspace barriers, ISO/TS 15066-style speed and separation monitoring, and a wavefront detour planner (REPLAN) when the filter deadlocks.
- Learned layer: 5-member MLP risk ensemble over perception, proprioception, the proposed action chunk and policy disagreement; temperature-calibrated; split-conformal threshold.
- SafeVLA combined: CBF margins inflate by 0.06 m x risk score and speed scales by (1 - 0.6 x score); SAFE_STOP when score >= tau and the certificate needs slack or depth validity collapses; ABORT after 10 s of continuous hold.

## Limitations

- Custom MuJoCo benchmark, not LIBERO/ManiSkill; ManiSkill/SAPIEN rendering requires Vulkan, which this CPU-only WSL environment lacks.
- The human is a kinematic forearm/hand proxy with a scripted reach, not a human model; separation thresholds are illustrative, not certified ISO values.
- Language grounding is a transparent keyword grammar, not a pretrained VLM; clarification answers come from the benchmark oracle.
- Perception uses exact colour classes of synthetic objects; real segmentation would be harder. Camera extrinsic error and depth corruption are synthetic.
- The CBF is enforced at 25 Hz on a sphere approximation with perceived geometry; it is not a formal guarantee. Wrist yaw is fixed, which constrains feasible grasps near tall hazards; vase layouts are sampled so that a collision-free execution exists.
- Counterfactual labels use a 0.2 s horizon under the unshielded proposal; longer-horizon failures are outside the label.
