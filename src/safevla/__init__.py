"""SafeVLA v4: runtime safety for an articulated MuJoCo Franka Panda.

Simulation only. Modules: scene (MJCF), env (physics + ground-truth safety
monitor), perception (RGB-D point clouds), language (rule-based grounding),
expert/policy (behaviour cloning), shield (hard layer), risk (learned layer),
runtime (arbitration and episode logging), render (HD replay), report.
"""
