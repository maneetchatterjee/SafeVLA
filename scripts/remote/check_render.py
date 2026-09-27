"""Headless rendering and GPU check for SafeVLA v4 (run with MUJOCO_GL=egl).

Reports the GL backend and renderer string, times a 1080p frame of the real
Panda scene and a perception frame, and reports CUDA availability for torch.
Exits non-zero if offscreen rendering fails or returns a black image.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np


def main():
    backend = os.environ.get("MUJOCO_GL", "(unset)")
    print("MUJOCO_GL =", backend, "| PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM", "(unset)"))
    from safevla.env import SafePandaEnv
    from safevla.perception import Perception

    env = SafePandaEnv()
    env.reset(0, "ood_combined")
    renderer = mujoco.Renderer(env.model, 1080, 1920)
    renderer.update_scene(env.data, camera="hd_main")
    frame = renderer.render()
    try:
        from OpenGL import GL

        print("GL_RENDERER =", GL.glGetString(GL.GL_RENDERER).decode(), "| GL_VENDOR =", GL.glGetString(GL.GL_VENDOR).decode())
    except Exception as error:  # the string is informative only
        print("GL_RENDERER unavailable:", error)
    start = time.perf_counter()
    for _ in range(10):
        renderer.update_scene(env.data, camera="hd_main")
        frame = renderer.render()
    hd = (time.perf_counter() - start) / 10
    perception = Perception(env)
    start = time.perf_counter()
    for _ in range(20):
        perception.observe(env.data)
    sensor = (time.perf_counter() - start) / 20
    import torch

    print(f"1080p frame: {1000 * hd:.1f} ms | perception observe (2 RGB-D cams): {1000 * sensor:.1f} ms | mean pixel {frame.mean():.1f}")
    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    print("CPU cores:", os.cpu_count())
    if frame.mean() < 5:
        raise SystemExit("rendered frame is black: offscreen GL is not working")
    import cv2

    out = ROOT / "reports/remote_render_check.png"
    cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("wrote", out)


if __name__ == "__main__":
    main()
