"""OpenVLA-on-LIBERO glue, vendored from openvla/openvla (MIT) experiments/robot/*.

Copied instead of imported because those modules pull in tensorflow_datasets, whose
protobuf requirement conflicts with TensorFlow 2.15. Behaviour matches the official
LIBERO evaluation: 180-degree-rotated agentview, JPEG round-trip + lanczos3 resize to
224, 0.9-area center crop, 10 no-op settle steps, binarised + inverted gripper.
Only deviation: float16 weights/inputs (V100 has no bfloat16).
"""

import os
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

tf.config.set_visible_devices([], "GPU")  # TF only resizes images
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10


def get_libero_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # affects object positions even with a fixed init state
    return env, task.language


def dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, size=(224, 224)):
    img = tf.image.encode_jpeg(img)  # the RLDS training data went through JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def libero_image(obs, size=224):
    return resize_image(obs["agentview_image"][::-1, ::-1], (size, size))


def center_crop(image, crop_scale=0.9):
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)[None]
    side = float(np.clip(np.sqrt(crop_scale), 0, 1))
    off = (1 - side) / 2
    x = tf.image.crop_and_resize(x, [[off, off, off + side, off + side]], [0], (224, 224))[0]
    return tf.image.convert_image_dtype(tf.clip_by_value(x, 0, 1), tf.uint8, saturate=True).numpy()


def postprocess(action):
    """[0,1] gripper -> [-1,1], binarise, invert (OpenVLA: 1 = close; LIBERO: -1 = open)."""
    action = np.array(action, dtype=np.float64)
    action[-1] = -np.sign(2 * action[-1] - 1)
    return action


class OpenVLAPolicy:
    def __init__(self, suite="libero_spatial", device="cuda:0", dtype=torch.float16):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.ckpt = f"openvla/openvla-7b-finetuned-{suite.replace('_', '-')}"
        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(self.ckpt, trust_remote_code=True)
        self.attn, errors = None, []
        for attn in ["sdpa", "eager"]:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(self.ckpt, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation=attn).to(device)
                self.attn = attn
                break
            except Exception as exc:  # older remote code may not declare sdpa support
                errors.append(f"{attn}: {type(exc).__name__}: {exc}")
        if self.attn is None:
            raise RuntimeError("OpenVLA failed to load (CUDA_VISIBLE_DEVICES=%s): %s" % (os.environ.get("CUDA_VISIBLE_DEVICES"), " | ".join(errors)))
        self.unnorm_key = suite if suite in self.model.norm_stats else f"{suite}_no_noops"

    def __call__(self, obs, instruction):
        image = Image.fromarray(center_crop(libero_image(obs))).convert("RGB")
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            raw = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return raw, postprocess(raw)
