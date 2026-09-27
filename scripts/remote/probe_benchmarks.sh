#!/usr/bin/env bash
# Read-only feasibility probe for ManiSkill 3 + LIBERO/OpenVLA on this server. Installs nothing.
# Usage (from the repository root):  bash scripts/remote/probe_benchmarks.sh 2>&1 | tee probe_benchmarks.log
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "== host";  hostname; uname -r; nproc; free -g | head -2
echo "== disk (repo, home, /tmp)"; df -h . ~ /tmp 2>/dev/null | awk 'NR==1||!seen[$1]++'
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | head -5
echo "== Vulkan (ManiSkill/SAPIEN rendering needs an NVIDIA Vulkan ICD)"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | grep -E "deviceName|driverVersion|apiVersion|ERROR" | head -6 || echo "vulkaninfo not installed"
ldconfig -p 2>/dev/null | grep -E "libvulkan.so|libGLX_nvidia|libEGL_nvidia" | head -5
echo "== EGL (robosuite/LIBERO rendering)"; ls /usr/share/glvnd/egl_vendor.d 2>&1
echo "== python / conda"
command -v conda && conda --version && conda env list 2>/dev/null | head -10
for p in python3.10 python3.11 python3.12 python3; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done
echo "== sudo without password?"; sudo -n true 2>/dev/null && echo yes || echo no
echo "== network"
for u in https://huggingface.co https://pypi.org https://github.com https://download.pytorch.org; do
  printf "%-32s " "$u"; curl -s -o /dev/null -m 10 -w "%{http_code}\n" "$u" || echo "unreachable"
done
echo "== HF cache"; du -sh ~/.cache/huggingface 2>/dev/null || echo "none"; echo "HF_HOME=${HF_HOME:-unset}"
echo "== done"
