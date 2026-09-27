"""Single-worker Ubuntu experiment queue with durable status and process monitoring."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_tracking/ubuntu_pipeline.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    raw = config_path.read_bytes()
    config = json.loads(raw)
    out = ROOT / "experiment_tracking" / config["id"]
    out.mkdir(exist_ok=True)
    lock = (out / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This queue already has an active worker")
    status_path = out / "status.json"
    digest = hashlib.sha256(raw).hexdigest()
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {
            "config_sha256": digest,
            "stages": {},
            "created_unix": time.time(),
        }
    )
    if status["config_sha256"] != digest:
        raise SystemExit("Configuration changed; use a new pipeline id")
    status.update(
        worker_pid=os.getpid(), state="running", full_research_scope_complete=False
    )
    # Dependencies are re-evaluated in this attempt; old blocked labels are stale.
    for record in status["stages"].values():
        if record.get("state") == "blocked":
            record["state"] = "pending"
    child = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        status.update(state="interrupted", heartbeat_unix=time.time())
        atomic_json(status_path, status)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for stage in config["stages"]:
        name = stage["id"]
        previous = status["stages"].get(name, {})
        if previous.get("state") == "complete":
            continue
        missing = [
            dependency
            for dependency in stage.get("depends_on", [])
            if status["stages"].get(dependency, {}).get("state") != "complete"
        ]
        if missing:
            status["stages"][name] = {"state": "blocked", "dependencies": missing}
            atomic_json(status_path, status)
            continue
        attempt = previous.get("attempts", 0) + 1
        log = out / f"{name}.attempt{attempt}.log"
        # A stage may run in another interpreter (e.g. a benchmark's conda env): "python": "~/maneet/bench/env-maniskill/bin/python"
        command = [os.path.expanduser(os.path.expandvars(stage.get("python", sys.executable))), *stage["args"]]
        record = {
            "state": "running",
            "attempts": attempt,
            "started_unix": time.time(),
            "command": command,
            "log": str(log.relative_to(ROOT)),
        }
        status["stages"][name] = record
        sources = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ["src/safevla", "bench", "bench/safebench", "scripts"]
            for p in (ROOT / folder).glob("*.py")
        }
        atomic_json(out / f"{name}.attempt{attempt}.sources.json", sources)
        environment = os.environ.copy()
        environment.update(
            MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
            PYOPENGL_PLATFORM=os.environ.get("PYOPENGL_PLATFORM", os.environ.get("MUJOCO_GL", "egl")),
            ROBOTICS_CAMERA="perspective",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        environment.update(stage.get("env", {}))
        with log.open("w") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = child.pid
            status["active_stage"] = name
            while child.poll() is None:
                status["heartbeat_unix"] = time.time()
                record["elapsed_seconds"] = time.time() - record["started_unix"]
                record["log_bytes"] = log.stat().st_size
                atomic_json(status_path, status)
                if record["elapsed_seconds"] > stage.get("timeout_hours", 24) * 3600:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    record["timeout"] = True
                    break
                time.sleep(15)
            record.update(returncode=child.wait(), finished_unix=time.time())
        outputs = {path: (ROOT / path).exists() for path in stage.get("outputs", [])}
        record["outputs_exist"] = outputs
        record["state"] = (
            "complete"
            if record["returncode"] == 0 and all(outputs.values())
            else "failed"
        )
        atomic_json(status_path, status)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": name, **record}) + "\n")
    status.update(
        state="finished"
        if all(r["state"] == "complete" for r in status["stages"].values())
        else "finished_with_failures",
        active_stage=None,
        heartbeat_unix=time.time(),
    )
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
