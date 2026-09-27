#!/usr/bin/env bash
# Show SafeVLA queue progress: stage states and the tail of the active stage log.
# Usage: bash scripts/remote/status.sh [queue]      queue: safevla_v4 (default) | safevla_v4_v2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
QUEUE="${1:-safevla_v4}" python3 - <<'EOF'
import json, os, pathlib, time
p = pathlib.Path("experiment_tracking") / os.environ["QUEUE"] / "status.json"
if not p.exists():
    raise SystemExit(f"queue {os.environ['QUEUE']} not started")
s = json.loads(p.read_text())
age = time.time() - s.get("heartbeat_unix", 0)
print(f"queue {os.environ['QUEUE']}: {s['state']}  active: {s.get('active_stage')}  heartbeat {age:.0f}s ago")
for name, st in s["stages"].items():
    mins = st.get("elapsed_seconds", 0) / 60
    print(f"  {name:12s} {st['state']:10s} {mins:6.1f} min  rc={st.get('returncode', '-')}")
active = s.get("active_stage")
if active:
    log = pathlib.Path(s["stages"][active]["log"])
    lines = [l for l in log.read_text(errors="ignore").splitlines() if "warning" not in l.lower() and "wrapped" not in l]
    print("\n".join(lines[-5:]))
EOF
