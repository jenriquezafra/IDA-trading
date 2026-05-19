#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "== VPS stack: restart =="
scripts/vps_stack.sh restart

echo
echo "== Local tunnels: restart =="
scripts/vps_tunnels.sh restart

echo
echo "== VPS stack: health =="
if ! scripts/vps_stack.sh health; then
  cat >&2 <<'EOF'

VPS stack is running, but health is degraded.
If IBKR shows ConnectionRefusedError on 127.0.0.1:4002, open VNC and finish the IB Gateway login:
  vnc://127.0.0.1:5901

Saved incidents:
  scripts/vps_stack.sh incidents 50

EOF
  exit 1
fi

echo
echo "== Local app tunnel: health =="
for attempt in {1..10}; do
  if curl -fsS -m 3 http://127.0.0.1:8700/health >/dev/null; then
    echo "local app tunnel OK"
    break
  fi
  if [[ "${attempt}" -eq 10 ]]; then
    echo "local app tunnel failed after ${attempt} attempts" >&2
    exit 1
  fi
  sleep 1
done

cat <<'EOF'

READY
App: http://127.0.0.1:8700
VNC: vnc://127.0.0.1:5901

EOF
