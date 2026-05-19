#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

vps_host="${IDA_TRADING_VPS_HOST:-195.201.229.200}"
vps_user="${IDA_TRADING_VPS_ROOT_USER:-root}"
ssh_key="${IDA_TRADING_VPS_SSH_KEY:-${HOME}/.ssh/trading_strats_hetzner_ed25519}"
remote_app_dir="${IDA_TRADING_VPS_APP_DIR:-/opt/trading-strats/app}"
remote_runtime_dir="${IDA_TRADING_VPS_RUNTIME_DIR:-/opt/trading-strats/runtime}"
remote_bin_dir="${IDA_TRADING_VPS_BIN_DIR:-/opt/trading-strats/bin}"
remote_stack_script="${remote_bin_dir}/trading-strats-stack.sh"

if [[ ! -f "${ssh_key}" ]]; then
  echo "SSH key not found: ${ssh_key}" >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for deploy" >&2
  exit 1
fi

ssh_opts=(
  -i "${ssh_key}"
  -o BatchMode=yes
  -o ConnectTimeout=8
)

echo "== Sync code to VPS =="
sync_sources=(configs ops scripts src tests requirements.txt)
for optional_file in pyproject.toml pytest.ini setup.cfg Makefile; do
  if [[ -e "${optional_file}" ]]; then
    sync_sources+=("${optional_file}")
  fi
done
rsync -az --checksum \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.DS_Store' \
  --exclude 'data/' \
  --exclude 'models/' \
  --exclude 'outputs/' \
  --exclude 'reports/' \
  --exclude 'results/' \
  -e "ssh ${ssh_opts[*]}" \
  "${sync_sources[@]}" \
  "${vps_user}@${vps_host}:${remote_app_dir}/"

echo
echo "== Install VPS service files =="
ssh "${ssh_opts[@]}" "${vps_user}@${vps_host}" \
  "install -d '${remote_bin_dir}' '${remote_runtime_dir}' && \
   install -m 0755 '${remote_app_dir}/ops/bin/trading-strats-stack.sh' '${remote_stack_script}' && \
   install -m 0755 '${remote_app_dir}/ops/bin/trading-strats-ibgateway-session.sh' '${remote_bin_dir}/trading-strats-ibgateway-session.sh' && \
   install -m 0644 '${remote_app_dir}/ops/runtime/h1c_auto_daemon.yaml' '${remote_runtime_dir}/h1c_auto_daemon.yaml' && \
   install -m 0644 '${remote_app_dir}/ops/runtime/c2_auto_daemon.yaml' '${remote_runtime_dir}/c2_auto_daemon.yaml' && \
   install -m 0644 '${remote_app_dir}/ops/runtime/h1c_auto_runner.paper.yaml' '${remote_runtime_dir}/h1c_auto_runner.paper.yaml' && \
   install -m 0644 '${remote_app_dir}/ops/runtime/c2_auto_runner.paper.yaml' '${remote_runtime_dir}/c2_auto_runner.paper.yaml' && \
   install -m 0644 '${remote_app_dir}/ops/systemd/'*.service /etc/systemd/system/ && \
   install -m 0644 '${remote_app_dir}/ops/systemd/'*.timer /etc/systemd/system/"

echo
echo "== Restart VPS stack =="
scripts/vps_stack.sh restart

echo
echo "== VPS health =="
health_status=0
scripts/vps_stack.sh health || health_status="$?"

if [[ "${health_status}" -ne 0 ]]; then
  cat <<'EOF'

DEPLOYED, BUT VPS HEALTH IS DEGRADED
The code and service files were copied to the VPS, but the stack is not fully healthy.
If IBKR shows ConnectionRefusedError on 127.0.0.1:4002, open VNC and finish the IB Gateway login:
  vnc://127.0.0.1:5901

Inspect saved incidents with:
  scripts/vps_stack.sh incidents 50

EOF
  exit "${health_status}"
fi

cat <<'EOF'

DEPLOYED
Start/recover everything later with:
  scripts/ida_up.sh

Inspect saved incidents with:
  scripts/vps_stack.sh incidents 50

EOF
