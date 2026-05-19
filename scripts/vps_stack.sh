#!/usr/bin/env bash
set -euo pipefail

vps_host="${IDA_TRADING_VPS_HOST:-195.201.229.200}"
vps_user="${IDA_TRADING_VPS_ROOT_USER:-root}"
ssh_key="${IDA_TRADING_VPS_SSH_KEY:-${HOME}/.ssh/trading_strats_hetzner_ed25519}"
remote_script="${IDA_TRADING_VPS_STACK_SCRIPT:-/opt/trading-strats/bin/trading-strats-stack.sh}"
if [[ "$#" -gt 0 ]]; then
  cmd="$1"
  shift
else
  cmd="status"
fi

if [[ ! -f "${ssh_key}" ]]; then
  echo "SSH key not found: ${ssh_key}" >&2
  exit 1
fi

ssh -i "${ssh_key}" \
  -o BatchMode=yes \
  -o ConnectTimeout=8 \
  "${vps_user}@${vps_host}" \
  "${remote_script}" "${cmd}" "$@"
