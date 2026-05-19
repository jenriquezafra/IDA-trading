#!/usr/bin/env bash
set -euo pipefail

display_num="${TRADING_STRATS_DISPLAY:-:99}"
screen_spec="${TRADING_STRATS_SCREEN:-1280x900x24}"
log_dir="${TRADING_STRATS_LOG_DIR:-/opt/trading-strats/logs/ibgateway}"
ibgateway_home="${TRADING_STRATS_IBGATEWAY_HOME:-/opt/trading-strats/ibgateway}"

mkdir -p "${log_dir}"

cleanup() {
  set +e
  for pid in "${ibgateway_pid:-}" "${fluxbox_pid:-}" "${xvfb_pid:-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null
      wait "${pid}" 2>/dev/null
    fi
  done
}
trap cleanup EXIT INT TERM

display_id="${display_num#:}"
if ! pgrep -u "$(id -u)" -f "Xvfb ${display_num}" >/dev/null 2>&1; then
  rm -f "/tmp/.X${display_id}-lock"
fi

/usr/bin/Xvfb "${display_num}" -screen 0 "${screen_spec}" >"${log_dir}/xvfb.log" 2>&1 &
xvfb_pid=$!

export DISPLAY="${display_num}"
sleep 1

/usr/bin/fluxbox >"${log_dir}/fluxbox.log" 2>&1 &
fluxbox_pid=$!

sleep 1

"${ibgateway_home}/ibgateway" >"${log_dir}/ibgateway.log" 2>&1 &
ibgateway_pid=$!

wait "${ibgateway_pid}"
