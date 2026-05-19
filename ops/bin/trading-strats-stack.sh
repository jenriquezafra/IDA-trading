#!/usr/bin/env bash
set -euo pipefail

app_dir="${TRADING_STRATS_APP_DIR:-/opt/trading-strats/app}"

infrastructure_services=(
  trading-strats-ibgateway.service
  trading-strats-vnc.service
)

app_services=(
  trading-strats-paper.service
  trading-strats-paper-c2.service
  trading-strats-control-center.service
)

services=("${infrastructure_services[@]}" "${app_services[@]}")

timer_units=(
  trading-strats-ibkr-watchdog.timer
)

usage() {
  cat <<'EOF'
Usage: trading-strats-stack.sh {start|restart|full-restart|status|health|logs|incidents}

start    Start all Trading Strats services and timers.
restart  Start IB/VNC if down; restart paper daemons and app.
full-restart Restart every service, including IB Gateway.
status   Show systemd state and listening ports.
health   Run read-only app/IBKR health checks.
logs     Show recent logs for app, daemons, IB Gateway, and watchdog.
incidents Show the latest append-only operational incident events.
EOF
}

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root or through sudo." >&2
    exit 1
  fi
}

start_stack() {
  need_root
  systemctl daemon-reload
  systemctl enable --now "${timer_units[@]}"
  systemctl start "${services[@]}"
}

restart_stack() {
  need_root
  systemctl daemon-reload
  systemctl enable --now "${timer_units[@]}"
  systemctl start "${infrastructure_services[@]}"
  systemctl restart "${app_services[@]}"
}

full_restart_stack() {
  need_root
  systemctl daemon-reload
  systemctl enable --now "${timer_units[@]}"
  systemctl restart "${services[@]}"
}

show_status() {
  systemctl --no-pager --full status "${services[@]}" "${timer_units[@]}" || true
  echo
  ss -ltnp | grep -E ':(4002|5901|8700)\b' || true
  echo
  systemctl list-timers trading-strats-ibkr-watchdog.timer --no-pager || true
}

run_health() {
  local failed=0
  echo "== app health =="
  curl -sS -m 5 http://127.0.0.1:8700/health || failed=1
  echo
  echo "== IBKR watchdog =="
  cd "${app_dir}"
  .venv/bin/python -m src.execution.ibkr_watchdog --config configs/execution/ibkr_watchdog.yaml || failed=1
  echo
  echo "== control center connections =="
  curl -sS -m 10 http://127.0.0.1:8700/control-center/connections || failed=1
  echo
  return "${failed}"
}

show_logs() {
  journalctl \
    -u trading-strats-control-center.service \
    -u trading-strats-paper.service \
    -u trading-strats-paper-c2.service \
    -u trading-strats-ibgateway.service \
    -u trading-strats-ibkr-watchdog.service \
    -n 160 --no-pager -o short-iso
}

show_incidents() {
  local limit="${1:-50}"
  local events_path="${TRADING_STRATS_EVENTS_PATH:-${app_dir}/results/paper/operational_events/events.jsonl}"
  if [[ ! -f "${events_path}" ]]; then
    echo "No operational incidents recorded yet: ${events_path}"
    return 0
  fi
  tail -n "${limit}" "${events_path}"
}

cmd="${1:-}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
case "${cmd}" in
  start)
    start_stack
    show_status
    ;;
  restart)
    restart_stack
    show_status
    ;;
  full-restart)
    full_restart_stack
    show_status
    ;;
  status)
    show_status
    ;;
  health)
    run_health
    ;;
  logs)
    show_logs
    ;;
  incidents)
    show_incidents "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
