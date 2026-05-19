#!/usr/bin/env bash
set -euo pipefail

label="com.ida-trading.vps-tunnels"
plist_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ops/launchd/${label}.plist"
plist_target="${HOME}/Library/LaunchAgents/${label}.plist"
cmd="${1:-status}"

usage() {
  cat <<'EOF'
Usage: scripts/vps_tunnels.sh {install|start|stop|restart|status|foreground}

install     Install and start a macOS launchd tunnel for app and VNC.
start       Start the installed launchd tunnel.
stop        Stop the installed launchd tunnel.
restart     Restart the installed launchd tunnel.
status      Show launchd tunnel status and local listening ports.
foreground  Open SSH tunnels in the current terminal.
EOF
}

show_ports() {
  echo
  lsof -nP -iTCP:8700 -sTCP:LISTEN || true
  lsof -nP -iTCP:5901 -sTCP:LISTEN || true
}

install_tunnel() {
  mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/Library/Logs"
  cp "${plist_source}" "${plist_target}"
  launchctl bootout "gui/$(id -u)" "${plist_target}" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "${plist_target}"
  launchctl enable "gui/$(id -u)/${label}"
  launchctl kickstart -k "gui/$(id -u)/${label}"
}

start_tunnel() {
  launchctl kickstart -k "gui/$(id -u)/${label}"
}

stop_tunnel() {
  launchctl bootout "gui/$(id -u)" "${plist_target}" >/dev/null 2>&1 || true
}

foreground_tunnel() {
  ssh -i "${IDA_TRADING_VPS_SSH_KEY:-${HOME}/.ssh/trading_strats_hetzner_ed25519}" \
    -N \
    -L 8700:127.0.0.1:8700 \
    -L 5901:127.0.0.1:5901 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "trader@${IDA_TRADING_VPS_HOST:-195.201.229.200}"
}

case "${cmd}" in
  install)
    install_tunnel
    launchctl print "gui/$(id -u)/${label}" || true
    show_ports
    ;;
  start)
    start_tunnel
    show_ports
    ;;
  stop)
    stop_tunnel
    show_ports
    ;;
  restart)
    stop_tunnel
    install_tunnel
    show_ports
    ;;
  status)
    launchctl print "gui/$(id -u)/${label}" || true
    show_ports
    ;;
  foreground)
    foreground_tunnel
    ;;
  *)
    usage
    exit 2
    ;;
esac
