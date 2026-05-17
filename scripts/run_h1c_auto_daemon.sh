#!/bin/zsh
set -euo pipefail

cd /Users/jenriquezafra/Proyectos/Dev/python/IDA-trading
exec /usr/bin/caffeinate -dimsu /Users/jenriquezafra/Proyectos/Dev/python/IDA-trading/.venv/bin/python -u -m src.execution.h1c_auto_daemon
