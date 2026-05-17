#!/bin/zsh
set -euo pipefail

cd /Users/jenriquezafra/Proyectos/Dev/python/IDA-trading
exec /Users/jenriquezafra/Proyectos/Dev/python/IDA-trading/.venv/bin/python -m src.execution.h1c_auto_runner --skip-cboe
