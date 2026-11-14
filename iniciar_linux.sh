#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Execute bash instalar_linux.sh primeiro.'
  exit 1
fi
export PYTHONUTF8=1
exec .venv/bin/python main.py "$@"
