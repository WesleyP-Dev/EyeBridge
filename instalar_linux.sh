#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "x86_64" ]]; then
  echo 'Use Debian/Raspberry Pi OS de 64 bits (aarch64 ou x86_64).'
  exit 1
fi
if [[ "${EUID}" -eq 0 ]]; then
  echo 'Execute como usuario normal; o script usa sudo apenas para apt.'
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 fonts-dejavu-core v4l-utils usbutils
if [[ -f /proc/device-tree/model ]] && tr -d '\0' < /proc/device-tree/model | grep -q 'Raspberry Pi'; then
  sudo apt-get install -y python3-picamera2
fi
if [[ ! -x .venv/bin/python ]]; then
  if python3 -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12))' 2>/dev/null; then
    python3 -m venv --system-site-packages .venv
  else
    python3 -m venv .bootstrap
    .bootstrap/bin/python -m pip install --only-binary=:all: uv
    .bootstrap/bin/uv --cache-dir .uv-cache python install 3.11 --install-dir .python
    managed_python="$(find .python -path '*/bin/python3.11' -type f -print -quit)"
    if [[ -z "$managed_python" ]]; then
      echo 'Python 3.11 local nao encontrado.'
      exit 1
    fi
    .bootstrap/bin/uv --cache-dir .uv-cache venv --python "$managed_python" .venv
    .bootstrap/bin/uv --cache-dir .uv-cache pip install --python .venv/bin/python pip
  fi
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use Python 3.11/3.12"'
dependencies=requirements.txt
if [[ "$(uname -m)" == "aarch64" ]]; then
  dependencies=requirements-linux-arm64.lock
fi
.venv/bin/python -m pip install --only-binary=:all: -r "$dependencies"
.venv/bin/python main.py --check
echo 'Pronto. Execute: bash iniciar_linux.sh --config settings.pi5.json'
