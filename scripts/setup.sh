#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .cache/pip .deps/packages
export PIP_CACHE_DIR="$PWD/.cache/pip"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version > /dev/null 2>&1; then
  python3 - <<'PY'
from pathlib import Path
import urllib.request
Path('scripts/get-pip.py').write_bytes(urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', timeout=60).read())
PY
  .venv/bin/python scripts/get-pip.py
fi
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
# sd1 Ubuntu 22.04 / Python 3.10용 헤더. 시스템에는 설치하지 않고 내부에 풀기만 한다.
if [[ ! -f .deps/usr/include/python3.10/Python.h ]]; then
  python3 - <<'PY'
from pathlib import Path
import urllib.request
name='libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb'
destination=Path('.deps/packages') / name
url='https://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/' + name
destination.write_bytes(urllib.request.urlopen(url, timeout=120).read())
PY
  dpkg-deb -x .deps/packages/libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb .deps
fi
.venv/bin/python -m gaim.data --output-dir data/medqa --train-size 100 --dev-size 20 --test-size 100 --seed 20260921
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/verify_model.py
.venv/bin/python scripts/preflight.py
.venv/bin/python -m pytest -q
echo '준비 완료: bash scripts/run_smoke.sh my_first_run'
