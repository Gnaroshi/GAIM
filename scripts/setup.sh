#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 알려진 환경만 설치한다. 다른 플랫폼에 Ubuntu 헤더를 억지로 적용하지 않는다.
if [[ "${1:-}" != "" && "${1:-}" != "--check" ]]; then
  echo 'Usage: bash scripts/setup.sh [--check]' >&2
  exit 2
fi
python3 - <<'PY'
import os, platform, shutil, subprocess, sys
from pathlib import Path
if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 10):
    raise SystemExit('지원 환경: Ubuntu 22.04 x86_64 / python3=Python 3.10. GPU 없는 기록 검증은 python3 scripts/verify_reference.py를 사용하세요.')
release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
if release.get('ID', '').strip('"') != 'ubuntu' or release.get('VERSION_ID', '').strip('"') != '22.04':
    raise SystemExit('이 설치는 Ubuntu 22.04에서 검증되었습니다. 다른 OS에서는 자동 설치하지 않습니다.')
for name in ('nvidia-smi', 'dpkg-deb', 'gcc'):
    if not shutil.which(name):
        raise SystemExit(f'필수 도구가 없습니다: {name}. 시스템 설치는 서버 관리 절차를 따르세요.')
from gaim.environment import configure
configure()
mask = os.environ['CUDA_VISIBLE_DEVICES']
info = subprocess.check_output(['nvidia-smi', '--id='+mask, '--query-gpu=index,memory.used', '--format=csv,noheader'], text=True)
if len(info.strip().splitlines()) != 4 or any(int(line.split(',')[1].strip().split()[0]) > 2048 for line in info.strip().splitlines()):
    raise SystemExit('선택한 GPU 네 장이 모두 비어 있지 않습니다. 다른 작업은 중단하지 않았습니다.')
print('환경 사전 검사 통과. 선택한 GPU:', mask)
PY
if [[ "${1:-}" == "--check" ]]; then
  exit 0
fi
mkdir -p logs .cache/pip .deps/packages
export PIP_CACHE_DIR="$PWD/.cache/pip"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version > /dev/null 2>&1; then
  python3 - <<'PY'
from pathlib import Path
import hashlib
import urllib.request
payload=urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', timeout=60).read()
if hashlib.sha256(payload).hexdigest() != 'fb24e693bab954209a063d90953621412ccad4a500905a726286e038f508ddf6':
    raise SystemExit('공식 get-pip.py 내용이 관측 버전과 달라졌습니다. 실행 전에 bootstrap 출처를 확인하세요.')
Path('scripts/get-pip.py').write_bytes(payload)
PY
  .venv/bin/python scripts/get-pip.py
fi
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt -c docs/results/reference/environment.txt
# 원래 실행과 같은 Ubuntu 헤더가 필요하면 시스템을 바꾸지 않고 로컬에 추출한다.
if [[ ! -f .deps/usr/include/python3.10/Python.h ]] && ! .venv/bin/python -c 'from pathlib import Path; import sysconfig; raise SystemExit(not (Path(sysconfig.get_path("include"))/"Python.h").exists())'; then
  python3 - <<'PY'
from pathlib import Path
import hashlib
import urllib.request
name='libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb'
destination=Path('.deps/packages') / name
url='https://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/' + name
payload=urllib.request.urlopen(url, timeout=120).read()
if hashlib.sha256(payload).hexdigest() != 'c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9':
    raise SystemExit('Python 헤더 패키지 checksum 불일치. 압축을 풀지 않습니다.')
destination.write_bytes(payload)
PY
  dpkg-deb -x .deps/packages/libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb .deps
fi
.venv/bin/python -m gaim.data --output-dir data/medqa --train-size 100 --dev-size 20 --test-size 100 --seed 20260921
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/verify_model.py
.venv/bin/python scripts/preflight.py
.venv/bin/python -m pytest -q
echo '준비 완료: bash scripts/run_smoke.sh my_first_run'
