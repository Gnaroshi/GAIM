"""내려받은 가중치의 SHA-256을 Hugging Face 캐시의 원본 해시와 대조한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaim.environment import configure, ROOT
configure()
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--wait-seconds", type=int, default=0)
args = parser.parse_args()
cfg = json.loads((ROOT / "configs/pilot.json").read_text())
deadline = time.monotonic() + args.wait_seconds
while True:
    path = Path(snapshot_download(repo_id=cfg["model"], revision=cfg["revision"], local_files_only=True))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    filenames = sorted(set(index["weight_map"].values()))
    if all((path / name).exists() for name in filenames):
        break
    if time.monotonic() >= deadline:
        raise SystemExit("Weight download is not complete")
    time.sleep(5)
checks = []
for name in filenames:
    file = path / name
    value = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    expected = file.resolve().name
    actual = value.hexdigest()
    if actual != expected:
        raise ValueError(f"Weight checksum mismatch: {name}. Preserve logs and download a fresh copy.")
    checks.append({"file": name, "size_bytes": file.stat().st_size, "sha256": actual, "verified": True})
    print(f"Verified {name}", flush=True)
(ROOT / "model-verification.json").write_text(json.dumps({"model": cfg["model"], "revision": cfg["revision"], "files": checks}, indent=2))
