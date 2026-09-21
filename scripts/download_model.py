"""설정에 고정된 공개 모델만 프로젝트 캐시에 내려받는다."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaim.environment import configure, ROOT

configure()
from huggingface_hub import snapshot_download

config = json.loads((ROOT / "configs/pilot.json").read_text())
print(snapshot_download(repo_id=config["model"], revision=config["revision"],
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"], max_workers=4))
