"""서버의 전역 환경을 바꾸지 않고 프로젝트 내부 의존성 경로를 사용한다."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def configure() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["TRITON_CACHE_DIR"] = str(ROOT / ".cache/triton")
    os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # sd1의 system Python에는 개발용 헤더가 없어 로컬로 추출한 Ubuntu 헤더를 제공한다.
    headers = ROOT / ".deps/usr/include"
    if headers.exists():
        paths = [str(headers), str(headers / "python3.10"), str(headers / "x86_64-linux-gnu/python3.10")]
        inherited = os.environ.get("CPATH")
        if inherited:
            paths.append(inherited)
        os.environ["CPATH"] = ":".join(dict.fromkeys(":".join(paths).split(":")))
