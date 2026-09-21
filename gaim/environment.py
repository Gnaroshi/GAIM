"""서버의 전역 환경을 바꾸지 않고 프로젝트 내부 의존성 경로를 사용한다."""
from pathlib import Path
import os
import re
from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSICAL_GPUS = (4, 5, 6, 7)


def parse_gpu_mask(value: str) -> tuple[int, ...]:
    """CUDA의 논리 번호 순서를 보존하며 실제 GPU 네 장을 검증한다."""
    if not isinstance(value, str) or not re.fullmatch(r"\s*[0-9]+\s*(,\s*[0-9]+\s*){3}", value):
        raise ValueError("GPU mask must contain exactly four numeric physical IDs, e.g. 0,1,2,3")
    devices = tuple(int(part.strip()) for part in value.split(","))
    if len(set(devices)) != 4:
        raise ValueError("GPU mask must contain four unique physical IDs")
    return devices


def physical_gpus(environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """명시적 GAIM 설정, 기존 CUDA 설정, 원 실행 설정 순서로 선택한다."""
    env = os.environ if environ is None else environ
    raw = env.get("GAIM_CUDA_DEVICES", env.get("CUDA_VISIBLE_DEVICES", gpu_mask(DEFAULT_PHYSICAL_GPUS)))
    selected = parse_gpu_mask(raw)
    # launcher가 정한 순서와 child의 새 환경이 다르면 모델을 올리기 전에 중단한다.
    expected = env.get("GAIM_EXPECTED_CUDA_DEVICES")
    if expected is not None and selected != parse_gpu_mask(expected):
        raise ValueError("GPU mapping differs from the launcher; start a new run with a consistent environment")
    return selected


def gpu_mask(devices: Sequence[int]) -> str:
    return ",".join(str(device) for device in devices)


def require_gpu_mapping(expected: Sequence[int], *, environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    selected = physical_gpus(environ)
    if (len(expected) != 4 or any(type(device) is not int for device in expected)
            or tuple(expected) != selected):
        raise ValueError("GPU mapping differs from the saved manifest; use the original mapping or a new run")
    return selected


def child_environment(devices: Sequence[int]) -> dict[str, str]:
    """자식 프로세스에도 네 물리 GPU와 순서를 동일하게 전달한다."""
    require_gpu_mapping(devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_mask(devices)
    env["GAIM_EXPECTED_CUDA_DEVICES"] = gpu_mask(devices)
    return env


def configure() -> tuple[int, ...]:
    devices = physical_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_mask(devices)
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
    return devices
