"""네 GPU 평가 실행, 고정 manifest, 중단 후 이어 실행을 관리한다."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(cfg: dict) -> None:
    if cfg["run_kind"] not in ("pilot", "evaluation"):
        raise ValueError("run_kind must be pilot or evaluation")
    if cfg["split"] not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    if cfg["split"] == "test" and cfg["run_kind"] != "evaluation":
        raise ValueError("Test split is reserved for evaluation")
    if not isinstance(cfg["k"], int) or not 1 <= cfg["k"] <= 20:
        raise ValueError("k must be an integer in 1..20")
    if not isinstance(cfg["limit"], int) or cfg["limit"] < 4:
        raise ValueError("Use at least four questions, one per GPU")
    if not cfg["categories"] or set(cfg["categories"]) - {"nonclinical_background", "answer_pressure"}:
        raise ValueError("Unknown perturbation category")
    if cfg["target_temperature"] != 0:
        raise ValueError("This protocol uses a deterministic target (temperature=0)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Override question count, e.g. 4 for smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    from .environment import configure
    configure()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = json.loads(args.config.read_text())
    if args.limit is not None:
        cfg["limit"] = args.limit
    validate_config(cfg)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This run is already running; refusing duplicate GPU jobs")
    # 서로 다른 run도 이 프로젝트의 네 GPU를 동시에 중복 점유하지 않는다.
    gpu_lock = (ROOT / ".gpu.lock").open("w")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another GAIM run is using GPUs 4,5,6,7")
    data_dir = ROOT / cfg["data_dir"]
    dataset_manifest = data_dir / "manifest.json"
    items = [json.loads(line) for line in (data_dir / f"{cfg['split']}.jsonl").read_text().splitlines() if line.strip()]
    if len(items) < cfg["limit"]:
        raise ValueError(f"Only {len(items)} questions available; requested {cfg['limit']}")
    items = items[:cfg["limit"]]
    # 모델의 답을 보기 전에 문항 목록을 고정한다. 정답 여부로 표본을 교체하지 않는다.
    ids = [item["id"] for item in items]
    code_files = ["backend.py", "worker.py", "perturbations.py", "run.py", "environment.py"]
    code_hashes = {name: sha(ROOT / "gaim" / name) for name in code_files}
    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("Run exists. Choose a new directory or pass --resume.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != cfg or manifest["question_ids"] != ids:
            raise ValueError("Config/questions differ from saved run; use a new run directory")
        if manifest["dataset_manifest_sha256"] != sha(dataset_manifest) or manifest["code_sha256"] != code_hashes:
            raise ValueError("Dataset or execution code changed; use a new run directory")
        saved_items = [json.loads(line) for line in (run_dir / "questions.jsonl").read_text().splitlines() if line.strip()]
        if saved_items != items:
            raise ValueError("Saved questions were modified; refusing resume")
    else:
        if args.resume:
            raise ValueError("No run manifest exists to resume")
        from huggingface_hub import HfApi
        revision = HfApi().model_info(cfg["model"], revision=cfg["revision"]).sha
        manifest = {
            "protocol_version": 1, "run_kind": cfg["run_kind"], "config": cfg, "k": cfg["k"],
            "question_ids": ids, "dataset_manifest_sha256": sha(dataset_manifest),
            "dataset_manifest": json.loads(dataset_manifest.read_text()),
            "code_sha256": code_hashes, "target_model": cfg["model"], "target_revision": revision,
            "attacker_model": cfg["model"], "attacker_revision": revision,
            "shared_checkpoint_for_roles": True, "physical_gpus": [4, 5, 6, 7],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validity_policy": "Syntactic checks are provisional; human review required for confirmed flips.",
            "invalid_candidate_policy": "One generator call; consume target call with clean input; never count as attack success.",
        }
        (run_dir / "questions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    from huggingface_hub import snapshot_download
    print(f"Preparing {cfg['model']} @ {manifest['target_revision']}", flush=True)
    snapshot_download(repo_id=cfg["model"], revision=manifest["target_revision"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"], max_workers=4)
    if args.download_only:
        print("Downloaded. Re-run with --resume and without --download-only.")
        return
    gpu_info = subprocess.check_output([
        "nvidia-smi", "--id=4,5,6,7", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"
    ], text=True)
    (run_dir / "gpu_before.txt").write_text(gpu_info)
    print(gpu_info, flush=True)
    for line in gpu_info.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        used = int(fields[3].split()[0])
        if used > 2048:
            raise SystemExit(f"GPU {fields[0]} already has {used} MiB allocated. Left other processes untouched.")
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    environment_path = run_dir / "environment.txt"
    if environment_path.exists() and environment_path.read_text() != environment:
        raise ValueError("Python dependencies changed since run creation; use a new run directory")
    environment_path.write_text(environment)
    children, handles = [], []
    try:
        for worker in range(4):
            handle = (run_dir / f"worker_{worker}.log").open("a")
            handles.append(handle)
            child = subprocess.Popen([sys.executable, "-u", "-m", "gaim.worker", "--run-dir", str(run_dir), "--worker", str(worker)],
                                     stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            children.append(child)
        (run_dir / "processes.json").write_text(json.dumps({"launcher": os.getpid(), "workers": [p.pid for p in children]}, indent=2))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker failed with exit code {failed}; see worker logs")
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Worker failed; see worker logs")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for handle in handles:
            handle.close()
    subprocess.run([sys.executable, "-m", "gaim.metrics", "--run-dir", str(run_dir), "--k", str(cfg["k"])], check=True)
    (run_dir / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"Complete: {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
