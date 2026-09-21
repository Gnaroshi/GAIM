"""학습하지 않은 고정 평가 입력을 네 adapter에 동일하게 재생하여 비교한다."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .backend import LocalModel, call_seed
from .perturbations import score_answer
from .worker import append_record, read_jsonl
from .metrics import review_content_hash
from .environment import configure, gpu_mask, require_gpu_mapping, child_environment

VARIANTS = ("clean", "random", "independent", "adaptive")
ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_records(source: Path) -> list[dict]:
    records = []
    seen = set()
    source_manifest = json.loads((source / "run.json").read_text())
    expected_ids = set(source_manifest["question_ids"])
    reviews = {}
    review_path = source / "review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["question_id"], row["condition"], int(row["attempt"])
                if key in reviews:
                    raise ValueError("Duplicate review decisions")
                reviews[key] = row
    for path in sorted(source.glob("worker_*.jsonl")):
        for record in read_jsonl(path):
            if record["question_id"] not in expected_ids:
                raise ValueError("Evaluation case is not in the fixed source question list")
            key = record["question_id"], record["condition"], record["attempt"]
            saved_review = reviews.get(key)
            if saved_review and saved_review["clinical_review"] in ("approved", "rejected"):
                expected_hash = review_content_hash(record)
                if review_content_hash(saved_review) != expected_hash or saved_review.get("content_sha256", expected_hash) != expected_hash:
                    raise ValueError("Review decision belongs to different content; re-review the current input")
                record["clinical_review"] = saved_review["clinical_review"]
            if record["condition"] == "clean_repeat":
                continue
            if record["status"] == "error":
                raise ValueError("Source evaluation has operational errors")
            if record["condition"] != "baseline" and (not record["syntactic_valid"] or record["clinical_review"] == "rejected"):
                continue
            if key in seen:
                raise ValueError("Duplicate evaluation case")
            seen.add(key)
            records.append(record)
    return sorted(records, key=lambda r: (r["question_id"], r["condition"], r["attempt"]))


def summarize(records: list[dict]) -> dict:
    clean = [r for r in records if r["condition"] == "baseline"]
    perturbed = [r for r in records if r["condition"] != "baseline"]
    groups = {}
    for record in perturbed:
        groups.setdefault(record["question_id"], []).append(record)
    return {
        "clean_questions": len(clean), "clean_correct": sum(r["correct"] for r in clean),
        "clean_accuracy": sum(r["correct"] for r in clean) / len(clean) if clean else None,
        "perturbed_cases": len(perturbed), "perturbed_correct": sum(r["correct"] for r in perturbed),
        "perturbed_accuracy": sum(r["correct"] for r in perturbed) / len(perturbed) if perturbed else None,
        "all_perturbations_correct_questions": sum(all(r["correct"] for r in values) for values in groups.values()),
        "questions_with_perturbations": len(groups),
        "format_errors": sum(r["status"] == "format_error" for r in records),
        "refusals": sum(r["status"] == "refusal" for r in records),
        "operational_errors": sum(r["status"] == "error" for r in records),
    }


def worker(args, source_manifest, cases):
    variant = VARIANTS[args.worker]
    adapter = args.training_dir / variant / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise ValueError(f"Missing trained adapter: {adapter}")
    cfg = source_manifest["config"]
    model = LocalModel(cfg["model"], source_manifest["target_revision"], args.worker,
                       cfg["max_input_tokens"], adapter_path=str(adapter))
    path = args.output_dir / f"{variant}.jsonl"
    existing = read_jsonl(path)
    expected = {(r["question_id"], r["condition"], r["attempt"]): r for r in cases}
    for record in existing:
        key = record["question_id"], record["condition"], record["attempt"]
        case = expected.get(key)
        if case is None or record.get("variant") != variant or record.get("gold") != case["gold"]:
            raise ValueError("Saved replay response has inconsistent case, variant or gold")
        if record.get("status") == "error":
            raise ValueError("Replay contains an execution error; preserve it and use a new output directory")
        if record.get("input_sha256") != digest(case["target_messages"]):
            raise ValueError("Saved replay input changed")
    completed = {(r["question_id"], r["condition"], r["attempt"]) for r in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate replay records")
    for case in cases:
        key = case["question_id"], case["condition"], case["attempt"]
        if key in completed:
            continue
        try:
            result = model.generate(case["target_messages"], seed=call_seed(cfg["seed"], *key, "target"),
                                    temperature=cfg["target_temperature"], max_new_tokens=cfg["target_max_new_tokens"])
        except Exception as exc:
            append_record(path, {"question_id": key[0], "condition": key[1], "attempt": key[2],
                                "variant": variant, "gold": case["gold"], "answer": None, "correct": False,
                                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                                "input_sha256": digest(case["target_messages"])})
            raise
        score = score_answer(result["raw"])
        record = {"question_id": key[0], "condition": key[1], "attempt": key[2],
                  "variant": variant, "gold": case["gold"], "note": case["note"],
                  "clinical_review": case["clinical_review"], "answer": score["answer"],
                  "input_sha256": digest(case["target_messages"]),
                  "status": score["status"], "correct": score["status"] == "ok" and score["answer"] == case["gold"], **result}
        append_record(path, record)
        print(json.dumps({k: record[k] for k in ("variant", "question_id", "condition", "attempt", "correct")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument("--allow-provisional", action="store_true", help="Include unreviewed candidates for pipeline smoke only")
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    args.training_dir = args.training_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    devices = configure()
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_manifest = json.loads((args.source_run / "run.json").read_text())
    if source_manifest["config"]["split"] not in ("dev", "test"):
        raise ValueError("Replay evaluation must use held-out dev/test questions")
    if not (args.source_run / "COMPLETE").exists():
        raise ValueError("Source evaluation is incomplete")
    training_manifest = json.loads((args.training_dir / "training.json").read_text())
    if training_manifest.get("status") != "complete":
        raise ValueError("Training is incomplete")
    if (training_manifest["model"] != source_manifest["target_model"] or
            training_manifest["target_revision"] != source_manifest["target_revision"]):
        raise ValueError("Training and evaluation use different base model revisions")
    cases = selected_records(args.source_run)
    if not args.allow_provisional:
        cases = [r for r in cases if r["condition"] == "baseline" or r["clinical_review"] == "approved"]
    if not any(r["condition"] != "baseline" for r in cases):
        raise ValueError("No approved perturbations; review first or explicitly use --allow-provisional for smoke")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_replay = args.output_dir / "replay.json"
    if saved_replay.exists():
        require_gpu_mapping(json.loads(saved_replay.read_text())["physical_gpus"])
    if args.worker is not None:
        replay_manifest = json.loads(saved_replay.read_text())
        require_gpu_mapping(replay_manifest["physical_gpus"])
        worker(args, source_manifest, cases)
        return
    with (ROOT / ".gpu.lock").open("w") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_info = subprocess.check_output(["nvidia-smi", f"--id={gpu_mask(devices)}", "--query-gpu=index,memory.used", "--format=csv,noheader"], text=True)
        if any(int(line.split(",")[1].strip().split()[0]) > 2048 for line in gpu_info.strip().splitlines()):
            raise ValueError("A requested GPU is already in use; no other process was stopped")
        # source와 train 문항 겹침을 실제 학습 데이터 원본 ID로 검증한다.
        training_questions = set()
        training_artifacts = {}
        for variant in VARIANTS:
            dataset_path = args.training_dir / variant / "dataset.jsonl"
            if not dataset_path.exists():
                raise ValueError(f"Missing training provenance for {variant}")
            dataset = read_jsonl(dataset_path)
            if not dataset:
                raise ValueError(f"Empty training provenance for {variant}")
            training_questions.update(r["question_id"] for r in dataset)
            adapter = args.training_dir / variant / "adapter"
            if not (adapter / "adapter_config.json").exists() or not (adapter / "adapter_model.safetensors").exists():
                raise ValueError(f"Incomplete adapter: {variant}")
            for file in [dataset_path, *sorted(adapter.glob("*"))]:
                if file.is_file():
                    training_artifacts[str(file.relative_to(args.training_dir))] = file_digest(file)
        if not training_questions:
            raise ValueError("Training dataset provenance is missing")
        if training_questions & set(source_manifest["question_ids"]):
            raise ValueError("Train/evaluation question overlap")
        if training_questions != set(training_manifest["source_question_ids"]):
            raise ValueError("Training question provenance differs from training manifest")
        manifest = {"source_run": str(args.source_run), "training_dir": str(args.training_dir),
                    "allow_provisional": args.allow_provisional, "cases": len(cases),
                    "target_model": source_manifest["target_model"], "target_revision": source_manifest["target_revision"],
                    "source_manifest_sha256": digest(source_manifest), "cases_sha256": digest(cases),
                    "training_manifest_sha256": digest(training_manifest), "training_artifacts_sha256": training_artifacts,
                    "evaluation_type": "fixed perturbation transfer; no new attacks against trained adapters",
                    "variants": VARIANTS, "physical_gpus": list(devices),
                    "base_precision": "bfloat16 for both untrained and adapter inference"}
        path = args.output_dir / "replay.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Replay configuration changed; use a new output directory")
        path.write_text(json.dumps(manifest, indent=2))
        children, handles = [], []
        try:
            for index, variant in enumerate(VARIANTS):
                log = (args.output_dir / f"{variant}.log").open("a")
                handles.append(log)
                command = [sys.executable, "-u", "-m", "gaim.replay", "--source-run", str(args.source_run),
                           "--training-dir", str(args.training_dir), "--output-dir", str(args.output_dir), "--worker", str(index)]
                if args.allow_provisional:
                    command.append("--allow-provisional")
                children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_environment(devices)))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None, 0) for p in children):
                    raise RuntimeError("Replay worker failed; inspect variant logs")
                time.sleep(1)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError("Replay failed")
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
            for p in children:
                p.wait()
            for handle in handles:
                handle.close()
        results = {"untrained": summarize(cases)}
        for variant in VARIANTS:
            records = read_jsonl(args.output_dir / f"{variant}.jsonl")
            if len(records) != len(cases):
                raise ValueError(f"Incomplete evaluation for {variant}")
            results[variant] = summarize(records)
        (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))
        lines = ["# 추가 학습 후 고정 입력 평가", "", "같은 held-out 문제와 같은 교란을 모든 모델에 다시 입력한 결과입니다.",
                 "새로 학습한 모델에 재공격한 결과가 아니므로 적응형 공격 전반에 대한 방어 성능으로 해석하지 않습니다.", "",
                 f"미검토 교란 포함: {args.allow_provisional}. 포함한 경우 수치는 실행 검증용 잠정 결과입니다.", "",
                 "| 모델 | 원문 정답/문항 | 교란 정답/시도 | 모든 교란에 정답인 문항 | 형식 오류 |",
                 "|---|---:|---:|---:|---:|"]
        for name, result in results.items():
            lines.append(f"| {name} | {result['clean_correct']}/{result['clean_questions']} | {result['perturbed_correct']}/{result['perturbed_cases']} | {result['all_perturbations_correct_questions']}/{result['questions_with_perturbations']} | {result['format_errors']} |")
        lines.extend(["", "교란별 시도를 독립적인 임상 문항처럼 세면 안 됩니다. 표의 시도 단위 정확도는 기술 통계이며, 본 실험은 문항 수와 seed를 늘려 재검증해야 합니다."])
        (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
        (args.output_dir / "COMPLETE").write_text("complete\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
