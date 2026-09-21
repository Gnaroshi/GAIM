"""train 분할의 실제 생성 기록으로 네 개의 독립 QLoRA 어댑터를 학습한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from .metrics import aggregate_run
from .environment import configure, physical_gpus, gpu_mask, require_gpu_mapping, child_environment
from .perturbations import build_target_messages, literal_controls, validate_note


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean", "random", "independent", "adaptive")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: JSON object가 아닙니다.")
        rows.append(row)
    return rows


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_manifest(output: Path, value: dict) -> None:
    _write_json(output / "training_manifest.json", value)
    _write_json(output / "training.json", value)


def _review_decisions(source: Path) -> dict:
    decisions = {}
    path = source / "review.csv"
    if not path.exists():
        return decisions
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["question_id"], row["condition"], int(row["attempt"]))
            if key in decisions:
                raise ValueError("review.csv에 중복 판정이 있습니다.")
            if row["clinical_review"] not in {"pending", "approved", "rejected"}:
                raise ValueError("review.csv에 알 수 없는 판정이 있습니다.")
            decisions[key] = row["clinical_review"]
    return decisions


def build_training_data(source_run: str | Path, *, allow_provisional: bool = False,
                        seed: int = 20260921) -> tuple[dict[str, list[dict]], dict]:
    """실제 기록에서 공통 문항과 동일 행 수를 만든다. 모델 호출·파일 쓰기 없음."""
    source = Path(source_run).resolve()
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if config.get("split") != "train":
        raise ValueError("학습 원천은 split=train 생성 실행만 허용합니다. dev/test는 학습에 사용할 수 없습니다.")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("target_revision", "")):
        raise ValueError("학습 모델에는 원천 실행의 고정된 40자리 revision이 필요합니다.")
    audit = aggregate_run(source, bootstrap_samples=1, write=False)
    if not audit["integrity"]["valid"]:
        codes = sorted({issue["code"] for issue in audit["integrity"]["issues"] if issue["severity"] == "error"})
        raise ValueError(f"원천 생성 실행의 기록 검증이 실패했습니다: {codes}")
    items = _load_jsonl(source / "questions.jsonl")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("questions.jsonl의 문항 ID가 중복됩니다.")
    if {item["id"] for item in items} != set(manifest["question_ids"]):
        raise ValueError("보관 문항과 실행 manifest의 ID가 다릅니다.")
    split_ids = manifest.get("dataset_manifest", {}).get("selection", {}).get("sampled_ids", {})
    train_ids = set(split_ids.get("train", []))
    heldout_ids = set(split_ids.get("dev", [])) | set(split_ids.get("test", []))
    if not train_ids or any(item["id"] not in train_ids or item["id"] in heldout_ids for item in items):
        raise ValueError("데이터 manifest에서 train 소속과 dev/test 분리를 확인할 수 없습니다.")
    rows = [row for path in sorted(source.glob("worker_*.jsonl")) for row in _load_jsonl(path)]
    if any(row.get("status") == "error" for row in rows):
        raise ValueError("원천 실행에 operational error가 있습니다. 불완전 실행으로 학습하지 않습니다.")
    decisions = _review_decisions(source)
    by_item: dict = {}
    for row in rows:
        key = (row["question_id"], row["condition"], row["attempt"])
        row = dict(row)
        saved = decisions.get(key)
        row["clinical_review"] = saved if saved in {"approved", "rejected"} else row.get("clinical_review", "pending")
        by_item.setdefault(row["question_id"], []).append(row)
    datasets = {arm: [] for arm in ARMS}
    included, excluded, selections = [], [], []
    provisional_used = 0
    for item in sorted(items, key=lambda item: item["id"]):
        qid = item["id"]
        question_hash = hashlib.sha256(item["question"].encode()).hexdigest()
        if item.get("source_split") != "train" or question_hash != item.get("question_sha256"):
            raise ValueError(f"원본 train 문항의 출처/hash가 일치하지 않습니다: {qid}")
        candidates = {condition: [] for condition in ("independent", "adaptive", "literal_control")}
        for row in by_item.get(qid, []):
            if row["gold"] != item["answer_idx"] or row.get("question_sha256") != question_hash:
                raise ValueError(f"worker의 정답 또는 원문 hash가 원본과 다릅니다: {qid}")
            condition = row["condition"]
            if condition not in candidates or not row.get("syntactic_valid"):
                continue
            note = row.get("note", "")
            if not validate_note(note, item)["syntactic_valid"]:
                raise ValueError(f"기록상 유효한 note가 현재 구문 검사에서 실패했습니다: {qid}/{condition}")
            if row.get("target_messages") != build_target_messages(item, note):
                raise ValueError(f"실제 target 입력이 원문과 note에서 재구성한 입력과 다릅니다: {qid}/{condition}")
            if row["clinical_review"] == "rejected":
                continue
            if row["clinical_review"] != "approved" and not allow_provisional:
                continue
            candidates[condition].append(row)
        chosen = {}
        for condition in ("independent", "adaptive"):
            ordered = sorted(candidates[condition], key=lambda row: row["attempt"])
            wrong = [row for row in ordered if row["status"] == "ok" and row["answer"] != item["answer_idx"]]
            if ordered:
                chosen[condition] = (wrong or ordered)[0]
        # random은 답을 관찰하지 않고 seed+문항 ID로 고정 문장을 선택한다.
        controls = literal_controls(item)
        number = int.from_bytes(hashlib.sha256(f"{seed}|{qid}|random".encode()).digest()[:8], "big") % len(controls)
        control = controls[number]
        matches = [row for row in candidates["literal_control"] if row["attempt"] == number + 1
                   and row["note"] == control["note"]]
        if matches:
            chosen["random"] = matches[0]
        missing = sorted({"random", "independent", "adaptive"} - set(chosen))
        if missing:
            excluded.append({"question_id": qid, "reason": "공통 유효·검토 조건을 만족하는 후보 없음", "missing": missing})
            continue
        included.append(qid)
        choice_summary = {"question_id": qid}
        for arm, selected in chosen.items():
            pending = selected["clinical_review"] != "approved"
            provisional_used += int(pending)
            choice_summary[arm] = {"condition": selected["condition"], "attempt": selected["attempt"],
                                   "target_answer": selected["answer"], "target_correct": selected["correct"],
                                   "clinical_review": selected["clinical_review"], "note": selected["note"]}
        selections.append(choice_summary)
        for arm in ARMS:
            for variant in ("clean", "additional"):
                selected = chosen.get(arm) if variant == "additional" else None
                note = selected["note"] if selected else ""
                datasets[arm].append({
                    "question_id": qid, "source_split": "train", "question_sha256": question_hash,
                    "arm": arm, "variant": variant, "messages": build_target_messages(item, note),
                    # 정답만 감독한다. 생성 모델의 해설이나 오답 reasoning은 학습 타깃으로 쓰지 않는다.
                    "completion": json.dumps({"answer": item["answer_idx"], "explanation": ""}, separators=(",", ":")),
                    "gold": item["answer_idx"], "note": note,
                    "clinical_review": selected["clinical_review"] if selected else "original_unchanged",
                    "selected_attempt": selected["attempt"] if selected else None,
                })
    if not included:
        raise ValueError("학습 가능한 공통 문항이 없습니다. independent/adaptive와 고정 random 문장의 검토표를 확인하세요. "
                         "--allow-provisional을 명시하면 미검토 자료를 탐색 학습에 포함할 수 있습니다.")
    source_paths = [source / "run.json", source / "questions.jsonl", *sorted(source.glob("worker_*.jsonl"))]
    if (source / "review.csv").exists():
        source_paths.append(source / "review.csv")
    info = {"source_run": str(source), "source_split": "train", "source_run_kind": manifest["run_kind"],
            "target_model": manifest["target_model"], "target_revision": manifest["target_revision"],
            "model": manifest["target_model"], "source_question_ids": included, "variants": list(ARMS),
            "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
            "source_sha256": {path.name: _sha(path) for path in source_paths},
            "question_ids": included, "n_questions": len(included), "rows_per_arm": 2 * len(included),
            "excluded": excluded, "selection": selections, "seed": seed,
            "allow_provisional": allow_provisional, "provisional_selected_notes": provisional_used,
            "claim_scope": "unreviewed_exploratory" if allow_provisional else "reviewed_training_data",
            "selection_policy": "same question intersection; first actual wrong answer among eligible attempts, otherwise first eligible; random fixed by seed before observing answers",
            "supervision": "gold answer JSON only; empty explanation; no generated rationale",
            "fairness": "2N rows and same optimizer steps/seed for every arm; token lengths differ and are reported"}
    return datasets, info


def prepare_training(source_run: str | Path, output_dir: str | Path, *, steps: int = 100,
                     max_seq_length: int = 2048, allow_provisional: bool = False,
                     seed: int = 20260921) -> dict:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps는 양수여야 합니다.")
    if max_seq_length not in (2048, 4096):
        raise ValueError("max-seq-length는 2048 또는 4096이어야 합니다.")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output-dir가 비어 있지 않습니다. 기존 학습 자료를 덮어쓰지 않습니다.")
    datasets, info = build_training_data(source_run, allow_provisional=allow_provisional, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    for arm, rows in datasets.items():
        arm_dir = output / arm
        arm_dir.mkdir()
        (arm_dir / "dataset.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    info.update({"schema_version": 1, "status": "prepared", "output_dir": str(output), "steps": steps,
                 "max_seq_length": max_seq_length, "physical_gpus": dict(zip(ARMS, physical_gpus())),
                 "code_sha256": {name: _sha(ROOT / "gaim" / name)
                                  for name in ("training.py", "metrics.py", "perturbations.py", "environment.py")},
                 "dataset_sha256": {arm: _sha(output / arm / "dataset.jsonl") for arm in ARMS},
                 "training": {"quantization": "4-bit NF4", "compute_dtype": "bfloat16", "lora_r": 8,
                              "lora_alpha": 16, "target_modules": "all-linear", "lora_dropout": 0,
                              "gradient_checkpointing": True, "batch_size": 1, "gradient_accumulation": 2,
                              "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.0,
                              "max_grad_norm": 1.0, "silent_truncation": False}})
    _save_manifest(output, info)
    return info


def tokenize_example(tokenizer, row: dict, max_seq_length: int) -> dict:
    """답변 구간에만 loss를 주며 길이 초과/템플릿 경계 불일치는 오류로 멈춘다."""
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template([*row["messages"], {"role": "assistant", "content": row["completion"]}],
                                               tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(complete, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template의 prompt/assistant 토큰 경계가 일치하지 않습니다. loss mask를 추측하지 않습니다.")
    if len(full_ids) > max_seq_length:
        raise ValueError(f"{row['question_id']}: sequence {len(full_ids)} > {max_seq_length}; truncation 없이 중단합니다.")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("감독할 답변 토큰이 없습니다.")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def _worker(output: Path, worker: int) -> None:
    configure()
    manifest = json.loads((output / "training_manifest.json").read_text())
    require_gpu_mapping([manifest["physical_gpus"][arm] for arm in ARMS])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    arm = ARMS[worker]
    arm_dir = output / arm
    if _sha(arm_dir / "dataset.jsonl") != manifest["dataset_sha256"][arm]:
        raise ValueError("학습 데이터가 준비 후 변경되었습니다.")
    rows = _load_jsonl(arm_dir / "dataset.jsonl")
    torch.cuda.set_device(worker)
    torch.set_num_threads(2)
    set_seed(manifest["seed"])
    model_id, revision = manifest["target_model"], manifest["target_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    encoded = [tokenize_example(tokenizer, row, manifest["max_seq_length"]) for row in rows]
    _write_json(arm_dir / "tokenization.json", {"rows": len(rows), "max_tokens": max(len(row["input_ids"]) for row in encoded),
                                                "total_tokens_per_epoch": sum(len(row["input_ids"]) for row in encoded)})
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True,
                                                quantization_config=quantization, torch_dtype=torch.bfloat16,
                                                device_map={"": worker}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0,
                                           bias="none", task_type="CAUSAL_LM"))
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    rng = random.Random(manifest["seed"])
    order, position = [], 0
    processed_tokens, processed_rows = 0, 0
    started = time.monotonic()
    log_path = arm_dir / "training_loss.jsonl"
    with log_path.open("w") as log:
        for step in range(1, manifest["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(2):
                if position >= len(order):
                    order = list(range(len(encoded)))
                    rng.shuffle(order)
                    position = 0
                example = encoded[order[position]]
                position += 1
                batch = {key: torch.tensor([value], dtype=torch.long, device=f"cuda:{worker}")
                         for key, value in example.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("유한하지 않은 loss입니다. 학습을 중단하고 로그를 보존합니다.")
                (loss / 2).backward()
                losses.append(float(loss.detach().cpu()))
                processed_rows += 1
                processed_tokens += len(example["input_ids"])
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("유한하지 않은 gradient입니다. optimizer를 갱신하지 않습니다.")
            optimizer.step()
            entry = {"step": step, "loss": sum(losses) / len(losses), "grad_norm": float(grad_norm),
                     "processed_rows": processed_rows, "processed_input_tokens": processed_tokens,
                     "elapsed_s": time.monotonic() - started, "arm": arm}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
    adapter_dir = arm_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    _write_json(arm_dir / "training_result.json", {"status": "complete", "arm": arm, "steps": manifest["steps"],
                "adapter_dir": str(adapter_dir), "base_model": model_id, "base_revision": revision,
                "processed_rows": processed_rows, "processed_input_tokens": processed_tokens,
                "claim_scope": manifest["claim_scope"], "provisional_selected_notes": manifest["provisional_selected_notes"],
                "peak_memory_bytes": torch.cuda.max_memory_allocated(worker), "elapsed_s": time.monotonic() - started})


def _launch(output: Path, manifest: dict) -> None:
    import fcntl

    devices = configure()
    require_gpu_mapping([manifest["physical_gpus"][arm] for arm in ARMS])
    env = child_environment(devices)
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    children, handles = [], []
    with (ROOT / ".gpu.lock").open("w") as gpu_lock:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("다른 GAIM 실행이 이 프로젝트의 GPU 잠금을 사용 중입니다.") from exc
        gpu_info = subprocess.check_output(["nvidia-smi", f"--id={gpu_mask(devices)}", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        (output / "gpu_before.txt").write_text(gpu_info)
        for line in gpu_info.strip().splitlines():
            index, used = [int(value.strip()) for value in line.split(",")]
            if used > 2048:
                raise ValueError(f"GPU {index}에 이미 {used} MiB가 사용 중입니다. 다른 프로세스를 변경하지 않았습니다.")
        environment_path = output / "environment.txt"
        environment_path.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                                            text=True, cwd=ROOT, env=env), encoding="utf-8")
        manifest["environment"] = {"pip_freeze_file": environment_path.name,
                                   "pip_freeze_sha256": _sha(environment_path),
                                   "python": sys.version, "executable": sys.executable, "hf_home": env["HF_HOME"]}
        manifest["code_sha256"] = {name: _sha(ROOT / "gaim" / name)
                                    for name in ("training.py", "metrics.py", "perturbations.py", "environment.py")}
        manifest["status"] = "running"
        _save_manifest(output, manifest)
        try:
            for worker, arm in enumerate(ARMS):
                handle = (output / arm / "worker.log").open("w")
                handles.append(handle)
                children.append(subprocess.Popen([sys.executable, "-u", "-m", "gaim.training", "--output-dir", str(output),
                                                  "--worker", str(worker)], cwd=ROOT, env=env,
                                                 stdout=handle, stderr=subprocess.STDOUT))
            _write_json(output / "processes.json", {"launcher": os.getpid(), "workers": dict(zip(ARMS, [child.pid for child in children]))})
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("학습 worker가 실패했습니다. 조건별 worker.log를 확인하세요.")
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("학습 worker가 실패했습니다.")
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # 이번 launcher가 생성한 자식만 종료한다. 기존 GPU 프로세스는 건드리지 않는다.
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            for handle in handles:
                handle.close()
            _save_manifest(output, manifest)
    report = ("# 네 조건 QLoRA 학습 기록\n\n"
              f"상태: {manifest['status']}. 공통 train 문항 {manifest['n_questions']}개, 조건당 {manifest['rows_per_arm']}행, "
              f"optimizer {manifest['steps']}회.\n\n"
              f"해석 범위: {manifest['claim_scope']}. 미검토 선택 문장 {manifest['provisional_selected_notes']}개.\n\n"
              "clean은 원문을 두 번, 나머지 조건은 원문과 선택한 추가 문장을 한 번씩 사용했습니다. "
              "모든 조건은 같은 source gold 답만 감독했습니다. loss 감소는 견고성 개선의 증거가 아닙니다. "
              "별도 held-out 평가로 효과를 확인해야 합니다.\n")
    (output / "training_report.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--allow-provisional", action="store_true", help="미검토 후보를 명시적으로 탐색 학습에 포함")
    parser.add_argument("--prepare-only", action="store_true", help="데이터와 manifest만 작성하고 GPU 학습은 실행하지 않음")
    parser.add_argument("--worker", type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker is not None:
            _worker(args.output_dir.resolve(), args.worker)
            return 0
        if args.source_run is None:
            parser.error("--source-run이 필요합니다.")
        manifest = prepare_training(args.source_run, args.output_dir, steps=args.steps,
                                    max_seq_length=args.max_seq_length, allow_provisional=args.allow_provisional, seed=args.seed)
        print(f"학습 데이터 준비: 공통 {manifest['n_questions']}문항, 조건당 {manifest['rows_per_arm']}행; {manifest['claim_scope']}", flush=True)
        if not args.prepare_only:
            _launch(args.output_dir.resolve(), manifest)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"학습 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
