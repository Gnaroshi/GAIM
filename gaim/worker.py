"""문항을 worker별로 분할하되, 한 문항의 모든 비교 조건은 같은 worker에서 실행."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .backend import LocalModel, call_seed
from .perturbations import (build_target_messages, build_attacker_messages,
                            parse_note, validate_note, score_answer, literal_controls)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt log {path}:{line_no}; preserve and repair before resume") from exc
    return records


def append_record(path: Path, record: dict) -> None:
    # 매 요청이 끝나면 즉시 flush. 중단되더라도 완료된 실제 응답을 재사용한다.
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def feedback(record: dict) -> dict:
    return {"candidate_note": record["candidate_note"], "applied_note": record["note"],
            "response": record["raw_response"], "correct": record["correct"],
            "status": record["status"], "syntactic_valid": record["syntactic_valid"],
            "validity_reasons": record["validity_reasons"]}


def schedule(config: dict) -> list[tuple[str, int, str]]:
    result = [("baseline", 0, "none")]
    for condition in ("clean_repeat", "independent", "adaptive"):
        for attempt in range(1, config["k"] + 1):
            category = "none" if condition == "clean_repeat" else config["categories"][(attempt - 1) % len(config["categories"])]
            result.append((condition, attempt, category))
    if config.get("literal_controls", True):
        # controls의 실제 길이는 실행 시 확인한다.
        result.extend(("literal_control", i + 1, "nonclinical_background") for i in range(3))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run.json").read_text())
    cfg = manifest["config"]
    questions = read_jsonl(args.run_dir / "questions.jsonl")[args.worker::4]
    path = args.run_dir / f"worker_{args.worker}.jsonl"
    existing = read_jsonl(path)
    if any(r.get("status") == "error" for r in existing):
        raise ValueError("Run contains operational errors; preserve logs and start a new run after fixing the cause")
    done = {(r["question_id"], r["condition"], r["attempt"]): r for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate completed request records; refusing resume")
    generation_path = args.run_dir / f"generations_{args.worker}.jsonl"
    generation_records = read_jsonl(generation_path)
    if any(r.get("error") for r in generation_records):
        raise ValueError("Run contains a generator execution error; start a new run after fixing the cause")
    generations = {(r["question_id"], r["condition"], r["attempt"]): r for r in generation_records}
    if len(generations) != len(generation_records):
        raise ValueError("Duplicate generator records; refusing resume")
    model = LocalModel(cfg["model"], manifest["target_revision"], args.worker, cfg["max_input_tokens"])
    for item in questions:
        history = []
        controls = literal_controls(item)
        if len(controls) != 3:
            raise ValueError("Expected exactly three literal controls")
        for condition, attempt, category in schedule(cfg):
            key = (item["id"], condition, attempt)
            if key in done:
                if condition == "adaptive":
                    history.append(feedback(done[key]))
                continue
            note = ""
            generator = None
            validity = {"syntactic_valid": True, "reasons": []}
            category_name = category
            if condition == "literal_control":
                control = controls[attempt - 1]
                note = control["note"]
                category_name = control["category"]
                validity = validate_note(note, item)
            elif condition in ("independent", "adaptive"):
                generator = generations.get(key)
                if generator is None:
                    messages = build_attacker_messages(item, history if condition == "adaptive" else [], category)
                    try:
                        generation = model.generate(
                            messages, seed=call_seed(cfg["seed"], *key, "attacker"),
                            temperature=cfg["attacker_temperature"], max_new_tokens=cfg["attacker_max_new_tokens"])
                    except Exception as exc:
                        append_record(generation_path, dict(question_id=item["id"], condition=condition,
                                      attempt=attempt, messages=messages, error=f"{type(exc).__name__}: {exc}"))
                        raise
                    generator = dict(question_id=item["id"], condition=condition, attempt=attempt,
                                     messages=messages, **generation)
                    append_record(generation_path, generator)
                    generations[key] = generator
                try:
                    note = parse_note(generator["raw"])
                    validity = validate_note(note, item)
                except ValueError as exc:
                    validity = {"syntactic_valid": False, "reasons": [str(exc)]}
            # 무효 후보도 예산을 소비하지만 target에는 원문만 전송한다.
            # 실패한 후보를 몰래 여러 번 재생성하여 한 방법에 더 많은 기회를 주지 않는다.
            applied_note = note if validity["syntactic_valid"] else ""
            messages = build_target_messages(item, applied_note)
            record = {
                "question_id": item["id"], "condition": condition, "attempt": attempt,
                "category": category_name, "candidate_note": note, "note": applied_note,
                "syntactic_valid": validity["syntactic_valid"], "validity_reasons": validity["reasons"],
                "clinical_review": "approved" if condition in ("baseline", "clean_repeat") else "pending",
                "gold": item["answer_idx"], "run_kind": cfg["run_kind"],
                "question_sha256": item["question_sha256"], "target_messages": messages,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "generator_input_tokens": generator["input_tokens"] if generator else 0,
                "generator_output_tokens": generator["output_tokens"] if generator else 0,
                "generator_latency_s": generator["latency_s"] if generator else 0,
                "control_name": controls[attempt - 1]["name"] if condition == "literal_control" else None,
            }
            try:
                response = model.generate(
                    messages, seed=call_seed(cfg["seed"], *key, "target"),
                    temperature=cfg["target_temperature"], max_new_tokens=cfg["target_max_new_tokens"])
                score = score_answer(response["raw"])
                record.update({k: v for k, v in response.items() if k != "raw"})
                record.update(answer=score["answer"], status=score["status"],
                              explanation=score["explanation"], raw_response=response["raw"],
                              correct=score["status"] == "ok" and score["answer"] == item["answer_idx"])
            except Exception as exc:
                record.update(answer=None, status="error", explanation="", raw_response="", correct=False,
                              error=f"{type(exc).__name__}: {exc}", input_tokens=0, output_tokens=0, latency_s=0)
                append_record(path, record)
                raise  # OOM/환경 오류를 모델의 오답으로 바꿔서 계속 실행하지 않는다.
            append_record(path, record)
            done[key] = record
            if condition == "adaptive":
                history.append(feedback(record))
            print(json.dumps({"worker": args.worker, "question": item["id"], "condition": condition,
                              "attempt": attempt, "answer": record["answer"], "correct": record["correct"],
                              "valid": validity["syntactic_valid"]}), flush=True)
    (args.run_dir / f"worker_{args.worker}.done").write_text("complete\n")


if __name__ == "__main__":
    main()
