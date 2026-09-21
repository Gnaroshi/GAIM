#!/usr/bin/env python3
"""공개된 실제 기록을 CPU만으로 검증한다. 모델 호출·다운로드·파일 쓰기 없음."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

# 임의의 현재 작업 폴더에서 실행해도 이 저장소의 순수 Python 모듈을 읽는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from gaim.metrics import aggregate_run, review_content_hash
from gaim.perturbations import build_target_messages, score_answer
from gaim.training import build_training_data

ARMS = ("clean", "random", "independent", "adaptive")
SOURCE_FILES = {"run.json", "questions.jsonl", "review.csv", "COMPLETE"} | {
    f"{prefix}_{index}.{suffix}"
    for prefix, suffix in (("worker", "jsonl"), ("generations", "jsonl"), ("worker", "done"))
    for index in range(4)
}
EXPECTED_FILES = {f"{split}/{name}" for split in ("train", "test") for name in SOURCE_FILES} | {
    f"test_replay/{name}"
    for name in ("summary.json", "replay.json", "COMPLETE", *(f"{arm}.jsonl" for arm in ARMS))
}


class ReferenceError(ValueError):
    """기록과 검증 가능한 사실이 일치하지 않을 때 발생한다."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceError(message)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(value) -> str:
    # 원래 replay manifest가 사용한 JSON 직렬화 규칙을 그대로 적용한다.
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f"JSONL objects required: {path.name}")
    return rows


def key(row: dict) -> tuple:
    require(type(row.get("attempt")) is int, "Record attempt must be an integer")
    return row["question_id"], row["condition"], row["attempt"]


def rescore(row: dict, raw_field: str, location: str) -> None:
    """저장된 correct 값 대신 실제 응답 JSON을 다시 파싱해 일치 여부를 확인한다."""
    require(isinstance(row.get(raw_field), str), f"Missing raw response: {location}")
    score = score_answer(row[raw_field])
    require(row.get("status") == score["status"], f"Response status mismatch: {location}")
    require(row.get("answer") == score["answer"], f"Parsed answer mismatch: {location}")
    expected = score["status"] == "ok" and score["answer"] == row["gold"]
    require(type(row.get("correct")) is bool and row["correct"] == expected,
            f"Correctness mismatch: {location}")
    require(row["status"] != "error", f"Operational error: {location}")


def verify_files(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    files = manifest.get("files", {})
    require(manifest.get("schema_version") == 1, "Unsupported reference manifest schema")
    require(set(files) == EXPECTED_FILES, "Reference manifest must list exactly the 39 allowlisted files")
    for relative, entry in files.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"Missing file or symlink: {relative}")
        payload = path.read_bytes()
        require(len(payload) == entry["bytes"], f"Byte count mismatch: {relative}")
        require(sha(payload) == entry["published_sha256"], f"SHA-256 mismatch: {relative}")
        if relative != "test_replay/replay.json":
            require(entry["published_sha256"] == entry["original_sha256"],
                    f"Unexpected change from original record: {relative}")
        if path.name == "COMPLETE" or path.suffix == ".done":
            require(bool(payload.strip()), f"Empty completion marker: {relative}")
    return manifest


def verify_source(root: Path, split: str, n_questions: int) -> tuple[dict, list[dict]]:
    source = root / split
    manifest = read_json(source / "run.json")
    questions = read_rows(source / "questions.jsonl")
    ids = [question["id"] for question in questions]
    require(len(ids) == len(set(ids)) == n_questions, f"Question count or duplicate ID: {split}")
    require(ids == manifest["question_ids"], f"Question order/IDs differ from run manifest: {split}")
    require(manifest["config"]["split"] == split and manifest["config"]["k"] == 5,
            f"Frozen split or attempt budget differs: {split}")
    items = dict(zip(ids, questions))
    for question in questions:
        require(question["source_split"] == split, f"Source split mismatch: {split}/{question['id']}")
        require(sha(question["question"].encode()) == question["question_sha256"],
                f"Original question hash mismatch: {split}/{question['id']}")
    rows = [row for index in range(4) for row in read_rows(source / f"worker_{index}.jsonl")]
    expected = {(qid, "baseline", 0) for qid in ids}
    expected |= {(qid, condition, attempt) for qid in ids
                 for condition, budget in (("clean_repeat", 5), ("independent", 5),
                                            ("adaptive", 5), ("literal_control", 3))
                 for attempt in range(1, budget + 1)}
    keys = [key(row) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            f"Target records missing, duplicated or unexpected: {split}")
    for row in rows:
        location = f"{split}/{key(row)}"
        question = items[row["question_id"]]
        require(row["gold"] == question["answer_idx"], f"Gold mismatch: {location}")
        require(row["question_sha256"] == question["question_sha256"], f"Question hash mismatch: {location}")
        require(row["target_messages"] == build_target_messages(question, row["note"]),
                f"Actual model input differs from original question/options/note: {location}")
        rescore(row, "raw_response", location)
    generations = [row for index in range(4) for row in read_rows(source / f"generations_{index}.jsonl")]
    generation_keys = [key(row) for row in generations]
    expected_generations = {entry for entry in expected if entry[1] in ("independent", "adaptive")}
    require(len(generation_keys) == len(set(generation_keys)) and set(generation_keys) == expected_generations,
            f"Generator records missing, duplicated or unexpected: {split}")
    audit = aggregate_run(source, bootstrap_samples=1, write=False)
    require(audit["integrity"]["valid"], f"Core integrity checks failed: {split}")
    return manifest, rows


def select_cases(source: Path, rows: list[dict]) -> list[dict]:
    """GPU 모듈을 가져오지 않고 저장된 review와 원래 replay 선별 규칙을 적용한다."""
    reviews = {}
    with (source / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["attempt"] = int(row["attempt"])
            item_key = key(row)
            require(item_key not in reviews, "Duplicate review decision")
            reviews[item_key] = row
    cases = []
    for original in rows:
        row = dict(original)
        review = reviews.get(key(row))
        if review and review["clinical_review"] in ("approved", "rejected"):
            content_hash = review_content_hash(row)
            require(review_content_hash(review) == content_hash and
                    review.get("content_sha256", content_hash) == content_hash,
                    f"Review belongs to a different input: {key(row)}")
            row["clinical_review"] = review["clinical_review"]
        if row["condition"] == "clean_repeat":
            continue
        if row["condition"] != "baseline" and (not row["syntactic_valid"] or row["clinical_review"] == "rejected"):
            continue
        cases.append(row)
    return sorted(cases, key=key)


def summarize(rows: list[dict]) -> dict:
    # GPU를 import하는 replay 모듈과 분리해서 같은 정수 집계를 독립 재계산한다.
    clean = [row for row in rows if row["condition"] == "baseline"]
    perturbed = [row for row in rows if row["condition"] != "baseline"]
    groups = {}
    for row in perturbed:
        groups.setdefault(row["question_id"], []).append(row)
    clean_correct = sum(row["correct"] for row in clean)
    perturbed_correct = sum(row["correct"] for row in perturbed)
    return {
        "clean_questions": len(clean), "clean_correct": clean_correct,
        "clean_accuracy": clean_correct / len(clean) if clean else None,
        "perturbed_cases": len(perturbed), "perturbed_correct": perturbed_correct,
        "perturbed_accuracy": perturbed_correct / len(perturbed) if perturbed else None,
        "all_perturbations_correct_questions": sum(all(row["correct"] for row in group) for group in groups.values()),
        "questions_with_perturbations": len(groups),
        "format_errors": sum(row["status"] == "format_error" for row in rows),
        "refusals": sum(row["status"] == "refusal" for row in rows),
        "operational_errors": sum(row["status"] == "error" for row in rows),
    }


def verify_reference(root: Path) -> dict:
    root = Path(root).resolve()
    published = verify_files(root)
    train_manifest, train_rows = verify_source(root, "train", 12)
    test_manifest, test_rows = verify_source(root, "test", 100)
    require(not set(train_manifest["question_ids"]) & set(test_manifest["question_ids"]), "Train/test overlap")
    for field in ("target_model", "target_revision", "dataset_manifest_sha256"):
        require(train_manifest[field] == test_manifest[field], f"Train/test provenance differs: {field}")

    replay = read_json(root / "test_replay/replay.json")
    require(replay["variants"] == list(ARMS) and replay["allow_provisional"] is True,
            "Replay variants or clinical-review scope changed")
    require(replay["source_manifest_sha256"] == digest(test_manifest), "Replay source manifest hash mismatch")
    for field in ("target_model", "target_revision"):
        require(replay[field] == test_manifest[field], f"Replay model differs: {field}")
    datasets, training = build_training_data(root / "train", allow_provisional=True)
    require(training["n_questions"] == 10 and training["rows_per_arm"] == 20,
            "Frozen training selection is not 10 questions / 20 rows per arm")
    require(training["question_ids"] == published["validation"]["training_question_ids"],
            "Reconstructed training question IDs differ")
    for arm in ARMS:
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in datasets[arm]).encode()
        calculated = sha(payload)
        require(calculated == published["validation"]["reconstructed_training_dataset_sha256"][arm] ==
                replay["training_artifacts_sha256"][f"{arm}/dataset.jsonl"],
                f"Original training dataset SHA-256 mismatch: {arm}")

    cases = select_cases(root / "test", test_rows)
    require(len(cases) == replay["cases"] == 1190, "Replay must contain the frozen 1190 inputs")
    require(digest(cases) == replay["cases_sha256"], "Selected replay cases hash mismatch")
    by_key = {key(row): row for row in cases}
    summaries = {"untrained": summarize(cases)}
    for arm in ARMS:
        rows = read_rows(root / f"test_replay/{arm}.jsonl")
        keys = [key(row) for row in rows]
        require(len(keys) == len(set(keys)) == len(cases) and set(keys) == set(by_key),
                f"Replay cases missing, duplicated or unexpected: {arm}")
        for row in rows:
            case = by_key[key(row)]
            location = f"test_replay/{arm}/{key(row)}"
            require(row["variant"] == arm, f"Variant mismatch: {location}")
            for field in ("gold", "note", "clinical_review"):
                require(row[field] == case[field], f"Replay {field} differs from fixed input: {location}")
            require(row["input_sha256"] == digest(case["target_messages"]),
                    f"Replay input hash mismatch: {location}")
            rescore(row, "raw", location)
        summaries[arm] = summarize(rows)
    require(summaries == read_json(root / "test_replay/summary.json"), "Recomputed replay summary differs")
    require(not any(name in sys.modules for name in ("torch", "transformers", "bitsandbytes")),
            "CPU verification unexpectedly imported a model runtime")
    return {
        "status": "passed", "scope": "stored-record verification; no inference or training performed",
        "published_files_verified": len(EXPECTED_FILES),
        "source_responses_rescored": {"train": len(train_rows), "test": len(test_rows)},
        "training_questions": training["n_questions"], "training_rows_per_arm": training["rows_per_arm"],
        "training_dataset_hashes_match_original": True,
        "replay_inputs_per_variant": len(cases), "replay_responses_rescored": len(cases) * len(ARMS),
        "same_inputs_and_gold_across_variants": True, "clinical_scope": "unreviewed_exploratory",
        "test_results": {name: {key: summary[key] for key in ("clean_correct", "clean_questions",
                         "perturbed_correct", "perturbed_cases")} for name, summary in summaries.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "repro/reference")
    args = parser.parse_args()
    try:
        result = verify_reference(args.reference_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
