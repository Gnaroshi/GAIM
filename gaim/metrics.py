"""고정 호출 예산 실험의 기록 검증, 문항 단위 집계 및 수동 검토표 생성."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONDITIONS = ("clean_repeat", "independent", "adaptive", "literal_control")
REQUIRED_CONDITIONS = CONDITIONS[:3]
STATUSES = {"ok", "refusal", "format_error", "error"}
REVIEWS = {"pending", "approved", "rejected"}
KEY_FIELDS = ("question_id", "condition", "attempt")
REVIEW_FIELDS = (*KEY_FIELDS, "clinical_review", "reviewer", "review_note", "category",
                 "gold", "answer", "syntactic_valid", "candidate_note", "note", "target_messages",
                 "question_sha256", "content_sha256", "distractor", "modified_question")
REVIEW_BINDING_FIELDS = (*KEY_FIELDS, "gold", "note", "candidate_note", "question_sha256", "target_messages")


def review_content_hash(row: dict) -> str:
    """CSV와 JSONL의 같은 검토 대상에 동일한 SHA를 부여한다. 판정/모델 답은 제외."""
    payload = {field: row.get(field) if row.get(field) is not None else ""
               for field in REVIEW_BINDING_FIELDS}
    payload["question_id"] = str(payload["question_id"])
    if payload["attempt"] != "":
        payload["attempt"] = int(payload["attempt"])
    messages = payload["target_messages"]
    if isinstance(messages, str):
        messages = json.loads(messages) if messages else []
    payload["target_messages"] = messages
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_review_binding(row: dict) -> bool:
    if any(field not in row for field in REVIEW_BINDING_FIELDS):
        return False
    if any(not isinstance(row.get(field), str) for field in ("gold", "note", "candidate_note", "question_sha256")):
        return False
    if not row["question_sha256"] or not row["gold"]:
        return False
    messages = row["target_messages"]
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return False
    return bool(isinstance(messages, list) and messages)


def _issue(issues: list, code: str, message: str, *, severity: str = "error", **context: Any) -> None:
    issues.append({"severity": severity, "code": code, "message": message, **context})


def _key(row: dict) -> tuple[str, str, int]:
    return str(row["question_id"]), row["condition"], int(row["attempt"])


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _csv_text(rows: list[dict], fields: list | tuple) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: json.dumps(row[field], ensure_ascii=False)
                         if isinstance(row.get(field), (dict, list)) else row.get(field, "")
                         for field in fields})
    return buffer.getvalue()


def _load_metadata(run_dir: Path, issues: list) -> dict:
    metadata: dict = {}
    for name in ("config.json", "manifest.json", "run.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON object가 아닙니다")
            if isinstance(data.get("config"), dict):
                metadata.update(data["config"])
            metadata.update(data)
        except (OSError, ValueError) as exc:
            _issue(issues, "invalid_metadata", str(exc), file=name)
    return metadata


def _load_rows(run_dir: Path, issues: list) -> list[dict]:
    rows: list[dict] = []
    files = sorted(run_dir.glob("worker_*.jsonl"))
    if not files:
        _issue(issues, "missing_worker_files", "worker_*.jsonl 기록이 없습니다.")
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON object가 아닙니다")
            except ValueError as exc:
                _issue(issues, "malformed_jsonl", str(exc), file=path.name, line=line_number)
                continue
            row = dict(row)
            row["source_file"], row["source_line"] = path.name, line_number
            row["exclusion_reason"] = ""
            if any(field not in row for field in KEY_FIELDS):
                row["exclusion_reason"] = "missing_key"
            elif (not str(row["question_id"]).strip()
                  or not isinstance(row["attempt"], int) or isinstance(row["attempt"], bool)
                  or row["condition"] not in ("baseline", *CONDITIONS)
                  or (row["condition"] == "baseline" and row["attempt"] != 0)
                  or (row["condition"] != "baseline" and row["attempt"] < 1)):
                row["exclusion_reason"] = "invalid_key"
            if row["exclusion_reason"]:
                _issue(issues, row["exclusion_reason"], "문항/조건/시도 키가 올바르지 않습니다.",
                       file=path.name, line=line_number)
            else:
                row["question_id"] = str(row["question_id"])
            rows.append(row)
    for path in sorted(run_dir.glob("generations_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                generation = json.loads(line)
                if not isinstance(generation, dict):
                    raise ValueError("JSON object가 아닙니다")
                if generation.get("error"):
                    _issue(issues, "generator_operational_error", "생성기 실행 오류가 기록되어 있습니다.",
                           file=path.name, line=line_number, error=generation["error"])
            except ValueError as exc:
                _issue(issues, "malformed_generation_jsonl", str(exc), file=path.name, line=line_number)
    return rows


def _read_review(path: Path, issues: list) -> tuple[dict, list[dict]]:
    saved: dict = {}
    duplicates: list[dict] = []
    if not path.exists():
        return saved, duplicates
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or any(field not in reader.fieldnames for field in
                                             (*KEY_FIELDS, "clinical_review")):
                raise ValueError("수동 검토표에 필수 열이 없습니다.")
            for row in reader:
                try:
                    key = _key(row)
                except (ValueError, KeyError, TypeError):
                    duplicates.append(row)
                    _issue(issues, "invalid_review_key", "수동 검토표 키가 올바르지 않습니다.")
                    continue
                if row.get("clinical_review") not in REVIEWS:
                    _issue(issues, "invalid_review_decision", "수동 검토 상태가 올바르지 않습니다.", key=key)
                if key in saved:
                    duplicates.append(row)
                    _issue(issues, "duplicate_review_key", "수동 검토표에 중복 키가 있습니다.", key=key)
                else:
                    saved[key] = row
    except (OSError, ValueError) as exc:
        _issue(issues, "unreadable_review", str(exc))
    return saved, duplicates


def _validate_and_merge(rows: list[dict], saved_review: dict, issues: list, budgets: dict) -> dict:
    grouped: dict = defaultdict(list)
    for row in rows:
        if not row["exclusion_reason"]:
            grouped[_key(row)].append(row)
    index: dict = {}
    for key, members in grouped.items():
        if len(members) != 1:
            # 중복 중 첫 행만 고르면 worker 재실행에 따라 유리한 결과를 고를 수 있다.
            for row in members:
                row["exclusion_reason"] = "duplicate_key"
            _issue(issues, "duplicate_key", "중복된 모든 행을 집계에서 제외했습니다.", key=key, count=len(members))
            continue
        row = members[0]
        if key[2] > budgets.get(key[1], 0):
            row["exclusion_reason"] = "over_budget"
            _issue(issues, "over_budget", "선언한 호출 예산을 초과한 시도입니다.", key=key)
            continue
        saved = saved_review.get(key, {})
        worker_review = row.get("clinical_review", "pending")
        row["review_binding_valid"] = True
        try:
            row["content_sha256"] = review_content_hash(row) if _has_review_binding(row) else ""
        except (ValueError, TypeError):
            row["content_sha256"] = ""
        # 사람이 확정한 승인/기각과 메모는 재집계 시 worker의 pending 값으로 덮지 않는다.
        if saved.get("clinical_review") in {"approved", "rejected"}:
            if not _has_review_binding(saved) or not row["content_sha256"]:
                row["review_binding_valid"] = False
                _issue(issues, "review_binding_missing", "확정 검토에 원문·추가 문장·입력의 연결 정보가 없어 판정을 적용하지 않았습니다.", key=key)
            else:
                try:
                    saved_hash = review_content_hash(saved)
                    bound = saved_hash == row["content_sha256"] and saved.get("content_sha256", "") in ("", saved_hash)
                except (ValueError, TypeError):
                    bound = False
                if not bound:
                    row["review_binding_valid"] = False
                    _issue(issues, "review_content_mismatch", "검토한 내용과 현재 worker 입력이 달라 기존 판정을 적용하지 않았습니다.", key=key)
            if row["review_binding_valid"]:
                if worker_review in {"approved", "rejected"} and worker_review != saved["clinical_review"]:
                    _issue(issues, "review_disagreement", "worker와 저장된 검토 판정이 달라 수동 판정을 유지했습니다.",
                           severity="warning", key=key)
                row["clinical_review"] = saved["clinical_review"]
            else:
                row["clinical_review"] = "pending"
        else:
            row["clinical_review"] = worker_review
        row["reviewer"] = saved.get("reviewer", "")
        row["review_note"] = saved.get("review_note", "")
        if row["clinical_review"] not in REVIEWS:
            _issue(issues, "invalid_clinical_review", "알 수 없는 검토 상태입니다.", key=key)
            row["clinical_review"] = "pending"
        if row.get("status") not in STATUSES:
            _issue(issues, "invalid_status", "응답 상태가 올바르지 않습니다.", key=key)
            row["status"] = "error"
        if row["status"] == "error":
            _issue(issues, "target_operational_error", "target 실행 오류가 있어 완료된 성능 비교로 취급하지 않습니다.", key=key)
        labels = row.get("answer_labels", ["A", "B", "C", "D"])
        if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
            _issue(issues, "invalid_answer_labels", "선택지 레이블은 문자열 목록이어야 합니다.", key=key)
            labels = ["A", "B", "C", "D"]
        if row.get("gold") not in labels:
            row["exclusion_reason"] = "invalid_gold"
            _issue(issues, "invalid_gold", "정답이 허용 선택지에 없습니다.", key=key)
            continue
        if not isinstance(row.get("syntactic_valid"), bool):
            _issue(issues, "missing_syntactic_validity", "구문 유효성 검사가 기록되지 않았습니다.", key=key)
            row["syntactic_valid"] = False
        if row["status"] == "ok" and row.get("answer") not in labels:
            _issue(issues, "invalid_answer", "ok 응답의 답이 허용 선택지가 아닙니다.", key=key)
            row["status"] = "format_error"
        row["derived_correct"] = row["status"] == "ok" and row.get("answer") == row["gold"]
        if not isinstance(row.get("correct"), bool) or row["correct"] != row["derived_correct"]:
            _issue(issues, "correct_flag_mismatch", "correct 플래그와 실제 답/정답이 다릅니다. 답에서 다시 계산했습니다.", key=key)
        row["provisional_wrong"] = (row["status"] == "ok" and bool(row["syntactic_valid"])
                                    and row.get("answer") != row["gold"])
        row["confirmed_wrong"] = row["provisional_wrong"] and row["clinical_review"] == "approved"
        index[key] = row
    for key, row in list(index.items()):
        baseline = index.get((key[0], "baseline", 0))
        if baseline and row["gold"] != baseline["gold"]:
            _issue(issues, "gold_disagreement", "동일 문항의 정답이 baseline과 다릅니다.", key=key)
            row["exclusion_reason"] = "gold_disagreement"
            del index[key]
    return index


def _slot_stats(index: dict, question_ids: list[str], condition: str, k: int) -> dict:
    attempts = [0] if condition == "baseline" else list(range(1, k + 1))
    records = [index[(qid, condition, attempt)] for qid in question_ids for attempt in attempts
               if (qid, condition, attempt) in index]
    denominator = len(question_ids) * len(attempts)
    statuses = {status: sum(row["status"] == status for row in records) for status in sorted(STATUSES)}
    statuses["missing"] = denominator - len(records)
    correct = sum(row["derived_correct"] for row in records)
    syntactic_invalid = sum(not row["syntactic_valid"] for row in records)
    return {"correct": correct, "denominator_budget_slots": denominator,
            "accuracy": _ratio(correct, denominator), "observed_slots": len(records),
            "status_counts": statuses, "syntactic_invalid": syntactic_invalid,
            "complete": len(records) == denominator,
            "input_tokens": sum(row.get("input_tokens", 0) or 0 for row in records
                                if isinstance(row.get("input_tokens", 0), (int, float))),
            "output_tokens": sum(row.get("output_tokens", 0) or 0 for row in records
                                 if isinstance(row.get("output_tokens", 0), (int, float))),
            "generator_input_tokens": sum(row.get("generator_input_tokens", 0) or 0 for row in records
                                           if isinstance(row.get("generator_input_tokens", 0), (int, float))),
            "generator_output_tokens": sum(row.get("generator_output_tokens", 0) or 0 for row in records
                                            if isinstance(row.get("generator_output_tokens", 0), (int, float)))}


def _outcomes(index: dict, question_ids: list[str], condition: str, k: int, field: str) -> list[int]:
    # 같은 문항의 K회 시도는 하나의 0/1 결과다. 시도들을 독립 표본처럼 세지 않는다.
    return [int(any(index.get((qid, condition, attempt), {}).get(field, False)
                    for attempt in range(1, k + 1))) for qid in question_ids]


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def paired_bootstrap(adaptive: list[int], independent: list[int], *, samples: int = 2000,
                     seed: int = 20260921) -> dict:
    """같은 문항의 두 조건을 함께 재표집한 차이와 percentile 95% CI."""
    if len(adaptive) != len(independent):
        raise ValueError("paired bootstrap은 같은 문항 집합을 요구합니다.")
    if samples < 1:
        raise ValueError("bootstrap samples는 양수여야 합니다.")
    n = len(adaptive)
    if not n:
        return {"n_questions": 0, "difference": None, "ci95": None, "samples": samples, "seed": seed}
    differences = [a - b for a, b in zip(adaptive, independent)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return {"n_questions": n, "difference": sum(differences) / n,
            "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
            "samples": samples, "seed": seed}


def _resolve_budget(metadata: dict, rows: list, explicit_k: int | None, issues: list) -> tuple[int, str]:
    if explicit_k is not None:
        if explicit_k < 1:
            raise ValueError("K는 1 이상이어야 합니다.")
        return explicit_k, "cli"
    for field in ("k", "K", "budget_k", "max_attempts", "attempts_per_condition", "target_call_budget"):
        value = metadata.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, f"metadata.{field}"
    observed = [row.get("attempt", 0) for row in rows
                if isinstance(row.get("attempt"), int) and row.get("condition") != "baseline"]
    k = max([1, *observed])
    _issue(issues, "unknown_planned_budget", "예정 K가 없어 관측 최댓값으로 표시했습니다. 완주 여부를 확인할 수 없습니다.")
    return k, "inferred_observed_maximum"


def aggregate_run(run_dir: str | Path, *, k: int | None = None, bootstrap_samples: int = 2000,
                  seed: int = 20260921, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"실험 디렉터리가 없습니다: {run_dir}")
    issues: list = []
    metadata = _load_metadata(run_dir, issues)
    rows = _load_rows(run_dir, issues)
    k, budget_source = _resolve_budget(metadata, rows, k, issues)
    # literal_control은 세 개의 고정 문장을 검사하는 보조 조건이며 주조건의 K와 분리한다.
    literal_count = metadata.get("literal_control_count", 3)
    if not isinstance(literal_count, int) or isinstance(literal_count, bool) or literal_count < 1:
        _issue(issues, "invalid_literal_control_count", "고정 대조 문장 수가 올바르지 않습니다.")
        literal_count = 3
    budgets = {"baseline": 0, **{condition: k for condition in REQUIRED_CONDITIONS}, "literal_control": literal_count}
    saved_review, duplicate_reviews = _read_review(run_dir / "review.csv", issues)
    index = _validate_and_merge(rows, saved_review, issues, budgets)
    question_ids = sorted({str(row["question_id"]) for row in rows if "question_id" in row})
    expected_ids = metadata.get("question_ids", metadata.get("expected_question_ids", []))
    if isinstance(expected_ids, list):
        expected_set = {str(qid) for qid in expected_ids}
        if expected_set:
            unexpected = sorted(set(question_ids) - expected_set)
            if unexpected:
                _issue(issues, "unexpected_question_ids", "실행 계획에 없는 문항 기록이 있습니다.", question_ids=unexpected)
        question_ids = sorted(set(question_ids) | expected_set)
    expected_count = metadata.get("n_questions", metadata.get("num_questions"))
    if isinstance(expected_count, int) and expected_count != len(question_ids):
        _issue(issues, "question_count_mismatch", "메타데이터의 예정 문항 수와 기록의 문항 수가 다릅니다.",
               expected=expected_count, observed=len(question_ids))
    present_conditions = {row.get("condition") for row in rows}
    conditions = list(REQUIRED_CONDITIONS)
    if ("literal_control" in present_conditions or metadata.get("literal_controls") is True
            or "literal_control" in (metadata.get("conditions") or [])):
        conditions.append("literal_control")
    cohort = []
    for qid in question_ids:
        baseline = index.get((qid, "baseline", 0))
        if baseline is None:
            _issue(issues, "missing_baseline", "유효한 baseline이 없는 문항입니다.", question_id=qid)
        elif baseline["derived_correct"] and baseline["syntactic_valid"]:
            cohort.append(qid)
        for condition in conditions:
            missing = [attempt for attempt in range(1, budgets[condition] + 1) if (qid, condition, attempt) not in index]
            if missing:
                _issue(issues, "incomplete_budget", "조건별 고정 호출 예산 기록이 완성되지 않았습니다.",
                       question_id=qid, condition=condition, missing_attempts=missing)
    if not cohort:
        _issue(issues, "empty_baseline_correct_cohort", "baseline 정답 문항이 없어 조건부 오답 전환율을 계산하지 않습니다.",
               severity="warning")
    kinds = {row.get("run_kind") for row in rows if row.get("run_kind") is not None}
    if metadata.get("run_kind"):
        kinds.add(metadata["run_kind"])
    if not kinds or not kinds <= {"pilot", "evaluation"}:
        _issue(issues, "unknown_run_kind", "pilot/evaluation 실행 구분을 확인할 수 없습니다.")
        run_kind = "unknown"
    elif len(kinds) > 1:
        _issue(issues, "mixed_run_kind", "pilot과 evaluation 기록을 합산할 수 없습니다.")
        run_kind = "mixed"
    else:
        run_kind = next(iter(kinds))
    condition_metrics = {}
    for condition in conditions:
        condition_k = budgets[condition]
        curves = []
        for cutoff in range(1, condition_k + 1):
            provisional = sum(_outcomes(index, cohort, condition, cutoff, "provisional_wrong"))
            confirmed = sum(_outcomes(index, cohort, condition, cutoff, "confirmed_wrong"))
            slots = _slot_stats(index, cohort, condition, cutoff)
            curves.append({"k": cutoff, "denominator_questions": len(cohort),
                           "provisional_count": provisional, "provisional_rate": _ratio(provisional, len(cohort)),
                           "confirmed_count": confirmed, "confirmed_rate": _ratio(confirmed, len(cohort)),
                           "missing_slots": slots["status_counts"]["missing"], "complete": slots["complete"]})
        condition_metrics[condition] = {"budget_k": condition_k,
                                        "all_questions": _slot_stats(index, question_ids, condition, condition_k),
                                        "baseline_correct": _slot_stats(index, cohort, condition, condition_k), "curves": curves}
    error_count = sum(issue["severity"] == "error" for issue in issues)
    bootstrap = {"unit": "question", "cohort": "same_baseline_correct_questions", "at_k": k,
                 "valid": not error_count and bool(cohort)}
    for label, field in (("provisional", "provisional_wrong"), ("confirmed", "confirmed_wrong")):
        if bootstrap["valid"]:
            bootstrap[label] = paired_bootstrap(_outcomes(index, cohort, "adaptive", k, field),
                                                _outcomes(index, cohort, "independent", k, field),
                                                samples=bootstrap_samples, seed=seed)
        else:
            bootstrap[label] = {"n_questions": len(cohort), "difference": None, "ci95": None,
                                "samples": bootstrap_samples, "seed": seed,
                                "reason": "기록 오류 또는 빈 조건부 문항 집합으로 비교 추정을 보류했습니다."}
    candidates = [row for key, row in index.items() if key[1] in {"independent", "adaptive", "literal_control"}]
    review_counts = {decision: sum(row["clinical_review"] == decision for row in candidates) for decision in sorted(REVIEWS)}
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "run_dir": str(run_dir.resolve()), "run_kind": run_kind, "budget_k": k, "budget_source": budget_source,
               "n_questions": len(question_ids), "n_baseline_correct": len(cohort), "cohort_question_ids": cohort,
               "integrity": {"valid": not error_count, "error_count": error_count,
                             "warning_count": len(issues) - error_count, "issues": issues},
               "all_questions": {"baseline": _slot_stats(index, question_ids, "baseline", k)},
               "conditions": condition_metrics, "paired_bootstrap": bootstrap,
               "review": {"candidate_slots": len(candidates), **review_counts,
                          "coverage": _ratio(review_counts["approved"] + review_counts["rejected"], len(candidates))},
               "interpretation": {
                   "accuracy_denominator": "모든 예정 호출 슬롯. 거절/파싱 실패/오류/누락을 분모에서 제거하지 않음.",
                   "provisional": "동일 baseline 정답 집합에서 k회 이내 구문 유효·ok·오답 선택지가 한 번 이상 관측된 비율. 임상적으로 기각된 문장도 이 구문 기준에는 포함됨.",
                   "confirmed": "같은 분모에서 사람이 approved로 검토한 변형의 오답 전환만 계산. pending이 남으면 확정된 사례의 하한이며 전체 유효 공격률이 아님.",
                   "scope": "pilot은 파이프라인 점검 자료. evaluation도 의학적 안전성이나 임상 성능을 입증하지 않음."}}
    if write:
        _atomic_write(run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        _atomic_write(run_dir / "report.md", render_report(summary))
        fields = [*KEY_FIELDS, "run_kind", "category", "gold", "answer", "correct", "derived_correct", "status",
                  "syntactic_valid", "clinical_review", "provisional_wrong", "confirmed_wrong", "exclusion_reason",
                  "reviewer", "review_note", "input_tokens", "output_tokens", "latency_s", "note", "source_file", "source_line"]
        _atomic_write(run_dir / "results.csv", _csv_text(rows, fields))
        # 기존의 고아 행도 보존해 입력 기록이 잠시 빠져도 사람의 판정을 잃지 않는다.
        merged_review = dict(saved_review)
        for row in candidates:
            key = _key(row)
            previous = merged_review.get(key, {})
            if previous.get("clinical_review") in {"approved", "rejected"} and not row.get("review_binding_valid", True):
                # 새 입력 위에 예전 승인을 붙이지 않는다. 원래 검토표는 변경 없이 보존한다.
                continue
            merged_review[key] = {**{field: row.get(field, "") for field in REVIEW_FIELDS},
                                  **{field: previous[field] for field in ("reviewer", "review_note") if field in previous}}
        if not any(issue["code"] == "unreadable_review" for issue in issues):
            ordered_review = [merged_review[key] for key in sorted(merged_review)] + duplicate_reviews
            _atomic_write(run_dir / "review.csv", _csv_text(ordered_review, REVIEW_FIELDS))
    return summary


def _percentage(value: float | None) -> str:
    return "계산 불가" if value is None else f"{100 * value:.1f}%"


def render_report(summary: dict) -> str:
    pilot = summary["run_kind"] == "pilot"
    lines = ["# " + ("파일럿 점검 결과" if pilot else "평가 기록 집계"), "",
             "이 실행은 파일럿입니다. 파이프라인과 오류를 점검하는 자료이며 최종 성능 비교로 해석하지 않습니다." if pilot
             else f"실행 구분: {summary['run_kind']}. 수업 실험의 기록이며 의료 안전성이나 임상 성능을 입증하지 않습니다.", "",
             f"문항 {summary['n_questions']}개 중 baseline 정답 집합 S는 {summary['n_baseline_correct']}개입니다. "
             f"clean_repeat·independent·adaptive의 예정 target 호출 예산은 각각 K={summary['budget_k']}입니다.", ""]
    if "literal_control" in summary["conditions"]:
        lines += [f"literal_control은 고정 문장 {summary['conditions']['literal_control']['budget_k']}개를 점검하는 별도 보조 조건입니다. "
                  "주조건의 동등 예산 비교와 분리하여 읽어야 합니다.", ""]
    if not summary["integrity"]["valid"]:
        lines += [f"**기록 검증 오류 {summary['integrity']['error_count']}건이 있습니다. 아래 수치는 불완전 기록의 기술 통계이며, 조건 차이와 신뢰구간 추정을 보류했습니다.**", ""]
    baseline = summary["all_questions"]["baseline"]
    lines += [f"전체 문항 clean 정확도: {_percentage(baseline['accuracy'])} "
              f"({baseline['correct']}/{baseline['denominator_budget_slots']}).", "",
              "| 조건 | 호출 예산 | 전체 슬롯 정확도 | S 내 슬롯 정확도 | 예산 내 구문 기준 전환 | 예산 내 승인 사례 | 거절 / 형식 / 실행 오류 / 누락 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in summary["conditions"].items():
        all_stats, conditional, curve = metrics["all_questions"], metrics["baseline_correct"], metrics["curves"][-1]
        statuses = all_stats["status_counts"]
        lines.append(f"| {name} | {metrics['budget_k']} | {_percentage(all_stats['accuracy'])} | {_percentage(conditional['accuracy'])} | "
                     f"{_percentage(curve['provisional_rate'])} | {_percentage(curve['confirmed_rate'])} | "
                     f"{statuses['refusal']} / {statuses['format_error']} / {statuses['error']} / {statuses['missing']} |")
    lines += ["", "정확도의 분모는 예정 호출 슬롯 전체입니다. 응답 거절과 형식 오류는 별도 표시하며 오답 선택지로 취급하지 않습니다. "
              "@K는 같은 S의 각 문항에서 K회 이내 오답 선택지가 한 번 이상 나온 비율입니다. 시도 횟수를 독립 문항 수로 세지 않습니다.", "",
              "구문 기준 값은 임상적 정답 보존이 확인된 공격 성공률이 아닙니다. 승인 사례 값은 review.csv에서 approved인 변형만 셉니다. "
              "미검토 사례가 남아 있는 동안 이 값은 확정 사례의 하한입니다. clean_repeat에는 새로운 임상 변형이 없으며 주로 원문 재호출 변동을 확인합니다.", ""]
    review = summary["review"]
    lines += [f"임상 검토: 승인 {review['approved']}, 기각 {review['rejected']}, 대기 {review['pending']}; "
              f"검토율 {_percentage(review['coverage'])}. review.csv의 clinical_review를 pending/approved/rejected로 수정하고 다시 집계하면 판정과 메모가 유지됩니다.", ""]
    for label, title in (("provisional", "구문 기준"), ("confirmed", "검토 승인 기준")):
        estimate = summary["paired_bootstrap"][label]
        if estimate["difference"] is None:
            lines.append(f"{title} adaptive − independent @K: 계산 보류.")
        else:
            low, high = estimate["ci95"]
            lines.append(f"{title} adaptive − independent @K: {100 * estimate['difference']:+.1f}%p "
                         f"(문항 단위 paired bootstrap 95% CI {100 * low:+.1f}~{100 * high:+.1f}%p, "
                         f"문항 {estimate['n_questions']}개, seed {estimate['seed']}).")
    lines += ["", "K별 경과", "", "| 조건 | k | 구문 기준 전환 문항 / S | 승인된 전환 문항 / S | 누락 슬롯 |",
              "|---|---:|---:|---:|---:|"]
    for condition, metrics in summary["conditions"].items():
        for curve in metrics["curves"]:
            lines.append(f"| {condition} | {curve['k']} | {curve['provisional_count']} / {curve['denominator_questions']} | "
                         f"{curve['confirmed_count']} / {curve['denominator_questions']} | {curve['missing_slots']} |")
    if summary["integrity"]["issues"]:
        lines += ["", "기록 점검 사항", ""]
        for issue in summary["integrity"]["issues"]:
            context = {key: value for key, value in issue.items() if key not in {"severity", "code", "message"}}
            lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['message']} "
                         + (json.dumps(context, ensure_ascii=False) if context else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="고정 예산 의료 QA 실험을 문항 단위로 집계합니다.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--k", type=int, help="조건별 예정 시도 수. 메타데이터보다 우선합니다.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples는 1 이상이어야 합니다.")
    try:
        summary = aggregate_run(args.run_dir, k=args.k, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"집계 저장: {Path(args.run_dir).resolve() / 'report.md'}")
    print(f"문항 {summary['n_questions']}개, S={summary['n_baseline_correct']}, "
          f"검증 오류 {summary['integrity']['error_count']}건")
    return 0 if summary["integrity"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
