#!/usr/bin/env python3
"""저장된 전체 실행 결과를 CPU로 재집계한다. 모델 호출·학습·원본 수정 없음."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaim.full_data import input_sha256

ARMS = ("base", "clean", "random", "flip", "near_miss")
KINDS = ("clean", "supports_wrong", "supports_correct", "perturbed")


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_difference(differences, samples=10000):
    """문장 행이 아닌 원본 문제를 같은 쌍으로 재추출한다. 학습 seed 변동은 포함하지 않는다."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20260922)
    draws = []
    for start in range(0, samples, 500):
        indices = rng.integers(0, len(values), size=(min(500, samples - start), len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    return {"delta_pp": float(values.mean() * 100),
            "ci95_pp": (np.quantile(draws, [0.025, 0.975]) * 100).tolist(),
            "source_questions": len(values), "bootstrap_samples": samples}


def summarize(experiment: Path):
    manifest = json.loads((experiment / "experiment.json").read_text())
    cases_path = experiment / "evaluation/test_cases.jsonl"
    cases = read_rows(cases_path)
    by_case = {row["case_id"]: row for row in cases}
    assert len(by_case) == len(cases) == 12730
    case_hashes = {key: input_sha256(row) for key, row in by_case.items()}
    qids = sorted({row["question_id"] for row in cases})
    question = {row["question_id"]: row for row in cases if row["kind"] == "clean" and not row["defended"]}
    paths = {"base": experiment / "evaluation/base_predictions.jsonl"}
    paths.update({arm: experiment / "arms" / arm / "test_predictions.jsonl" for arm in ARMS[1:]})
    predictions = {}
    metrics = {}
    provenance = {"experiment_sha256": sha(experiment / "experiment.json"), "test_cases_sha256": sha(cases_path)}
    for arm, path in paths.items():
        rows = read_rows(path)
        indexed = {row["case_id"]: row for row in rows}
        assert len(indexed) == len(rows) == len(cases) and set(indexed) == set(by_case)
        # 저장된 정오 표시만 믿지 않고 같은 입력·원본 정답·실제 선택을 대조한다.
        for key, row in indexed.items():
            case = by_case[key]
            predicted = max("ABCD", key=lambda letter: row["probabilities"][letter])
            assert predicted == row["predicted"]
            assert row["correct"] == (predicted == case["answer_idx"])
            assert row["answer_idx"] == case["answer_idx"]
            assert row["input_sha256"] == case_hashes[key]
            assert row["kind"] == case["kind"] and row["note"] == case["note"]
            assert row["defended"] == case["defended"]
            if case["note"]:
                assert (case["suggested_answer"] == case["answer_idx"]) == (case["kind"] == "supports_correct")
        predictions[arm] = indexed
        provenance[arm + "_predictions_sha256"] = sha(path)
        stored_path = experiment / ("evaluation/base_summary.json" if arm == "base" else f"arms/{arm}/test_summary.json")
        stored = json.loads(stored_path.read_text())
        stored = stored.get("metrics", stored)
        group_metrics = {}
        for defended in (False, True):
            prompt = "defended" if defended else "neutral"
            clean_by_q = {case["question_id"]: indexed[case["case_id"]] for case in cases
                          if case["kind"] == "clean" and case["defended"] == defended}
            for subset in ("all", "previously_observed", "previously_unseen"):
                cohort = [case for case in cases if case["defended"] == defended and
                          (subset == "all" or case["historical_test_seen"] == (subset == "previously_observed"))]
                for kind in KINDS:
                    selected = [case for case in cohort if (bool(case["note"]) if kind == "perturbed" else case["kind"] == kind)]
                    key = f"{subset}/{prompt}/{kind}"
                    correct = sum(indexed[case["case_id"]]["correct"] for case in selected)
                    entry = {"correct": correct, "n": len(selected), "accuracy": correct / len(selected)}
                    if kind != "clean":
                        originally_correct = [case for case in selected if clean_by_q[case["question_id"]]["correct"]]
                        flips = sum(not indexed[case["case_id"]]["correct"] for case in originally_correct)
                        entry.update({"correct_to_wrong": flips, "clean_correct_cases": len(originally_correct),
                                      "correct_to_wrong_rate": flips / len(originally_correct) if originally_correct else None,
                                      "followed_opinion": sum(indexed[case["case_id"]]["predicted"] == case["suggested_answer"] for case in selected)})
                    assert stored[key]["correct"] == correct and stored[key]["n"] == len(selected)
                    group_metrics[key] = entry
                # 문제마다 의견 네 개 모두에 정답을 유지한 비율. 네 행을 독립 문제로 세지 않는다.
                note_groups = defaultdict(list)
                for case in cohort:
                    if case["note"]:
                        note_groups[case["question_id"]].append(indexed[case["case_id"]]["correct"])
                group_metrics[f"{subset}/{prompt}/all_four_opinions_correct"] = {
                    "correct": sum(all(values) for values in note_groups.values()), "n": len(note_groups),
                    "accuracy": sum(all(values) for values in note_groups.values()) / len(note_groups)}
        metrics[arm] = group_metrics

    # 같은 문제의 평균 정답률을 사용해 세 오답 의견을 함께 재추출한다.
    def values(arm, kind, subset="all"):
        selected_qids = [qid for qid in qids if subset == "all" or not question[qid]["historical_test_seen"]]
        groups = defaultdict(list)
        for case in cases:
            if not case["defended"] and (bool(case["note"]) if kind == "perturbed" else case["kind"] == kind):
                groups[case["question_id"]].append(predictions[arm][case["case_id"]]["correct"])
        return np.array([np.mean(groups[qid]) for qid in selected_qids])

    comparisons = {}
    for left, right, kind in (("clean", "base", "clean"), ("random", "clean", "supports_wrong"),
                              ("random", "clean", "supports_correct"), ("near_miss", "random", "supports_wrong"),
                              ("near_miss", "flip", "supports_wrong"), ("near_miss", "random", "supports_correct"),
                              ("near_miss", "clean", "perturbed")):
        for subset in ("all", "previously_unseen"):
            comparisons[f"{subset}/{left}-minus-{right}/{kind}"] = bootstrap_difference(values(left, kind, subset) - values(right, kind, subset))

    training = {}
    selections = {}
    for arm in ARMS[1:]:
        path = experiment / "training" / f"{arm}.jsonl"
        rows = read_rows(path)
        notes = {row["question_id"]: row for row in rows if row["note"]}
        selections[arm] = notes
        state = json.loads((experiment / "arms" / arm / "train_state.json").read_text())
        training[arm] = {"source_questions": len({row["question_id"] for row in rows}), "rows": len(rows),
                         "opinion_rows": len(notes), "opinions_supporting_gold": sum(row["suggested_answer"] == row["answer_idx"] for row in notes.values()),
                         "completed_epochs": state["completed_epochs"], "processed_rows": state["processed_rows"],
                         "optimizer_steps": state["optimizer_steps"], "microbatch": state["microbatch"],
                         "peak_reserved_gib": state["peak_reserved_bytes"] / 1024 ** 3,
                         "peak_allocated_gib": state["peak_allocated_bytes"] / 1024 ** 3,
                         "training_file_sha256": sha(path)}
        if arm not in ("clean", "random"):
            training[arm]["different_from_random"] = sum(row["candidate_index"] != selections["random"][qid]["candidate_index"] for qid, row in notes.items())

    # 추가 학습이 아니라, 저장된 원문 점수에서 권유된 선택지만 배제하는 진단용 규칙이다.
    exclusion = {}
    for source in ("base", "clean"):
        source_clean = {case["question_id"]: predictions[source][case["case_id"]] for case in cases if case["kind"] == "clean" and not case["defended"]}
        stats = {}
        for kind in ("supports_wrong", "supports_correct"):
            selected = [case for case in cases if not case["defended"] and case["kind"] == kind]
            n_correct = 0
            for case in selected:
                probabilities = source_clean[case["question_id"]]["probabilities"]
                answer = max((letter for letter in "ABCD" if letter != case["suggested_answer"]), key=probabilities.get)
                n_correct += answer == case["answer_idx"]
            stats[kind] = {"correct": n_correct, "n": len(selected), "accuracy": n_correct / len(selected)}
        exclusion[source + "_exclude_suggested_letter"] = stats

    # 전체 빈도는 위에서 집계한다. 아래 사례는 실패 동작을 설명할 목적으로 고른 실제 기록이다.
    examples = []
    for qid in qids:
        original = question[qid]
        support = next(case for case in cases if case["question_id"] == qid and case["kind"] == "supports_correct" and not case["defended"])
        if (not original["historical_test_seen"] and
            all(predictions[arm][original["case_id"]]["correct"] for arm in ARMS[1:]) and
            predictions["clean"][support["case_id"]]["correct"] and
            all(not predictions[arm][support["case_id"]]["correct"] for arm in ("random", "flip", "near_miss"))):
            examples.append({"question_id": qid, "question": original["question"], "options": original["options"],
                             "gold": original["answer_idx"], "note": support["note"], "case_id": support["case_id"],
                             "predictions": {arm: {"original": predictions[arm][original["case_id"]]["predicted"],
                                                   "with_correct_opinion": predictions[arm][support["case_id"]]["predicted"],
                                                   "original_gold_probability": predictions[arm][original["case_id"]]["gold_probability"],
                                                   "opinion_gold_probability": predictions[arm][support["case_id"]]["gold_probability"]} for arm in ARMS}})
            if len(examples) == 3:
                break
    return {"experiment": experiment.name, "manifest_counts": manifest["counts"], "model": manifest["model"],
            "revision": manifest["revision"], "training": training, "metrics": metrics,
            "selection": json.loads((experiment / "training/selection_counts.json").read_text()),
            "paired_bootstrap": comparisons, "posthoc_exclusion_diagnostic": exclusion, "examples": examples,
            "provenance": provenance, "analysis_script_sha256": sha(Path(__file__)),
            "validation": {"matched_case_ids_and_input_hashes": True, "correctness_recomputed_from_choice_logits": True,
                           "stored_summary_counts_matched": True, "prediction_rows_checked": len(cases) * len(ARMS)},
            "uncertainty_scope": "Exploratory paired question bootstrap, 10000 resamples, seed20260922. Conditional on one trained checkpoint per arm; no training-seed uncertainty or multiple-comparison correction."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "validation": result["validation"]}, ensure_ascii=False))
