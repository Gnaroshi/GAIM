"""작은 명시적 fixtures만 사용하며 실제 실험 자료를 만들지 않는 집계 검증."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gaim.metrics import aggregate_run, main, paired_bootstrap, review_content_hash


def record(qid, condition, attempt, answer="A", **changes):
    row = {"question_id": qid, "condition": condition, "attempt": attempt,
           "answer": answer, "gold": "A", "correct": answer == "A", "status": "ok",
           "note": "explicit test fixture, not an experiment", "syntactic_valid": True,
           "clinical_review": "pending", "category": "fixture", "input_tokens": 10,
           "output_tokens": 1, "latency_s": 0.1, "run_kind": "pilot"}
    row.update(changes)
    row.setdefault("candidate_note", row["note"])
    row.setdefault("question_sha256", hashlib.sha256(qid.encode()).hexdigest())
    row.setdefault("target_messages", [{"role": "user", "content": f"fixture {qid}: {row['note']}"}])
    return row


def complete_records(qids=("q1", "q2"), k=2):
    return [row for qid in qids for row in
            [record(qid, "baseline", 0),
             *[record(qid, condition, attempt) for condition in ("clean_repeat", "independent", "adaptive")
               for attempt in range(1, k + 1)]]]


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_run(self, rows, k=2, qids=("q1", "q2")):
        (self.run_dir / "run.json").write_text(json.dumps({"k": k, "run_kind": "pilot", "question_ids": list(qids)}))
        (self.run_dir / "worker_0.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def aggregate(self, **kwargs):
        return aggregate_run(self.run_dir, bootstrap_samples=200, seed=3, **kwargs)

    def test_repeated_attempts_are_not_independent_questions(self):
        rows = complete_records(k=4)
        for row in rows:
            if (row["question_id"], row["condition"]) in {("q1", "adaptive"), ("q2", "independent")}:
                row.update(answer="B", correct=False)
        self.write_run(rows, k=4)
        result = self.aggregate()
        self.assertTrue(result["integrity"]["valid"])
        for condition in ("adaptive", "independent"):
            curve = result["conditions"][condition]["curves"][-1]
            self.assertEqual(curve["denominator_questions"], 2)
            self.assertEqual(curve["provisional_count"], 1)
            self.assertEqual(curve["provisional_rate"], 0.5)
        estimate = result["paired_bootstrap"]["provisional"]
        self.assertEqual(estimate["n_questions"], 2)
        self.assertEqual(estimate["difference"], 0)
        self.assertEqual(estimate, paired_bootstrap([1, 0], [0, 1], samples=200, seed=3))
        self.assertEqual(result["conditions"]["adaptive"]["all_questions"]["denominator_budget_slots"], 8)

    def test_missing_attempt_keeps_denominator_and_blocks_comparison(self):
        rows = complete_records()
        rows = [row for row in rows if (row["question_id"], row["condition"], row["attempt"]) != ("q1", "adaptive", 2)]
        self.write_run(rows)
        result = self.aggregate()
        stats = result["conditions"]["adaptive"]["all_questions"]
        self.assertEqual(stats["denominator_budget_slots"], 4)
        self.assertEqual(stats["accuracy"], 0.75)
        self.assertEqual(stats["status_counts"]["missing"], 1)
        self.assertFalse(result["integrity"]["valid"])
        self.assertIsNone(result["paired_bootstrap"]["provisional"]["difference"])
        self.assertIn("기록 검증 오류", (self.run_dir / "report.md").read_text())

    def test_refusal_format_and_execution_errors_are_not_wrong_options(self):
        rows = complete_records()
        changes = {("q1", 1): "refusal", ("q1", 2): "format_error", ("q2", 1): "error"}
        for row in rows:
            key = (row["question_id"], row["attempt"])
            if row["condition"] == "independent" and key in changes:
                row.update(status=changes[key], answer=None, correct=False)
        self.write_run(rows)
        result = self.aggregate()
        stats = result["conditions"]["independent"]["all_questions"]
        self.assertEqual(stats["accuracy"], 0.25)
        self.assertEqual(stats["status_counts"], {"error": 1, "format_error": 1, "ok": 1, "refusal": 1, "missing": 0})
        self.assertEqual(result["conditions"]["independent"]["curves"][-1]["provisional_count"], 0)
        self.assertFalse(result["integrity"]["valid"])
        self.assertIn("target_operational_error", {issue["code"] for issue in result["integrity"]["issues"]})

    def test_invalid_syntax_does_not_become_provisional_or_confirmed_flip(self):
        rows = complete_records()
        for row in rows:
            if row["condition"] == "adaptive":
                row.update(answer="B", correct=False, syntactic_valid=False, clinical_review="approved")
        self.write_run(rows)
        result = self.aggregate()
        curve = result["conditions"]["adaptive"]["curves"][-1]
        self.assertEqual(curve["provisional_count"], 0)
        self.assertEqual(curve["confirmed_count"], 0)
        self.assertEqual(result["conditions"]["adaptive"]["all_questions"]["syntactic_invalid"], 4)

    def test_empty_baseline_correct_cohort_is_undefined_not_zero(self):
        rows = complete_records()
        for row in rows:
            if row["condition"] == "baseline":
                row.update(answer="B", correct=False)
        self.write_run(rows)
        result = self.aggregate()
        self.assertEqual(result["n_baseline_correct"], 0)
        self.assertIsNone(result["conditions"]["adaptive"]["curves"][-1]["provisional_rate"])
        self.assertIsNone(result["paired_bootstrap"]["provisional"]["ci95"])
        self.assertEqual(result["all_questions"]["baseline"]["accuracy"], 0)

    def test_duplicate_baselines_are_quarantined_not_arbitrarily_selected(self):
        rows = complete_records()
        rows.append(record("q1", "baseline", 0, answer="B"))
        self.write_run(rows)
        result = self.aggregate()
        codes = {issue["code"] for issue in result["integrity"]["issues"]}
        self.assertIn("duplicate_key", codes)
        self.assertIn("missing_baseline", codes)
        self.assertEqual(result["cohort_question_ids"], ["q2"])
        self.assertIsNone(result["paired_bootstrap"]["provisional"]["difference"])
        with (self.run_dir / "results.csv").open() as handle:
            excluded = [row for row in csv.DictReader(handle) if row["exclusion_reason"] == "duplicate_key"]
        self.assertEqual(len(excluded), 2)

    def test_answer_flag_and_gold_disagreements_are_visible(self):
        rows = complete_records()
        for row in rows:
            if (row["question_id"], row["condition"], row["attempt"]) == ("q1", "baseline", 0):
                row.update(answer="B", correct=True)
            if (row["question_id"], row["condition"], row["attempt"]) == ("q2", "adaptive", 1):
                row.update(gold="B", answer="B", correct=True)
        self.write_run(rows)
        result = self.aggregate()
        codes = {issue["code"] for issue in result["integrity"]["issues"]}
        self.assertTrue({"correct_flag_mismatch", "gold_disagreement", "incomplete_budget"} <= codes)
        self.assertEqual(result["cohort_question_ids"], ["q2"])
        self.assertEqual(result["all_questions"]["baseline"]["accuracy"], 0.5)

    def test_manual_approved_review_and_note_survive_reaggregation(self):
        rows = complete_records()
        for row in rows:
            if (row["question_id"], row["condition"], row["attempt"]) == ("q1", "adaptive", 1):
                row.update(answer="B", correct=False)
        self.write_run(rows)
        first = self.aggregate()
        self.assertEqual(first["conditions"]["adaptive"]["curves"][-1]["confirmed_count"], 0)
        review_path = self.run_dir / "review.csv"
        with review_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            review_rows = list(reader)
        for row in review_rows:
            if (row["question_id"], row["condition"], row["attempt"]) == ("q1", "adaptive", "1"):
                row.update(clinical_review="approved", reviewer="fixture reviewer", review_note="manual fixture judgement")
        with review_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(review_rows)
        second = self.aggregate()
        third = self.aggregate()
        self.assertEqual(second["conditions"]["adaptive"]["curves"][-1]["confirmed_count"], 1)
        self.assertEqual(third["review"]["approved"], 1)
        self.assertIn("manual fixture judgement", review_path.read_text())
        self.assertIn("파일럿", (self.run_dir / "report.md").read_text())

    def test_unknown_budget_and_malformed_json_are_not_silently_accepted(self):
        self.write_run(complete_records())
        (self.run_dir / "run.json").unlink()
        with (self.run_dir / "worker_0.jsonl").open("a") as handle:
            handle.write("{\"truncated\":\n")
        result = self.aggregate()
        codes = {issue["code"] for issue in result["integrity"]["issues"]}
        self.assertTrue({"unknown_planned_budget", "malformed_jsonl"} <= codes)
        self.assertFalse(result["paired_bootstrap"]["valid"])

    def test_cli_writes_reports_but_returns_nonzero_for_incomplete_run(self):
        self.write_run([record("q1", "baseline", 0)], qids=("q1",))
        exit_code = main(["--run-dir", str(self.run_dir), "--k", "2", "--bootstrap-samples", "10"])
        self.assertEqual(exit_code, 2)
        self.assertTrue(all((self.run_dir / filename).exists()
                            for filename in ("summary.json", "report.md", "results.csv", "review.csv")))

    def test_literal_control_uses_three_fixed_slots_when_primary_k_is_two(self):
        rows = complete_records()
        rows.extend(record(qid, "literal_control", attempt) for qid in ("q1", "q2") for attempt in (1, 2, 3))
        self.write_run(rows)
        result = self.aggregate()
        self.assertTrue(result["integrity"]["valid"])
        self.assertEqual(result["conditions"]["literal_control"]["budget_k"], 3)
        self.assertEqual(result["conditions"]["literal_control"]["all_questions"]["denominator_budget_slots"], 6)
        self.assertEqual(result["conditions"]["adaptive"]["all_questions"]["denominator_budget_slots"], 4)

    def test_generator_error_blocks_completion_even_when_target_rows_exist(self):
        self.write_run(complete_records())
        (self.run_dir / "generations_0.jsonl").write_text(json.dumps({"question_id": "q1", "error": "fixture OOM"}) + "\n")
        result = self.aggregate()
        self.assertFalse(result["integrity"]["valid"])
        self.assertIsNone(result["paired_bootstrap"]["provisional"]["difference"])
        self.assertIn("generator_operational_error", {issue["code"] for issue in result["integrity"]["issues"]})

    def test_review_hash_normalizes_csv_json_and_rejects_changed_content(self):
        rows = complete_records()
        target = next(row for row in rows if (row["question_id"], row["condition"], row["attempt"]) == ("q1", "adaptive", 1))
        target.update(answer="B", correct=False)
        self.write_run(rows)
        self.aggregate()
        path = self.run_dir / "review.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, reviews = reader.fieldnames, list(reader)
        saved = next(row for row in reviews if (row["question_id"], row["condition"], row["attempt"]) == ("q1", "adaptive", "1"))
        self.assertEqual(saved["content_sha256"], review_content_hash(target))
        self.assertEqual(review_content_hash(saved), review_content_hash(target))
        saved["clinical_review"] = "approved"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(reviews)
        target["note"] = "Different unreviewed note"
        self.write_run(rows)
        result = self.aggregate()
        self.assertIn("review_content_mismatch", {issue["code"] for issue in result["integrity"]["issues"]})
        self.assertEqual(result["conditions"]["adaptive"]["curves"][-1]["confirmed_count"], 0)
        self.assertIn(saved["content_sha256"], path.read_text())
        self.assertNotIn("Different unreviewed note", path.read_text())

    def test_legacy_approved_without_binding_is_not_trusted(self):
        rows = complete_records()
        rows[0]["correct"] = True
        self.write_run(rows)
        path = self.run_dir / "review.csv"
        path.write_text("question_id,condition,attempt,clinical_review\nq1,adaptive,1,approved\n")
        result = self.aggregate()
        self.assertIn("review_binding_missing", {issue["code"] for issue in result["integrity"]["issues"]})
        self.assertFalse(result["integrity"]["valid"])
        self.assertEqual(result["review"]["approved"], 0)

    def test_legacy_full_binding_without_hash_is_accepted_and_upgraded(self):
        self.write_run(complete_records())
        self.aggregate()
        path = self.run_dir / "review.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = [field for field in reader.fieldnames if field != "content_sha256"]
            reviews = list(reader)
        reviews[0]["clinical_review"] = "approved"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(reviews)
        result = self.aggregate()
        self.assertTrue(result["integrity"]["valid"])
        self.assertEqual(result["review"]["approved"], 1)
        self.assertIn("content_sha256", path.read_text())


if __name__ == "__main__":
    unittest.main()
