"""코드 전용 fixture와 모델/프로세스 mock으로 재평가 경계를 검사한다.

실제 의료 데이터, GPU, 학습 adapter 또는 모델 응답을 사용하지 않는다.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gaim import replay
from gaim.metrics import REVIEW_FIELDS, review_content_hash


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture_case(condition="baseline", attempt=0, review="approved", **changes):
    note = "" if condition == "baseline" else "Code-only background fixture."
    record = {
        "question_id": "CODE_ONLY_EVAL", "condition": condition, "attempt": attempt,
        "gold": "B", "answer": "B", "correct": True, "status": "ok",
        "syntactic_valid": True, "clinical_review": review,
        "note": note, "candidate_note": note,
        "question_sha256": hashlib.sha256(b"EXPLICIT CODE-ONLY QUESTION").hexdigest(),
        "target_messages": [
            {"role": "system", "content": "EXPLICIT CODE-ONLY FIXTURE"},
            {"role": "user", "content": f"Immutable fixture for {condition}/{attempt}"},
        ],
    }
    record.update(changes)
    return record


class ReplayFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="gaim-code-only-replay-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.training, self.output = (
            self.root / name for name in ("source", "training", "output")
        )
        for directory in (self.source, self.training, self.output):
            directory.mkdir()
        self.manifest = {
            "target_model": "NOT_A_REAL_MODEL", "target_revision": "fixture-revision",
            "question_ids": ["CODE_ONLY_EVAL"],
            "config": {"model": "NOT_A_REAL_MODEL", "split": "dev", "seed": 5,
                       "max_input_tokens": 1024, "target_temperature": 0,
                       "target_max_new_tokens": 128},
        }
        self.cases = [fixture_case(), fixture_case("adaptive", 1)]
        self.save_source()
        (self.source / "COMPLETE").write_text("code-only fixture\n")
        self.training_manifest = {
            "model": "NOT_A_REAL_MODEL", "target_revision": "fixture-revision",
            "status": "complete", "source_question_ids": ["CODE_ONLY_TRAIN"],
        }
        (self.training / "training.json").write_text(json.dumps(self.training_manifest))
        for variant in replay.VARIANTS:
            directory = self.training / variant
            adapter = directory / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "NOT_A_REAL_MODEL", "r": 2,
            }))
            (adapter / "adapter_model.safetensors").write_bytes(b"NOT REAL WEIGHTS")
            write_jsonl(directory / "dataset.jsonl", [{"question_id": "CODE_ONLY_TRAIN"}])

    def save_source(self):
        (self.source / "run.json").write_text(json.dumps(self.manifest))
        write_jsonl(self.source / "worker_0.jsonl", self.cases)

    def reviews(self, rows):
        cases = {(case["question_id"], case["condition"], case["attempt"]): case
                 for case in self.cases}
        with (self.source / "review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for row in rows:
                key = row["question_id"], row["condition"], int(row["attempt"])
                bound = {field: cases[key].get(field, "") for field in REVIEW_FIELDS}
                bound.update(row)
                bound["content_sha256"] = review_content_hash(bound)
                writer.writerow({field: json.dumps(value) if isinstance(value, (dict, list)) else value
                                 for field, value in bound.items()})

    def args(self, worker=0):
        return SimpleNamespace(source_run=self.source, training_dir=self.training,
                               output_dir=self.output, worker=worker, allow_provisional=False)

    def run_main(self, *, allow_provisional=False):
        """CLI 제어 흐름은 실행하되 GPU 조회·모델·자식 프로세스는 대체한다."""
        argv = ["replay", "--source-run", str(self.source), "--training-dir", str(self.training),
                "--output-dir", str(self.output)]
        if allow_provisional:
            argv.append("--allow-provisional")
        model_calls = []

        class FixtureModel:
            def __init__(self, *args, **kwargs):
                self.adapter = kwargs.get("adapter_path")

            def generate(self, messages, **kwargs):
                model_calls.append((self.adapter, copy.deepcopy(messages), kwargs))
                return {"raw": '{"answer":"B","explanation":"Code-only fixture."}',
                        "input_tokens": 1, "output_tokens": 1, "latency_s": 0,
                        "seed": kwargs["seed"], "hit_token_limit": False}

        def fake_process(command, **kwargs):
            worker_index = int(command[command.index("--worker") + 1])
            cases = replay.selected_records(self.source)
            if not allow_provisional:
                cases = [r for r in cases if r["condition"] == "baseline"
                         or r["clinical_review"] == "approved"]
            replay.worker(self.args(worker_index), self.manifest, cases)
            return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

        captured = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            stack.enter_context(patch.object(replay, "ROOT", self.root))
            stack.enter_context(patch.object(replay, "LocalModel", FixtureModel))
            stack.enter_context(patch.object(replay.subprocess, "Popen", side_effect=fake_process))
            stack.enter_context(patch.object(replay.subprocess, "check_output",
                                           return_value="4, 0 MiB\n5, 0 MiB\n6, 0 MiB\n7, 0 MiB\n"))
            stack.enter_context(redirect_stdout(captured))
            replay.main()
        return model_calls, captured.getvalue()


class SelectionTests(ReplayFixture):
    def test_matching_bound_approval_applies_to_pending_case(self):
        self.cases[1]["clinical_review"] = "pending"
        self.save_source()
        self.reviews([{"question_id": "CODE_ONLY_EVAL", "condition": "adaptive",
                       "attempt": 1, "clinical_review": "approved"}])
        selected = replay.selected_records(self.source)
        approved = [row for row in selected if row["condition"] == "adaptive"]
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["clinical_review"], "approved")

    def test_stale_approval_cannot_apply_to_changed_note_with_same_case_key(self):
        self.reviews([{"question_id": "CODE_ONLY_EVAL", "condition": "adaptive",
                       "attempt": 1, "clinical_review": "approved"}])
        self.cases[1].update(note="Different code-only note.",
                             candidate_note="Different code-only note.")
        self.cases[1]["target_messages"][1]["content"] = "Different code-only target input."
        self.save_source()
        with self.assertRaisesRegex(ValueError, "different content"):
            replay.selected_records(self.source)

    def test_manual_rejection_overrides_worker_approval(self):
        self.reviews([{"question_id": "CODE_ONLY_EVAL", "condition": "adaptive",
                       "attempt": 1, "clinical_review": "rejected"}])
        self.assertEqual([r["condition"] for r in replay.selected_records(self.source)],
                         ["baseline"])

    def test_duplicate_review_decisions_fail(self):
        row = {"question_id": "CODE_ONLY_EVAL", "condition": "adaptive",
               "attempt": 1, "clinical_review": "approved"}
        self.reviews([row, row])
        with self.assertRaises(ValueError):
            replay.selected_records(self.source)

    def test_invalid_candidates_and_clean_repeats_are_excluded(self):
        self.cases.extend([fixture_case("independent", 1, syntactic_valid=False),
                           fixture_case("clean_repeat", 1)])
        self.save_source()
        self.assertEqual(len(replay.selected_records(self.source)), 2)

    def test_operational_error_is_not_a_medical_error(self):
        self.cases[1].update(status="error", answer=None, correct=False)
        self.save_source()
        with self.assertRaises(ValueError):
            replay.selected_records(self.source)


class ReplayProtocolTests(ReplayFixture):
    def test_replay_records_explicit_gpu_mapping(self):
        with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "3,0,2,1"}, clear=True):
            self.run_main()
        manifest = json.loads((self.output / "replay.json").read_text())
        self.assertEqual(manifest["physical_gpus"], [3, 0, 2, 1])

    def test_replay_resume_rejects_changed_gpu_order(self):
        with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "0,1,2,3"}, clear=True):
            self.run_main()
        with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "3,2,1,0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.run_main()

    def test_every_adapter_receives_exact_saved_input_and_same_generation_settings(self):
        original = copy.deepcopy(self.cases)
        calls, output = self.run_main()
        self.assertEqual(self.cases, original)
        expected = replay.selected_records(self.source)
        for variant in replay.VARIANTS:
            variant_calls = [call for call in calls if Path(call[0]).parent.name == variant]
            self.assertEqual([call[1] for call in variant_calls],
                             [case["target_messages"] for case in expected])
            self.assertEqual([call[2] for call in variant_calls], [call[2] for call in calls[:2]])
        result = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(result["untrained"], replay.summarize(expected))
        self.assertTrue((self.output / "COMPLETE").exists())

    def test_pending_review_requires_explicit_provisional_mode(self):
        self.cases[1]["clinical_review"] = "pending"
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_main()
        calls, _ = self.run_main(allow_provisional=True)
        self.assertEqual(len(calls), 2 * len(replay.VARIANTS))
        manifest = json.loads((self.output / "replay.json").read_text())
        self.assertTrue(manifest["allow_provisional"])

    def test_train_evaluation_id_overlap_is_rejected(self):
        write_jsonl(self.training / "adaptive" / "dataset.jsonl",
                    [{"question_id": "CODE_ONLY_EVAL"}])
        with self.assertRaises(ValueError):
            self.run_main()

    def test_training_model_mismatch_is_rejected(self):
        self.training_manifest["model"] = "ANOTHER_NOT_REAL_MODEL"
        (self.training / "training.json").write_text(json.dumps(self.training_manifest))
        with self.assertRaises(ValueError):
            self.run_main()

    def test_training_revision_mismatch_is_rejected(self):
        self.training_manifest["target_revision"] = "different-code-only-revision"
        (self.training / "training.json").write_text(json.dumps(self.training_manifest))
        with self.assertRaises(ValueError):
            self.run_main()

    def test_each_adapter_requires_its_own_training_provenance(self):
        (self.training / "random" / "dataset.jsonl").unlink()
        with self.assertRaises(ValueError):
            self.run_main()

    def test_case_ids_must_belong_to_source_manifest(self):
        self.cases[1]["question_id"] = "CODE_ONLY_TRAIN"
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_main()

    def test_model_revision_change_cannot_reuse_existing_output(self):
        self.run_main()
        self.manifest["target_revision"] = "another-fixture-revision"
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_main()

    def test_changed_saved_input_cannot_reuse_existing_output(self):
        self.run_main()
        self.cases[1]["target_messages"][1]["content"] = "Changed code-only input, same case key."
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_main()

    def test_changed_adapter_weights_cannot_reuse_existing_output(self):
        self.run_main()
        weights = self.training / "adaptive" / "adapter" / "adapter_model.safetensors"
        weights.write_bytes(b"DIFFERENT CODE-ONLY FIXTURE WEIGHTS")
        with self.assertRaises(ValueError):
            self.run_main()

    def test_changed_target_generation_settings_cannot_reuse_existing_output(self):
        self.run_main()
        self.manifest["config"]["target_max_new_tokens"] = 64
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_main()

    def test_existing_error_output_is_not_treated_as_completed(self):
        self.run_main()
        path = self.output / "adaptive.jsonl"
        records = replay.read_jsonl(path)
        records[0].update(status="error", answer=None, correct=False)
        write_jsonl(path, records)
        with self.assertRaises(ValueError):
            self.run_main()

    def test_existing_wrong_gold_or_variant_cannot_be_silently_reused(self):
        self.run_main()
        path = self.output / "adaptive.jsonl"
        records = replay.read_jsonl(path)
        records[0].update(gold="D", variant="random")
        write_jsonl(path, records)
        with self.assertRaises(ValueError):
            self.run_main()

    def test_worker_keeps_wrong_choice_refusal_and_format_error_distinct(self):
        self.cases.append(fixture_case("independent", 1))
        outputs = [
            {"raw": '{"answer":"C","explanation":"B appeared first in this explanation."}'},
            {"raw": '{"answer":null,"explanation":"I decline to answer."}'},
            {"raw": "B"},
        ]
        with patch.object(replay, "LocalModel") as model_type, redirect_stdout(io.StringIO()):
            model_type.return_value.generate.side_effect = outputs
            replay.worker(self.args(), self.manifest, self.cases)
        records = replay.read_jsonl(self.output / f"{replay.VARIANTS[0]}.jsonl")
        self.assertEqual([record["status"] for record in records],
                         ["ok", "refusal", "format_error"])
        self.assertEqual([record["answer"] for record in records], ["C", None, None])
        self.assertEqual([record["correct"] for record in records], [False, False, False])
        summary = replay.summarize(records)
        self.assertEqual(summary["refusals"], 1)
        self.assertEqual(summary["format_errors"], 1)

    def test_worker_saves_operational_error_before_aborting(self):
        with patch.object(replay, "LocalModel") as model_type:
            model_type.return_value.generate.side_effect = RuntimeError("CODE_ONLY_FAKE_FAILURE")
            with self.assertRaises(RuntimeError):
                replay.worker(self.args(), self.manifest, self.cases)
        records = replay.read_jsonl(self.output / f"{replay.VARIANTS[0]}.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertIn("CODE_ONLY_FAKE_FAILURE", records[0]["error"])


if __name__ == "__main__":
    unittest.main()
