"""GPU/모델 없이 명시적 가짜 문항으로 학습 자료의 출처·공정성·loss mask를 검증."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gaim.perturbations import build_target_messages, literal_controls
from gaim.metrics import review_content_hash
from gaim.training import ARMS, build_training_data, prepare_training, tokenize_example


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{row['role']}>" + row["content"] + "<end>" for row in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.items = [{"id": qid, "question": f"Explicit fictional fixture {qid}?", "options": {
            "A": "Alpha option", "B": "Beta option", "C": "Gamma option", "D": "Delta option"},
            "answer_idx": "A", "source_split": "train"} for qid in ("q1", "q2")]
        for item in self.items:
            item["question_sha256"] = hashlib.sha256(item["question"].encode()).hexdigest()
        self.manifest = {"config": {"split": "train", "k": 2, "literal_controls": True}, "run_kind": "pilot", "k": 2,
                         "target_model": "fixture/no-model", "target_revision": "a" * 40,
                         "question_ids": ["q1", "q2"], "dataset_manifest_sha256": "fixture",
                         "dataset_manifest": {"selection": {"sampled_ids": {"train": ["q1", "q2"], "dev": ["dev"], "test": ["test"]}}}}
        self.rows = []
        for item in self.items:
            schedule = [("baseline", 0, ""),
                        *[(condition, attempt, "The patient enjoys collecting postcards." if condition != "clean_repeat" else "")
                          for condition in ("clean_repeat", "independent", "adaptive") for attempt in (1, 2)],
                        *[("literal_control", attempt, control["note"]) for attempt, control in enumerate(literal_controls(item), 1)]]
            for condition, attempt, note in schedule:
                answer = "B" if condition in {"independent", "adaptive"} and attempt == 2 else "A"
                self.rows.append({"question_id": item["id"], "condition": condition, "attempt": attempt,
                                  "note": note, "candidate_note": note, "target_messages": build_target_messages(item, note),
                                  "question_sha256": item["question_sha256"], "syntactic_valid": True,
                                  "clinical_review": "pending", "answer": answer, "gold": "A", "correct": answer == "A",
                                  "status": "ok", "run_kind": "pilot", "category": "fixture"})
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.source / "run.json").write_text(json.dumps(self.manifest))
        (self.source / "questions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in self.items))
        (self.source / "worker_0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def review(self, decision="approved", rejected_condition=None):
        with (self.source / "review.csv").open("w", newline="") as handle:
            binding = ["question_id", "condition", "attempt", "gold", "note", "candidate_note", "question_sha256", "target_messages"]
            writer = csv.DictWriter(handle, fieldnames=[*binding, "clinical_review", "content_sha256"])
            writer.writeheader()
            for row in self.rows:
                if row["condition"] in {"independent", "adaptive", "literal_control"}:
                    writer.writerow({key: json.dumps(row[key]) if key == "target_messages" else row[key] for key in binding} |
                                    {"clinical_review": "rejected" if row["condition"] == rejected_condition else decision,
                                     "content_sha256": review_content_hash(row)})

    def test_strict_excludes_pending_and_exploratory_flag_explicitly_marks_it(self):
        with self.assertRaisesRegex(ValueError, "공통 문항"):
            build_training_data(self.source)
        datasets, info = build_training_data(self.source, allow_provisional=True)
        self.assertEqual(info["claim_scope"], "unreviewed_exploratory")
        self.assertEqual(info["provisional_selected_notes"], 6)
        self.assertEqual({arm: len(rows) for arm, rows in datasets.items()}, {arm: 4 for arm in ARMS})

    def test_same_question_cohort_equal_rows_first_actual_wrong_and_gold_only(self):
        self.review()
        datasets, info = build_training_data(self.source)
        self.assertEqual(info["question_ids"], ["q1", "q2"])
        self.assertEqual(info["provisional_selected_notes"], 0)
        for arm, rows in datasets.items():
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["question_id"] for row in rows], ["q1", "q1", "q2", "q2"])
            for row in rows:
                self.assertEqual(json.loads(row["completion"]), {"answer": "A", "explanation": ""})
                if arm in ("independent", "adaptive") and row["variant"] == "additional":
                    self.assertEqual(row["selected_attempt"], 2)
        self.assertTrue(all(not row["note"] for row in datasets["clean"]))
        self.assertEqual([row["messages"] for row in datasets["clean"]][::2],
                         [row["messages"] for row in datasets["clean"]][1::2])

    def test_rejected_notes_are_excluded_even_with_provisional_flag(self):
        self.review(rejected_condition="adaptive")
        with self.assertRaisesRegex(ValueError, "공통 문항"):
            build_training_data(self.source, allow_provisional=True)

    def test_dev_source_cannot_be_mislabeled_as_train_by_source_split(self):
        self.manifest["config"]["split"] = "dev"
        self.save()
        with self.assertRaisesRegex(ValueError, "split=train"):
            build_training_data(self.source, allow_provisional=True)

    def test_heldout_membership_and_changed_prompt_are_rejected(self):
        self.manifest["dataset_manifest"]["selection"]["sampled_ids"]["dev"] = ["q1"]
        self.save()
        with self.assertRaisesRegex(ValueError, "dev/test"):
            build_training_data(self.source, allow_provisional=True)
        self.manifest["dataset_manifest"]["selection"]["sampled_ids"]["dev"] = []
        self.rows[3]["target_messages"][1]["content"] = "Changed original clinical question"
        self.save()
        with self.assertRaisesRegex(ValueError, "실제 target 입력"):
            build_training_data(self.source, allow_provisional=True)

    def test_random_choice_does_not_change_when_observed_answers_change(self):
        first, _ = build_training_data(self.source, allow_provisional=True)
        for row in self.rows:
            if row["condition"] == "literal_control":
                row.update(answer="D", correct=False)
        self.save()
        second, _ = build_training_data(self.source, allow_provisional=True)
        self.assertEqual([row["note"] for row in first["random"]], [row["note"] for row in second["random"]])

    def test_common_cohort_drops_item_from_every_arm_when_one_arm_is_invalid(self):
        for row in self.rows:
            if row["question_id"] == "q1" and row["condition"] == "adaptive":
                row["syntactic_valid"] = False
        self.save()
        datasets, info = build_training_data(self.source, allow_provisional=True)
        self.assertEqual(info["question_ids"], ["q2"])
        self.assertEqual(len(info["excluded"]), 1)
        self.assertTrue(all([row["question_id"] for row in rows] == ["q2", "q2"] for rows in datasets.values()))

    def test_prepare_preserves_model_revision_and_refuses_output_overwrite(self):
        output = self.root / "training"
        info = prepare_training(self.source, output, steps=20, allow_provisional=True)
        self.assertEqual(info["status"], "prepared")
        self.assertEqual(json.loads((output / "training.json").read_text())["target_revision"], "a" * 40)
        self.assertEqual(info["source_question_ids"], ["q1", "q2"])
        self.assertTrue(all((output / arm / "dataset.jsonl").exists() for arm in ARMS))
        with self.assertRaisesRegex(ValueError, "덮어쓰지"):
            prepare_training(self.source, output, steps=20, allow_provisional=True)
        exploratory = prepare_training(self.source, self.root / "exploratory-100", steps=100, allow_provisional=True)
        self.assertEqual(exploratory["steps"], 100)
        self.assertEqual(exploratory["claim_scope"], "unreviewed_exploratory")
        self.assertTrue(all(len(digest) == 64 for digest in exploratory["code_sha256"].values()))

    def test_token_mask_supervises_answer_only_and_never_truncates(self):
        row = {"question_id": "fixture", "messages": [{"role": "user", "content": "original question"}],
               "completion": '{"answer":"A","explanation":""}'}
        tokenizer = CharacterTokenizer()
        encoded = tokenize_example(tokenizer, row, 2048)
        prompt_len = len(tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True))
        self.assertEqual(encoded["labels"][:prompt_len], [-100] * prompt_len)
        self.assertEqual(encoded["labels"][prompt_len:], encoded["input_ids"][prompt_len:])
        with self.assertRaisesRegex(ValueError, "truncation"):
            tokenize_example(tokenizer, row, 10)


if __name__ == "__main__":
    unittest.main()
