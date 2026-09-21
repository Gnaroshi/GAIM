"""Offline source fixtures exercise data provenance; these are not experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gaim import data


FIXTURE_REVISION = "1" * 40


def fixture_record(number: int) -> dict:
    # 명시적인 테스트 문구이며 실제 MedQA 문항이나 임상 근거로 사용하지 않는다.
    options = {letter: f"fixture choice {number}/{letter}" for letter in "ABCD"}
    answer = "ABCD"[number % 4]
    return {"question": f"  FIXTURE ONLY {number}: line one\nline two °F?  ",
            "options": options, "answer_idx": answer, "answer": options[answer]}


def jsonl(records: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records)


class FixtureDownloads:
    def __init__(self, train: list[dict] | None = None, test: list[dict] | None = None):
        self.train = train if train is not None else [fixture_record(i) for i in range(12)]
        self.test = test if test is not None else [fixture_record(i) for i in range(100, 106)]
        self.calls: list[str] = []
        self.metadata = {"sha": FIXTURE_REVISION,
                         "siblings": [{"rfilename": name} for name in data.SOURCE_FILES.values()],
                         "cardData": {"license": "fixture-license-only"}}

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if "/api/datasets/" in url:
            return json.dumps(self.metadata).encode()
        if url.endswith(data.SOURCE_FILES["train"]):
            return jsonl(self.train)
        if url.endswith(data.SOURCE_FILES["test"]):
            return jsonl(self.test)
        raise AssertionError(f"Unexpected network URL in offline fixture: {url}")


class MedQAPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gaim-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "prepared"
        self.downloads = FixtureDownloads()

    def prepare(self, output: Path | None = None, **kwargs):
        options = {"train_size": 4, "dev_size": 2, "test_size": 3, "seed": 123,
                   "revision": FIXTURE_REVISION, "fetch_bytes": self.downloads}
        options.update(kwargs)
        return data.prepare_dataset(output or self.output, **options)

    def rows(self, split: str, output: Path | None = None) -> list[dict]:
        return [json.loads(line) for line in ((output or self.output) / f"{split}.jsonl").read_text().splitlines()]

    def test_preserves_source_content_and_records_explicit_dev_provenance(self):
        manifest = self.prepare()
        self.assertEqual(manifest["selection"]["selected_counts"], {"train": 4, "dev": 2, "test": 3})
        self.assertFalse(manifest["provenance"]["validation_is_official"])
        self.assertEqual(manifest["configuration"]["revision"], FIXTURE_REVISION)
        hashes = {}
        for split in data.SPLITS:
            hashes[split] = set()
            for row in self.rows(split):
                source_split = "train" if split == "dev" else split
                source = getattr(self.downloads, source_split)[row["source_index"]]
                self.assertEqual(row["source_split"], source_split)
                for key in ("question", "options", "answer_idx"):
                    self.assertEqual(row[key], source[key])
                self.assertEqual(row["question_sha256"], hashlib.sha256(source["question"].encode()).hexdigest())
                hashes[split].add(row["question_sha256"])
            self.assertEqual(manifest["selection"]["sampled_ids"][split], [r["id"] for r in self.rows(split)])
        self.assertFalse(hashes["train"] & hashes["dev"])
        self.assertFalse(hashes["train"] & hashes["test"])
        self.assertFalse(hashes["dev"] & hashes["test"])
        self.assertEqual((self.output / "sources" / data.SOURCE_FILES["train"]).read_bytes(), jsonl(self.downloads.train))
        self.assertTrue(all(FIXTURE_REVISION in url for url in self.downloads.calls))

    def test_identical_configuration_reuses_verified_data_without_network(self):
        original = self.prepare()

        def no_network(url):
            self.fail(f"Reuse must be offline: {url}")

        self.assertEqual(self.prepare(fetch_bytes=no_network), original)

    def test_size_or_seed_change_requires_explicit_overwrite(self):
        self.prepare()
        before = (self.output / "train.jsonl").read_bytes()
        for changed in ({"seed": 124}, {"train_size": 5}, {"dev_size": 3}):
            with self.subTest(changed=changed), self.assertRaises(data.DataPreparationError):
                self.prepare(**changed)
        self.assertEqual((self.output / "train.jsonl").read_bytes(), before)
        self.prepare(train_size=5, overwrite=True)
        self.assertEqual(len(self.rows("train")), 5)

    def test_expanding_train_keeps_dev_and_existing_train_members_fixed(self):
        self.prepare()
        second = Path(self.temporary.name) / "larger"
        self.prepare(output=second, train_size=7)
        self.assertEqual(self.rows("dev"), self.rows("dev", second))
        self.assertEqual(self.rows("train"), self.rows("train", second)[:4])
        self.assertEqual(self.rows("test"), self.rows("test", second))

    def test_same_seed_reproduces_all_prepared_artifacts(self):
        self.prepare()
        second = Path(self.temporary.name) / "same"
        self.prepare(output=second)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(self.output)).read_bytes())

    def test_dedup_reserves_entire_source_test_before_sampling(self):
        self.downloads.train.append(copy.deepcopy(self.downloads.train[0]))
        self.downloads.train.extend(copy.deepcopy(self.downloads.test))
        manifest = self.prepare()
        self.assertEqual(manifest["selection"]["within_source_duplicates_removed"]["train"], 1)
        self.assertEqual(manifest["selection"]["train_questions_overlapping_source_test_removed"], 6)
        original_test_hashes = {data.sha256_bytes(row["question"].encode()) for row in self.downloads.test}
        self.assertTrue(all(row["question_sha256"] not in original_test_hashes
                            for split in ("train", "dev") for row in self.rows(split)))

    def test_conflicting_duplicate_group_is_excluded_without_choosing_an_answer(self):
        conflicting = copy.deepcopy(self.downloads.train[0])
        conflicting["answer_idx"] = "B"
        conflicting["answer"] = conflicting["options"]["B"]
        self.downloads.train.append(conflicting)
        manifest = self.prepare()
        conflicts = manifest["selection"]["conflicting_question_groups_excluded"]["train"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["source_ids"], ["medqa-usmle4:train:00000000", "medqa-usmle4:train:00000012"])
        self.assertEqual(manifest["selection"]["conflicting_source_rows_excluded"]["train"], 2)
        self.assertTrue(all(row["question"] != conflicting["question"]
                            for split in ("train", "dev") for row in self.rows(split)))

    def test_original_answer_text_mismatch_is_rejected(self):
        self.downloads.train[0]["answer"] = "incorrect fixture answer"
        with self.assertRaisesRegex(data.DataPreparationError, "정답 문자와 정답 본문"):
            self.prepare()

    def test_tampered_prepared_record_and_manifest_are_rejected(self):
        self.prepare()
        path = self.output / "train.jsonl"
        rows = self.rows("train")
        rows[0]["answer_idx"] = "D" if rows[0]["answer_idx"] != "D" else "A"
        changed = jsonl(rows)
        path.write_bytes(changed)
        # 출력 checksum까지 함께 바꿔도 원천에서 다시 계산하므로 변조를 감지해야 한다.
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["train.jsonl"] = {"sha256": data.sha256_bytes(changed), "bytes": len(changed)}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(data.DataPreparationError):
            self.prepare()

    def test_tampered_raw_source_is_rejected_on_reuse(self):
        self.prepare()
        path = self.output / "sources" / data.SOURCE_FILES["train"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(data.DataPreparationError):
            self.prepare()

    def test_changed_hf_revision_or_missing_source_file_is_rejected(self):
        self.downloads.metadata["sha"] = "2" * 40
        with self.assertRaisesRegex(data.DataPreparationError, "revision SHA"):
            self.prepare()
        self.downloads.metadata["sha"] = FIXTURE_REVISION
        self.downloads.metadata["siblings"] = []
        with self.assertRaisesRegex(data.DataPreparationError, "JSONL"):
            self.prepare()

    def test_insufficient_rows_and_unpinned_revision_are_rejected(self):
        with self.assertRaisesRegex(data.DataPreparationError, "부족"):
            self.prepare(train_size=12)
        with self.assertRaisesRegex(data.DataPreparationError, "commit SHA"):
            self.prepare(revision="main")

    def test_failed_download_during_overwrite_preserves_existing_dataset(self):
        original = self.prepare()
        before = (self.output / "manifest.json").read_bytes()

        def failing_download(url):
            if "/api/" in url:
                return self.downloads(url)
            raise data.DataPreparationError("explicit offline fixture download failure")

        with self.assertRaises(data.DataPreparationError):
            self.prepare(overwrite=True, seed=999, fetch_bytes=failing_download)
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)
        self.assertEqual(self.prepare(), original)

    def test_overwrite_does_not_delete_unmanaged_files(self):
        self.prepare()
        unrelated = self.output / "research_notes.txt"
        unrelated.write_text("fixture user-owned notes")
        with self.assertRaisesRegex(data.DataPreparationError, "관리하지 않는"):
            self.prepare(overwrite=True)
        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
