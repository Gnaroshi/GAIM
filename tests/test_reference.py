"""공개 기준 기록의 변조가 파일 hash와 의미 검증 양쪽에서 탐지되는지 확인한다."""
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scripts.verify_reference import ReferenceError, verify_reference


REFERENCE = Path(__file__).resolve().parents[1] / "repro/reference"


def update_manifest(root, relative):
    """파일 hash도 함께 변경한 경우에 다음 단계의 의미 검증이 실패해야 한다."""
    path = root / relative
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["files"][relative]
    entry["bytes"] = path.stat().st_size
    entry["published_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["original_sha256"] = entry["published_sha256"]
    manifest_path.write_text(json.dumps(manifest))


class ReferenceVerificationTests(unittest.TestCase):
    def copy_reference(self):
        temporary = TemporaryDirectory(prefix="gaim-reference-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "reference"
        shutil.copytree(REFERENCE, root)
        return root

    def test_original_records_verify_without_model_runtime(self):
        result = verify_reference(REFERENCE)
        self.assertEqual(result["replay_responses_rescored"], 4760)
        self.assertEqual(result["test_results"]["untrained"]["clean_correct"], 66)

    def test_changed_file_fails_hash_before_scoring(self):
        root = self.copy_reference()
        with (root / "test_replay/clean.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ReferenceError, "Byte count mismatch"):
            verify_reference(root)

    def test_rehashed_semantic_mutations_fail(self):
        mutations = [
            ("duplicate", "missing, duplicated or unexpected"),
            ("wrong_input", "input hash mismatch"),
            ("wrong_gold", "gold differs"),
            ("changed_answer", "Parsed answer mismatch"),
            ("changed_summary", "Recomputed replay summary differs"),
        ]
        for mutation, error in mutations:
            with self.subTest(mutation=mutation):
                root = self.copy_reference()
                relative = "test_replay/summary.json" if mutation == "changed_summary" else "test_replay/clean.jsonl"
                path = root / relative
                if mutation == "changed_summary":
                    summary = json.loads(path.read_text())
                    summary["clean"]["clean_correct"] += 1
                    path.write_text(json.dumps(summary))
                else:
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if mutation == "duplicate":
                        rows[-1] = rows[0]
                    elif mutation == "wrong_input":
                        rows[0]["input_sha256"] = "0" * 64
                    elif mutation == "wrong_gold":
                        rows[0]["gold"] = "A" if rows[0]["gold"] != "A" else "B"
                    else:
                        rows[0]["answer"] = "A" if rows[0]["answer"] != "A" else "B"
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                update_manifest(root, relative)
                with self.assertRaisesRegex(ReferenceError, error):
                    verify_reference(root)


if __name__ == "__main__":
    unittest.main()
