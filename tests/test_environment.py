"""GPU를 호출하지 않고 장치 선택·상속·재개 보호를 검증한다."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gaim import environment, run, training, worker


class GPUSelectionTests(unittest.TestCase):
    def test_original_default(self):
        self.assertEqual(environment.physical_gpus({}), (4, 5, 6, 7))

    def test_explicit_override_takes_priority_and_preserves_order(self):
        selected = environment.physical_gpus({"GAIM_CUDA_DEVICES": "3, 0,2,1",
                                              "CUDA_VISIBLE_DEVICES": "4,5,6,7"})
        self.assertEqual(selected, (3, 0, 2, 1))

    def test_inherited_cuda_mask_is_respected(self):
        self.assertEqual(environment.physical_gpus({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
                         (0, 1, 2, 3))

    def test_invalid_masks_fail_instead_of_selecting_other_gpus(self):
        for variable in ("GAIM_CUDA_DEVICES", "CUDA_VISIBLE_DEVICES"):
            for mask in ("", "0", "0,1", "0,1,2", "0,1,2,3,4", "0,1,1,3",
                         "-1,0,1,2", "0,1,2,GPU-uuid", "0,,2,3", "0,1,2,3x"):
                with self.subTest(variable=variable, mask=mask), self.assertRaises(ValueError):
                    environment.physical_gpus({variable: mask})

    def test_equivalent_numeric_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            environment.parse_gpu_mask("0,00,1,2")

    def test_configure_normalizes_before_model_import(self):
        with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "3, 0, 2, 1"}, clear=True):
            self.assertEqual(environment.configure(), (3, 0, 2, 1))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,0,2,1")
            self.assertEqual(environment.configure(), (3, 0, 2, 1))

    def test_child_inherits_order_and_expected_mask(self):
        with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "0,1,2,3"}, clear=True):
            devices = environment.configure()
            child = environment.child_environment(devices)
            self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
            self.assertEqual(child["GAIM_EXPECTED_CUDA_DEVICES"], "0,1,2,3")
            self.assertEqual(environment.physical_gpus(child), devices)
            child["GAIM_CUDA_DEVICES"] = "3,2,1,0"
            with self.assertRaisesRegex(ValueError, "launcher"):
                environment.physical_gpus(child)

    def test_child_with_only_cuda_mask_is_protected(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=True):
            child = environment.child_environment(environment.configure())
            child["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
            with self.assertRaisesRegex(ValueError, "launcher"):
                environment.physical_gpus(child)

    def test_saved_mapping_must_match_order(self):
        env = {"GAIM_CUDA_DEVICES": "0,1,2,3"}
        self.assertEqual(environment.require_gpu_mapping([0, 1, 2, 3], environ=env), (0, 1, 2, 3))
        for saved in ([3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 2], [False, 1, 2, 3]):
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, "manifest"):
                environment.require_gpu_mapping(saved, environ=env)


class GPUEntryPointTests(unittest.TestCase):
    def test_evaluation_worker_rejects_changed_manifest_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run.json").write_text(json.dumps({"physical_gpus": [4, 5, 6, 7]}))
            with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "0,1,2,3"}, clear=True), \
                    patch("sys.argv", ["worker", "--run-dir", str(path), "--worker", "0"]), \
                    patch.object(worker, "LocalModel") as model:
                with self.assertRaisesRegex(ValueError, "manifest"):
                    worker.main()
                model.assert_not_called()

    def test_training_worker_rejects_mapping_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "training_manifest.json").write_text(json.dumps({
                "physical_gpus": dict(zip(training.ARMS, [4, 5, 6, 7]))}))
            with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "0,1,2,3"}, clear=True):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    training._worker(path, 0)

    def test_resume_rejects_changed_mapping_before_hub_or_gpu_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            cfg = {"run_kind": "evaluation", "split": "test", "k": 5, "limit": 4,
                   "categories": ["nonclinical_background"], "target_temperature": 0,
                   "data_dir": "data"}
            (root / "config.json").write_text(json.dumps(cfg))
            (data / "manifest.json").write_text("{}")
            (data / "test.jsonl").write_text("\n".join(json.dumps({"id": str(i)}) for i in range(4)))
            output = root / "run"
            output.mkdir()
            (output / "run.json").write_text(json.dumps({"physical_gpus": [4, 5, 6, 7]}))
            # 실제 launcher는 프로세스 종료 때 잠금을 닫는다. 같은 Python 프로세스에서
            # CLI 오류 경로를 검사하는 이 테스트는 열린 잠금 핸들을 직접 정리한다.
            opened_locks = []
            original_open = Path.open

            def track_open(path, *args, **kwargs):
                handle = original_open(path, *args, **kwargs)
                if path.name in (".lock", ".gpu.lock"):
                    opened_locks.append(handle)
                return handle

            with patch.dict(os.environ, {"GAIM_CUDA_DEVICES": "0,1,2,3"}, clear=True), \
                    patch.object(run, "ROOT", root), patch.object(run.os, "chdir"), \
                    patch.object(Path, "open", track_open), \
                    patch.object(run, "sha", return_value="CODE_ONLY_HASH"), \
                    patch.object(run.subprocess, "check_output") as query, \
                    patch("sys.argv", ["run", "--config", str(root / "config.json"),
                                       "--run-dir", str(output), "--resume"]):
                try:
                    with self.assertRaisesRegex(ValueError, "manifest"):
                        run.main()
                    query.assert_not_called()
                finally:
                    for handle in opened_locks:
                        handle.close()


if __name__ == "__main__":
    unittest.main()
