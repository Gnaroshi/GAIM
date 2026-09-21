"""GPU/모델 호출 없이 pane 분리, 실제 길이 padding, loss 가중치와 상태 저장을 확인한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gaim.train_arm import (accumulation_weights, choose_microbatch, claim_arm,
                            isolate_gpu, pad_examples, prepare_panes,
                            supervised_tokens, update_arm_status)
from gaim.training import ARMS, _save_manifest


class FakeCudaOOM(Exception):
    pass


def example(ids, labels):
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class PaneTrainingTest(unittest.TestCase):
    def test_padding_uses_current_real_length_and_masks_only_added_tokens(self):
        rows = [example([1, 2, 3], [-100, 2, 3]), example([4, 5], [-100, 5])]
        batch = pad_examples(rows, 99)
        self.assertEqual(batch["input_ids"], [[1, 2, 3], [4, 5, 99]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"], [[-100, 2, 3], [-100, 5, -100]])
        self.assertEqual(supervised_tokens(rows), 3)

    def test_token_weighted_accumulation_matches_one_full_batch_for_unequal_lengths(self):
        rows = [example([0, 1, 2], [-100, 1, 2]), example([0, 3], [-100, 3]),
                example([0, 4, 5, 6], [-100, 4, 5, 6]), example([0, 7], [-100, 7])]
        token_losses = [[1.0, 4.0], [3.0], [2.0, 4.0, 6.0], [8.0]]
        expected = sum(sum(loss) for loss in token_losses) / 7
        for size in (1, 2, 3, 4):
            weights = accumulation_weights(rows, size)
            actual = 0
            for start, weight in zip(range(0, 4, size), weights):
                values = sum(token_losses[start:start + size], [])
                actual += sum(values) / len(values) * weight
            self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(sum(weights), 1.0)
        self.assertNotEqual(accumulation_weights(rows, 1), [0.25] * 4)
        # 마지막 1행 minibatch를 앞의 3행과 똑같이 가중하지 않는다.
        self.assertEqual(accumulation_weights(rows, 3), [6 / 7, 1 / 7])

    def test_invalid_accumulation_and_empty_loss_fail(self):
        with self.assertRaisesRegex(ValueError, "양의 정수"):
            accumulation_weights([example([0, 1], [-100, 1])] * 4, 0)
        with self.assertRaisesRegex(ValueError, "감독"):
            accumulation_weights([example([0, 1], [0, -100])], 1)

    def test_auto_batch_finds_largest_fitting_integer_15_without_divisor_restriction(self):
        called = []
        def probe(size):
            called.append(size)
            if size >= 18:
                raise FakeCudaOOM()
            return size * 100
        size, trials = choose_microbatch(20, 1500, probe, FakeCudaOOM)
        self.assertEqual(size, 15)
        self.assertEqual(called, [10, 15, 18, 16])
        self.assertLessEqual(len(called), 5)
        self.assertEqual([trial["status"] for trial in trials], ["fits", "fits", "cuda_oom", "above_budget"])

    def test_auto_batch_reaches_full_20_in_at_most_five_probes(self):
        size, trials = choose_microbatch(20, 2000, lambda size: size * 100, FakeCudaOOM)
        self.assertEqual(size, 20)
        self.assertLessEqual(len(trials), 5)

    def test_auto_batch_does_not_swallow_nonmemory_errors(self):
        def probe(size):
            raise RuntimeError("bad model output")
        with self.assertRaisesRegex(RuntimeError, "bad model output"):
            choose_microbatch(20, 500, probe, FakeCudaOOM)
        with self.assertRaisesRegex(RuntimeError, "batch 1"):
            choose_microbatch(20, 1, lambda size: 1000, FakeCudaOOM)

    def test_single_arm_ignores_inherited_four_gpu_mask_after_checking_assignment(self):
        manifest = {"physical_gpus": dict(zip(ARMS, (4, 5, 6, 7)))}
        env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7", "GAIM_CUDA_DEVICES": "4,5,6,7"}
        isolate_gpu(manifest, "adaptive", 7, env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        with self.assertRaisesRegex(ValueError, "준비된 GPU"):
            isolate_gpu(manifest, "adaptive", 0, env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")

    def make_prepared(self, output):
        for arm in ARMS:
            (output / arm).mkdir(parents=True)
        manifest = {"status": "prepared", "arm_status": {arm: {"status": "prepared"} for arm in ARMS}}
        _save_manifest(output, manifest)
        return manifest

    def test_four_independent_completions_make_replay_compatible_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.make_prepared(output)
            for arm in ARMS[:-1]:
                manifest = update_arm_status(output, arm, "complete", batch_size=5)
                self.assertEqual(manifest["status"], "running")
                self.assertFalse((output / "COMPLETE").exists())
            manifest = update_arm_status(output, ARMS[-1], "complete", batch_size=10)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(json.loads((output / "training.json").read_text()), manifest)
            self.assertTrue((output / "COMPLETE").exists())

    def test_arm_failure_is_not_overwritten_by_other_arm_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.make_prepared(output)
            update_arm_status(output, "clean", "failed", error="fixture")
            manifest = update_arm_status(output, "random", "running")
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse((output / "COMPLETE").exists())

    def test_same_arm_cannot_run_twice_or_overwrite_old_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.make_prepared(output)
            with claim_arm(output, "clean"):
                with self.assertRaisesRegex(ValueError, "이미 실행"):
                    with claim_arm(output, "clean"):
                        pass
            (output / "clean/training_loss.jsonl").write_text("previous output\n")
            with self.assertRaisesRegex(ValueError, "이미 있습니다"):
                with claim_arm(output, "clean"):
                    pass

    def test_prepare_sets_new_precision_and_batch_without_changing_historical_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            original = {"training": {"quantization": "4-bit NF4", "batch_size": 1,
                                      "gradient_accumulation": 2}, "code_sha256": {}}
            with patch("gaim.train_arm.prepare_training", return_value=original) as prepare:
                manifest = prepare_panes(Path("fixture"), output, steps=20, effective_batch=20)
            self.assertEqual(prepare.call_args.kwargs["steps"], 20)
            self.assertEqual(manifest["training"]["quantization"], "none")
            self.assertEqual(manifest["training"]["effective_batch"], 20)
            self.assertFalse(manifest["training"]["gradient_checkpointing"])
            self.assertEqual(manifest["execution"], "one_arm_per_pane")
            self.assertEqual(set(manifest["arm_status"]), set(ARMS))


if __name__ == "__main__":
    unittest.main()
