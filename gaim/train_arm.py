"""한 pane에서 한 GPU로 BF16 LoRA 한 조건을 학습한다. 기존 NF4 재현 경로는 유지한다."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time

from .environment import configure
from .training import (ARMS, ROOT, _load_jsonl, _save_manifest, _sha, _write_json,
                       prepare_training, tokenize_example)


def prepare_panes(source_run: Path, output_dir: Path, *, steps: int = 20,
                  effective_batch: int = 20, allow_provisional: bool = False) -> dict:
    if type(effective_batch) is not int or effective_batch < 1:
        raise ValueError("effective-batch는 양의 정수여야 합니다.")
    manifest = prepare_training(source_run, output_dir, steps=steps,
                                allow_provisional=allow_provisional)
    manifest.update(schema_version=2, execution="one_arm_per_pane",
                    arm_status={arm: {"status": "prepared"} for arm in ARMS})
    manifest["training"].update(quantization="none", base_dtype="bfloat16",
                                adapter_dtype="float32", gradient_checkpointing=False,
                                batch_size="auto", gradient_accumulation="ceil(effective_batch / batch_size)",
                                effective_batch=effective_batch, vram_fraction=0.90,
                                loss_weighting="mean over supervised answer tokens in each effective batch")
    manifest["code_sha256"]["train_arm.py"] = _sha(Path(__file__))
    manifest["fairness"] = ("Same effective batch, optimizer steps, question cohort and answer-token weighting "
                            "for every arm; real microbatch size may differ by available VRAM.")
    _save_manifest(Path(output_dir).resolve(), manifest)
    return manifest


def isolate_gpu(manifest: dict, arm: str, physical_gpu: int, environ=None) -> None:
    """torch를 읽기 전에 이 pane의 GPU 한 장만 노출한다."""
    env = os.environ if environ is None else environ
    if type(physical_gpu) is not int or physical_gpu < 0:
        raise ValueError("gpu는 실제 GPU의 0 이상 정수 번호여야 합니다.")
    if physical_gpu != manifest["physical_gpus"][arm]:
        raise ValueError(f"{arm}의 준비된 GPU는 {manifest['physical_gpus'][arm]}입니다.")
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TRITON_CACHE_DIR"] = str(ROOT / ".cache/triton")
    env["TORCH_HOME"] = str(ROOT / ".cache/torch")


def pad_examples(examples: list[dict], pad_token_id: int) -> dict:
    """이번 minibatch의 실제 최대 길이까지만 padding하고 padding에는 loss를 주지 않는다."""
    if not examples:
        raise ValueError("빈 minibatch입니다.")
    width = max(len(row["input_ids"]) for row in examples)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in examples:
        length = len(row["input_ids"])
        if any(len(row[key]) != length for key in result):
            raise ValueError("토큰, attention, label 길이가 다릅니다.")
        result["input_ids"].append(row["input_ids"] + [pad_token_id] * (width - length))
        result["attention_mask"].append(row["attention_mask"] + [0] * (width - length))
        result["labels"].append(row["labels"] + [-100] * (width - length))
    return result


def supervised_tokens(examples: list[dict]) -> int:
    # Causal LM은 다음 토큰을 예측하므로 labels[0]은 loss에 포함되지 않는다.
    return sum(sum(label != -100 for label in row["labels"][1:]) for row in examples)


def accumulation_weights(examples: list[dict], microbatch: int) -> list[float]:
    """분할 minibatch의 mean loss를 토큰 수로 가중하면 전체 batch mean loss와 같다."""
    if microbatch < 1:
        raise ValueError("microbatch는 양의 정수여야 합니다.")
    total = supervised_tokens(examples)
    if total == 0:
        raise ValueError("감독할 답변 토큰이 없습니다.")
    return [supervised_tokens(examples[start:start + microbatch]) / total
            for start in range(0, len(examples), microbatch)]


def choose_microbatch(effective_batch: int, budget_bytes: int, probe, oom_exception) -> tuple[int, list[dict]]:
    """메모리가 batch 크기와 함께 증가한다고 보고 이진 탐색한다. 20행이면 최대 5회 검사한다."""
    attempts, best = [], None
    lower, upper = 1, effective_batch
    while lower <= upper:
        size = (lower + upper) // 2
        try:
            measured = probe(size)
        except oom_exception:
            attempts.append({"batch_size": size, "status": "cuda_oom"})
            upper = size - 1
            continue
        fits = measured <= budget_bytes
        attempts.append({"batch_size": size, "status": "fits" if fits else "above_budget",
                         "peak_plus_optimizer_bytes": measured})
        if fits:
            best = size
            lower = size + 1
        else:
            upper = size - 1
    if best is not None:
        return best, attempts
    raise RuntimeError("BF16 LoRA batch 1도 현재 GPU 메모리에 들어가지 않습니다. 다른 작업의 점유량을 확인하세요.")


def update_arm_status(output: Path, arm: str, status: str, **details) -> dict:
    """네 pane은 자기 결과만 쓰고, 짧은 manifest 갱신만 파일 잠금으로 직렬화한다."""
    with (output / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((output / "training_manifest.json").read_text())
        manifest["arm_status"][arm] = {"status": status, **details}
        states = [value["status"] for value in manifest["arm_status"].values()]
        manifest["status"] = ("complete" if all(state == "complete" for state in states)
                              else "failed" if "failed" in states else "running")
        _save_manifest(output, manifest)
        if manifest["status"] == "complete":
            (output / "COMPLETE").write_text("Four independent adapters completed.\n")
        return manifest


@contextlib.contextmanager
def claim_arm(output: Path, arm: str):
    arm_dir = output / arm
    with (arm_dir / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"{arm} pane이 이미 실행 중입니다.") from exc
        if any((arm_dir / name).exists() for name in ("training_loss.jsonl", "training_result.json", "adapter")):
            raise ValueError(f"{arm}의 학습 기록이 이미 있습니다. 새 output-dir를 준비하세요.")
        update_arm_status(output, arm, "running", pid=os.getpid())
        try:
            yield
        except BaseException as exc:
            update_arm_status(output, arm, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


def train_arm(output: Path, arm: str, physical_gpu: int) -> None:
    manifest = json.loads((output / "training_manifest.json").read_text())
    if manifest.get("execution") != "one_arm_per_pane":
        raise ValueError("먼저 train_arm --prepare-only로 새 pane 학습 폴더를 준비하세요.")
    isolate_gpu(manifest, arm, physical_gpu)
    # 위 GPU 한 장 설정 이후에만 torch/transformers를 import한다.
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model

    arm_dir = output / arm
    if _sha(arm_dir / "dataset.jsonl") != manifest["dataset_sha256"][arm]:
        raise ValueError("준비 후 학습 데이터가 변경되었습니다.")
    rows = _load_jsonl(arm_dir / "dataset.jsonl")
    with claim_arm(output, arm):
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        set_seed(manifest["seed"])
        model_id, revision = manifest["target_model"], manifest["target_revision"]
        print(f"[{arm}] GPU {physical_gpu}: BF16 LoRA, {len(rows)} rows, {manifest['steps']} updates", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        encoded = [tokenize_example(tokenizer, row, manifest["max_seq_length"]) for row in rows]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer에 pad/eos token ID가 없습니다.")
        _write_json(arm_dir / "tokenization.json", {
            "rows": len(rows), "max_tokens": max(len(row["input_ids"]) for row in encoded),
            "total_tokens_per_epoch": sum(len(row["input_ids"]) for row in encoded)})
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True,
                                                    torch_dtype=torch.bfloat16, device_map={"": 0},
                                                    attn_implementation="sdpa")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                                                lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        # AdamW의 두 state(exp_avg, exp_avg_sq)를 probe 뒤에도 확보할 수 있도록 뺀다.
        optimizer_reserve = 2 * sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        allocated_bytes = torch.cuda.memory_allocated(0)
        budget_bytes = min(int(total_bytes * 0.90), allocated_bytes + free_bytes - int(total_bytes * 0.10))
        effective = manifest["training"]["effective_batch"]
        longest = sorted(encoded, key=lambda row: len(row["input_ids"]), reverse=True)

        def tensor_batch(examples):
            return {key: torch.tensor(value, dtype=torch.long, device="cuda:0")
                    for key, value in pad_examples(examples, pad_id).items()}

        def probe(size):
            # 길이가 가장 긴 실제 행들로 최대 padding 비용을 확인한다. 2048로 채우지 않는다.
            examples = [longest[index % len(longest)] for index in range(size)]
            batch = loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            try:
                batch = tensor_batch(examples)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("batch 검사에서 유한하지 않은 loss가 나왔습니다.")
                loss.backward()
                torch.cuda.synchronize(0)
                measured = torch.cuda.max_memory_allocated(0) + optimizer_reserve
                print(f"[{arm}] batch {size}: peak + AdamW reserve {measured / 2**30:.2f} GiB "
                      f"/ budget {budget_bytes / 2**30:.2f} GiB", flush=True)
                return measured
            except torch.cuda.OutOfMemoryError:
                print(f"[{arm}] batch {size}: CUDA OOM; trying a smaller batch", flush=True)
                raise
            finally:
                del loss, batch
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        microbatch, attempts = choose_microbatch(effective, budget_bytes, probe, torch.cuda.OutOfMemoryError)
        # Probe는 backward만 실행했다. 가중치/state를 갱신하지 않았으며 RNG도 시작값으로 돌린다.
        set_seed(manifest["seed"])
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0, foreach=False)
        chunk_sizes = [min(microbatch, effective - start) for start in range(0, effective, microbatch)]
        accumulation = len(chunk_sizes)
        settings = {**manifest["training"], "batch_size": microbatch,
                    "gradient_accumulation": accumulation, "microbatch_sizes": chunk_sizes,
                    "budget_bytes": budget_bytes, "optimizer_state_reserve_bytes": optimizer_reserve,
                    "physical_gpu": physical_gpu, "probe_attempts": attempts,
                    "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                                "transformers": transformers.__version__, "peft": peft.__version__,
                                "gpu_name": torch.cuda.get_device_name(0)}}
        _write_json(arm_dir / "training_settings.json", settings)
        print(f"[{arm}] selected batch={microbatch}, accumulation={accumulation}, "
              f"microbatches={chunk_sizes}, effective batch={effective}; training starts", flush=True)
        rng, order, position = random.Random(manifest["seed"]), [], 0
        processed_rows = processed_tokens = 0
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        with (arm_dir / "training_loss.jsonl").open("w") as log:
            for step in range(1, manifest["steps"] + 1):
                examples = []
                for _ in range(effective):
                    if position >= len(order):
                        order = list(range(len(encoded)))
                        rng.shuffle(order)
                        position = 0
                    examples.append(encoded[order[position]])
                    position += 1
                weights = accumulation_weights(examples, microbatch)
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for chunk, weight in zip(range(0, effective, microbatch), weights):
                    batch = tensor_batch(examples[chunk:chunk + microbatch])
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("유한하지 않은 loss입니다. optimizer를 갱신하지 않습니다.")
                    (loss * weight).backward()
                    weighted_loss += float(loss.detach()) * weight
                    del loss, batch
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                processed_rows += effective
                processed_tokens += sum(len(row["input_ids"]) for row in examples)
                entry = {"arm": arm, "step": step, "loss": weighted_loss, "grad_norm": float(norm),
                         "processed_rows": processed_rows, "processed_input_tokens": processed_tokens,
                         "batch_size": microbatch, "gradient_accumulation": accumulation,
                         "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
                         "elapsed_s": time.monotonic() - started}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                if step == 1 or step % 5 == 0 or step == manifest["steps"]:
                    print(f"[{arm}] {step}/{manifest['steps']} loss={weighted_loss:.4f} "
                          f"peak={entry['peak_memory_bytes'] / 2**30:.2f} GiB "
                          f"elapsed={entry['elapsed_s']:.1f}s", flush=True)
        adapter = arm_dir / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        result = {"status": "complete", "arm": arm, "steps": manifest["steps"],
                  "adapter_dir": str(adapter), "base_model": model_id, "base_revision": revision,
                  "processed_rows": processed_rows, "processed_input_tokens": processed_tokens,
                  "claim_scope": manifest["claim_scope"],
                  "provisional_selected_notes": manifest["provisional_selected_notes"],
                  "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
                  "elapsed_s": time.monotonic() - started, "training": settings}
        _write_json(arm_dir / "training_result.json", result)
        finished = update_arm_status(output, arm, "complete", batch_size=microbatch,
                                     gradient_accumulation=accumulation, microbatch_sizes=chunk_sizes,
                                     peak_memory_bytes=result["peak_memory_bytes"])
        print(f"[{arm}] complete: {adapter}", flush=True)
        if finished["status"] == "complete":
            print("All four adapters completed; training.json is ready for replay evaluation.", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--effective-batch", type=int, default=20)
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args(argv)
    try:
        if args.prepare_only:
            if args.source_run is None or args.arm is not None or args.gpu is not None:
                parser.error("--prepare-only에는 --source-run이 필요하고 --arm/--gpu는 쓰지 않습니다.")
            configure()
            manifest = prepare_panes(args.source_run, args.output_dir, steps=args.steps,
                                     effective_batch=args.effective_batch, allow_provisional=args.allow_provisional)
            print(f"Prepared {manifest['rows_per_arm']} rows/arm, {args.steps} updates, "
                  f"effective batch={args.effective_batch}: {args.output_dir}")
        else:
            if args.arm is None or args.gpu is None:
                parser.error("학습 pane에는 --arm과 --gpu가 필요합니다.")
            train_arm(args.output_dir.resolve(), args.arm, args.gpu)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"학습 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
