"""전체 MedQA 학습: 원문을 자르지 않고 정답 한 글자에만 LoRA 손실을 계산한다."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean", "random", "flip", "near_miss")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_id(row: dict) -> str:
    return str(row.get("question_id", row["id"]))


def validate_rows(rows: list[dict], arm: str) -> int:
    """조건마다 같은 원문 두 개를 사용하고 정답을 바꾸지 않았는지 확인한다."""
    if not rows:
        raise ValueError("학습 데이터가 비어 있습니다.")
    grouped = defaultdict(list)
    for row in rows:
        if row["answer_idx"] not in "ABCD" or len(row["answer_idx"]) != 1:
            raise ValueError("정답은 A, B, C, D 중 하나여야 합니다.")
        if not row.get("question") or set(row["options"]) != set("ABCD"):
            raise ValueError("원문 질문과 네 선택지가 필요합니다.")
        grouped[source_id(row)].append(row)
    for qid, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(f"문제마다 입력 두 개가 필요합니다: {qid}")
        if any(pair[0][key] != pair[1][key] for key in ("question", "options", "answer_idx")):
            raise ValueError(f"원문 또는 정답이 바뀌었습니다: {qid}")
        notes = [row.get("note", "") for row in pair]
        if arm == "clean" and any(notes):
            raise ValueError("원문 학습 조건에 추가 문장이 있습니다.")
        if arm != "clean" and sum(bool(note) for note in notes) != 1:
            raise ValueError(f"원문 한 개와 추가 문장이 있는 입력 한 개가 필요합니다: {qid}")
    return len(grouped)


def epoch_groups(count: int, effective_batch: int, seed: int, epoch: int) -> list[list[int]]:
    """매 epoch 모든 행을 정확히 한 번 사용한다. 마지막 묶음도 버리지 않는다."""
    if count < 1 or effective_batch < 1:
        raise ValueError("count와 effective_batch는 양수여야 합니다.")
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return [order[start:start + effective_batch] for start in range(0, count, effective_batch)]


def padded_inputs(sequences: list[list[int]], pad_id: int) -> dict[str, list[list[int]]]:
    """왼쪽만 패딩한다. 마지막 위치는 항상 실제 prompt의 마지막 토큰이다."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("빈 토큰열은 사용할 수 없습니다.")
    width = max(map(len, sequences))
    return {
        "input_ids": [[pad_id] * (width - len(sequence)) + sequence for sequence in sequences],
        "attention_mask": [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences],
        "position_ids": [[0] * (width - len(sequence)) + list(range(len(sequence))) for sequence in sequences],
    }


def letter_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for letter in "ABCD":
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"정답 {letter}가 단일 토큰이 아닙니다: {tokens}")
        result[letter] = tokens[0]
    if len(set(result.values())) != 4:
        raise ValueError("정답 토큰이 서로 다르지 않습니다.")
    return result


def tokenize_rows(tokenizer, rows: list[dict], context_limit: int) -> list[list[int]]:
    from .full_data import build_messages
    sequences = []
    for row in rows:
        messages = build_messages(row, row.get("note", ""), defended=bool(row.get("defended", False)))
        sequence = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(sequence) > context_limit:
            raise ValueError(f"{source_id(row)}의 {len(sequence)} 토큰이 모델 한도를 넘습니다. 원문을 자르지 않습니다.")
        sequences.append(sequence)
    return sequences


def configure_one_gpu(manifest: dict, arm: str, gpu: str) -> None:
    assigned = manifest.get("physical_gpus", dict(zip(ARMS, (4, 5, 6, 7))))
    if isinstance(assigned, list):
        assigned = dict(zip(ARMS, assigned))
    if not str(gpu).isdigit() or str(assigned[arm]) != str(gpu):
        raise ValueError("이 조건에 지정된 GPU와 요청한 GPU가 다릅니다.")
    # torch를 불러오기 전에 한 장만 노출한다. 다른 GPU에는 모델을 올리지 않는다.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRITON_CACHE_DIR"] = str(ROOT / ".cache/triton")
    headers = ROOT / ".deps/usr/include"
    if headers.exists():
        paths = [str(headers), str(headers / "python3.10"), str(headers / "x86_64-linux-gnu/python3.10")]
        os.environ["CPATH"] = ":".join(paths + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))


def _dataset_path(directory: Path, stem: str, *, evaluation: bool = False) -> Path:
    candidates = [directory / ("evaluation" if evaluation else "training") / f"{stem}.jsonl",
                  directory / "datasets" / f"{stem}.jsonl"]
    return next((path for path in candidates if path.exists()), candidates[0])


def evaluation_summary(predictions: list[dict]) -> dict:
    """정답률과 같은 질문의 원문 정답→추가 문장 오답을 분리해 집계한다."""
    groups = defaultdict(list)
    clean = {}
    for row in predictions:
        if not row.get("note"):
            clean[(row["question_id"], bool(row.get("defended", False)))] = row
    for row in predictions:
        condition = "clean" if not row.get("note") else "perturbed"
        prompt = "defended" if row.get("defended") else "neutral"
        seen = row.get("historical_test_seen", row.get("previously_observed", False))
        subsets = ("all", "previously_observed" if seen else "previously_unseen")
        conditions = [condition]
        if row.get("note") and row.get("kind") in ("supports_wrong", "supports_correct"):
            conditions.append(row["kind"])
        for subset in subsets:
            for group_condition in conditions:
                groups[f"{subset}/{prompt}/{group_condition}"].append(row)
    summary = {}
    for name, rows in sorted(groups.items()):
        matched = [row for row in rows if row.get("note") and
                   clean.get((row["question_id"], bool(row.get("defended", False))), {}).get("correct")]
        entry = {"n": len(rows), "correct": sum(row["correct"] for row in rows),
                 "accuracy": sum(row["correct"] for row in rows) / len(rows),
                 "source_questions": len({row["question_id"] for row in rows})}
        if not name.endswith("/clean"):
            entry.update({"clean_correct_perturbed_cases": len(matched),
                          "correct_to_wrong": sum(not row["correct"] for row in matched),
                          "correct_to_wrong_rate": sum(not row["correct"] for row in matched) / len(matched) if matched else None})
        summary[name] = entry
    return summary


def train_and_evaluate(experiment: Path, arm: str, gpu: str, *, stage: str = "all") -> dict:
    experiment = Path(experiment).resolve()
    if experiment.is_dir():
        experiment = experiment / "experiment.json"
    directory = experiment.parent
    manifest = json.loads(experiment.read_text())
    configure_one_gpu(manifest, arm, str(gpu))
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = manifest.get("training", {})
    seed = int(settings.get("seed", manifest.get("seed", 20260922)))
    epochs = int(settings.get("epochs", 2))
    effective = int(settings.get("effective_batch", 16))
    lr = float(settings.get("lr", settings.get("learning_rate", 1e-4)))
    if epochs < 1 or effective < 1:
        raise ValueError("epochs와 effective_batch는 양수여야 합니다.")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.92, 0)
    model_name = manifest.get("model", MODEL)
    revision = manifest.get("revision", manifest.get("model_revision", REVISION))
    data_path = _dataset_path(directory, arm)
    rows = read_jsonl(data_path)
    n_questions = validate_rows(rows, arm)
    output = directory / "arms" / arm
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "train_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    signature = {"experiment_sha256": file_sha(experiment), "dataset_sha256": file_sha(data_path),
                 "model": model_name, "revision": revision, "seed": seed, "epochs": epochs,
                 "effective_batch": effective, "learning_rate": lr, "physical_gpu": str(gpu)}
    if state and state.get("signature") != signature:
        raise ValueError("저장된 학습 설정이나 데이터가 다릅니다. 새 실행 폴더를 사용하세요.")
    if (output / "TRAIN_COMPLETE").exists() and (stage == "train" or (output / "EVAL_COMPLETE").exists()):
        print("요청한 학습/평가가 이미 완료되어 기존 결과를 유지합니다.", flush=True)
        return state
    print(json.dumps({"stage": "loading", "arm": arm, "questions": n_questions, "rows": len(rows),
                      "epochs": epochs, "gpu": gpu}, ensure_ascii=False), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letters = letter_token_ids(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, local_files_only=True,
                                               torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                               attn_implementation="sdpa")
    context_limit = int(base.config.max_position_embeddings)
    encoded = tokenize_rows(tokenizer, rows, context_limit)
    gold_tokens = [letters[row["answer_idx"]] for row in rows]
    completed = int(state.get("completed_epochs", 0))
    checkpoint = output / f"epoch_{completed}"
    if completed:
        model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                                               lora_dropout=0.0, task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, foreach=False)
    if completed:
        checkpoint_state = torch.load(checkpoint / "optimizer.pt", map_location="cuda:0", weights_only=True)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        for parameter_state in optimizer.state.values():
            parameter_state["step"] = parameter_state["step"].cpu()
        torch.set_rng_state(checkpoint_state["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint_state["cuda_rng"].cpu())
    else:
        # 메모리 탐색에 optimizer 상태까지 포함한다. 이 단계는 가중치를 갱신하지 않는다.
        for parameter in trainable:
            optimizer.state[parameter] = {"step": torch.tensor(0.0), "exp_avg": torch.zeros_like(parameter),
                                           "exp_avg_sq": torch.zeros_like(parameter)}

    def tensors(indices: list[int], sequences=encoded):
        return {name: torch.tensor(value, dtype=torch.long, device="cuda:0")
                for name, value in padded_inputs([sequences[index] for index in indices], tokenizer.pad_token_id).items()}

    def loss_for(indices: list[int]):
        inputs = tensors(indices)
        # 길이×전체 어휘 logits를 만들지 않는다. 정답 위치 하나의 CE만 계산한다.
        logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
        labels = torch.tensor([gold_tokens[index] for index in indices], dtype=torch.long, device="cuda:0")
        return F.cross_entropy(logits.float(), labels, reduction="sum")

    microbatch = int(state.get("microbatch", 0))
    if stage != "evaluate" and completed < epochs:
        if not microbatch:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            longest = sorted(range(len(encoded)), key=lambda index: len(encoded[index]), reverse=True)
            probes = []
            model.train()
            candidates = [batch for batch in (1, 2, 4, 8, 16) if batch <= effective]
            if effective not in candidates:
                candidates.append(effective)
            for candidate in sorted(set(candidates)):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # 가장 긴 입력으로 batch 전체를 채워서 이후 어떤 조합도 감당하게 한다.
                indices = [longest[0]] * candidate
                try:
                    (loss_for(indices) / candidate).backward()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved()
                    probes.append({"batch": candidate, "peak_reserved_bytes": peak, "fits": peak <= total_memory * 0.90})
                    if peak > total_memory * 0.90:
                        break
                    microbatch = candidate
                except torch.OutOfMemoryError:
                    probes.append({"batch": candidate, "fits": False, "reason": "CUDA OOM"})
                    break
                finally:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            if not microbatch:
                raise RuntimeError("가장 긴 원문의 batch=1도 메모리에 들어가지 않습니다. 원문을 자르지 않고 중단합니다.")
            state.update({"microbatch": microbatch, "batch_probes": probes})
        state.update({"signature": signature, "status": "training", "completed_epochs": completed,
                      "source_questions": n_questions, "rows_per_epoch": len(rows),
                      "max_prompt_tokens": max(map(len, encoded)), "min_prompt_tokens": min(map(len, encoded)),
                      "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                      "training": {"base_dtype": "bfloat16", "method": "LoRA", "rank": 8, "alpha": 16,
                                   "dropout": 0, "target_modules": "all-linear", "loss": "full-vocabulary CE on next gold-letter token only",
                                   "gradient_checkpointing": True, "truncation": False},
                      "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                      "code_sha256": {name: file_sha(ROOT / "gaim" / name) for name in ("full_train.py", "full_data.py")}})
        write_json(state_path, state)
        print(json.dumps({"stage": "training", "arm": arm, "microbatch": microbatch,
                          "effective_batch": effective, "max_prompt_tokens": state["max_prompt_tokens"]}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        global_step = int(state.get("optimizer_steps", 0))
        start = time.time()
        for epoch in range(completed, epochs):
            model.train()
            processed, loss_sum = 0, 0.0
            groups = epoch_groups(len(rows), effective, seed, epoch)
            for group in groups:
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for offset in range(0, len(group), microbatch):
                    chunk = group[offset:offset + microbatch]
                    loss = loss_for(chunk)
                    group_loss += loss.detach().item()
                    (loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                processed += len(group)
                loss_sum += group_loss
                global_step += 1
                if global_step % 20 == 0 or processed == len(rows):
                    progress = {"stage": "training", "arm": arm, "epoch": epoch + 1, "epochs": epochs,
                                "epoch_rows": processed, "total_epoch_rows": len(rows), "optimizer_steps": global_step,
                                "mean_loss": loss_sum / processed, "grad_norm": float(norm),
                                "elapsed_seconds": round(time.time() - start, 1),
                                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)}
                    with (output / "progress.jsonl").open("a") as handle:
                        handle.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
            epoch_path = output / f"epoch_{epoch + 1}"
            epoch_path.mkdir(exist_ok=True)
            model.save_pretrained(epoch_path / "adapter")
            tokenizer.save_pretrained(epoch_path / "adapter")
            torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, epoch_path / "optimizer.pt")
            state.update({"completed_epochs": epoch + 1, "optimizer_steps": global_step,
                          "processed_rows": (epoch + 1) * len(rows), "last_epoch_mean_loss": loss_sum / len(rows),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "status": "training" if epoch + 1 < epochs else "complete"})
            write_json(state_path, state)
        model.save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        (output / "TRAIN_COMPLETE").write_text(json.dumps({"epochs": epochs, "processed_rows": epochs * len(rows)}) + "\n")
    elif completed >= epochs and not (output / "TRAIN_COMPLETE").exists():
        model.save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        (output / "TRAIN_COMPLETE").write_text(json.dumps({"epochs": epochs, "processed_rows": epochs * len(rows)}) + "\n")
    if stage == "train":
        return state
    if not (output / "TRAIN_COMPLETE").exists():
        raise ValueError("학습이 끝난 후에 평가할 수 있습니다.")
    cases_path = _dataset_path(directory, "test_cases", evaluation=True)
    if not cases_path.exists():
        print("학습 완료. 공통 시험 입력이 아직 없어 평가는 나중에 실행합니다.", flush=True)
        return state
    evaluate(model, tokenizer, letters, cases_path, output, context_limit, max(1, microbatch), tensors, torch)
    return state


def evaluate(model, tokenizer, letters: dict, cases_path: Path, output: Path, context_limit: int,
             microbatch: int, tensors, torch) -> None:
    from .full_data import input_sha256
    rows = read_jsonl(cases_path)
    prediction_path = output / "test_predictions.jsonl"
    if (output / "EVAL_COMPLETE").exists():
        print("학습 및 평가가 이미 완료되어 기존 결과를 유지합니다.", flush=True)
        return
    cases = tokenize_rows(tokenizer, rows, context_limit)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    torch.cuda.empty_cache()
    # 완료한 case는 다시 계산하지 않아도 된다. 각 case ID는 공유 시험 파일의 행 번호로 고정한다.
    done = read_jsonl(prediction_path) if prediction_path.exists() else []
    if len(done) > len(rows):
        raise ValueError("저장된 평가 행 수가 시험 입력보다 큽니다.")
    for index, prediction in enumerate(done):
        if prediction["case_index"] != index or prediction["input_sha256"] != input_sha256(rows[index]):
            raise ValueError("시험 입력과 저장된 평가 결과가 다릅니다.")
    predictions = done
    step = len(done)
    eval_batch = max(1, microbatch)
    with prediction_path.open("a") as handle:
        while step < len(rows):
            indices = list(range(step, min(step + eval_batch, len(rows))))
            try:
                with torch.inference_mode():
                    logits = model(**tensors(indices, cases), logits_to_keep=1, use_cache=False).logits[:, -1, :]
                    probs = logits[:, [letters[letter] for letter in "ABCD"]].float().softmax(dim=-1).cpu().tolist()
                del logits
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eval_batch == 1:
                    raise
                eval_batch = max(1, eval_batch // 2)
                continue
            for index, probabilities in zip(indices, probs):
                row = rows[index]
                answer = "ABCD"[max(range(4), key=probabilities.__getitem__)]
                prediction = {key: value for key, value in row.items() if key not in ("question", "options")}
                prediction.update({"case_index": index, "question_id": source_id(row), "gold": row["answer_idx"],
                                   "answer": answer, "predicted": answer, "correct": answer == row["answer_idx"],
                                   "probabilities": dict(zip("ABCD", probabilities)),
                                   "gold_probability": probabilities["ABCD".index(row["answer_idx"])],
                                   "input_tokens": len(cases[index]), "input_sha256": input_sha256(row)})
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                predictions.append(prediction)
            handle.flush()
            step += len(indices)
            if step % 256 < len(indices) or step == len(rows):
                print(json.dumps({"stage": "evaluation", "complete_cases": step, "total_cases": len(rows)}), flush=True)
    write_json(output / "test_summary.json", {"cases_sha256": file_sha(cases_path),
                                             "prediction_sha256": file_sha(prediction_path),
                                             "scoring": "argmax of normalized logits over A/B/C/D; no generated explanation",
                                             "metrics": evaluation_summary(predictions)})
    (output / "EVAL_COMPLETE").write_text(str(len(predictions)) + "\n")
    print(json.dumps({"stage": "complete", "output": str(output), "test_cases": len(predictions)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    args = parser.parse_args()
    train_and_evaluate(args.experiment, args.arm, args.gpu, stage=args.stage)


if __name__ == "__main__":
    main()
