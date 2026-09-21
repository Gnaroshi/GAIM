"""한 GPU의 후보 채점 → 공통 학습 자료 완성 → 해당 조건 학습/평가."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .full_data import (ARMS, assemble_training, build_messages, candidates, input_sha256,
                        make_test_cases, read_jsonl, write_json, write_jsonl)


def wait_for_gpu(gpu: int, minimum_free_mib: int = 22000) -> None:
    """다른 작업을 종료하지 않고 GPU가 비면 진행한다. pane은 살아 있다."""
    announced = False
    while True:
        raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                                       '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        free_mib = int(raw.strip())
        if free_mib >= minimum_free_mib:
            print(f'[GPU {gpu}] available: {free_mib} MiB free', flush=True)
            return
        if not announced:
            print(f'[GPU {gpu}] waiting for >= {minimum_free_mib} MiB free; currently {free_mib} MiB. Other jobs are untouched.', flush=True)
            announced = True
        time.sleep(60)


class LetterScorer:
    """답변을 길게 생성하지 않고 다음 A/B/C/D 토큰의 점수만 계산한다."""
    def __init__(self, model: str, revision: str, device: str = 'cuda:0', adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.letter_ids = []
        for letter in 'ABCD':
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Answer letter {letter} must be a single token')
            self.letter_ids.append(ids[0])
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={'': device}, attn_implementation='sdpa').eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()

    def score(self, cases: list[dict]) -> list[dict]:
        if not cases:
            return []
        texts = [self.tokenizer.apply_chat_template(
            build_messages(item, item.get('note', ''), item.get('defended', False)),
            tokenize=False, add_generation_prompt=True) for item in cases]
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        if max(lengths) > int(self.model.config.max_position_embeddings):
            raise ValueError('A complete source question exceeds the model context; no text was truncated')
        # 동적 padding에 맞춘 작은 묶음으로 OOM을 줄인다. 메모리를 넘으면 묶음만 반으로 나눈다.
        width = max(lengths)
        batch_size = max(1, min(8, 8192 // width))
        if len(cases) > batch_size:
            return [result for start in range(0, len(cases), batch_size)
                    for result in self.score(cases[start:start + batch_size])]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        positions = inputs['attention_mask'].long().cumsum(-1) - 1
        inputs['position_ids'] = positions.masked_fill(inputs['attention_mask'] == 0, 0)
        try:
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                probabilities = output.logits[:, -1, self.letter_ids].float().softmax(-1).cpu().tolist()
        except self.torch.cuda.OutOfMemoryError:
            del inputs
            self.torch.cuda.empty_cache()
            if len(cases) == 1:
                raise
            middle = len(cases) // 2
            return self.score(cases[:middle]) + self.score(cases[middle:])
        results = []
        for item, length, values in zip(cases, lengths, probabilities):
            predicted = 'ABCD'[max(range(4), key=lambda index: values[index])]
            results.append({'predicted': predicted, 'correct': predicted == item['answer_idx'],
                            'probabilities': dict(zip('ABCD', values)),
                            'gold_probability': values['ABCD'.index(item['answer_idx'])],
                            'input_tokens': length, 'input_sha256': input_sha256(item)})
        return results


def load_progress(path: Path) -> dict[str, dict]:
    """중단되기 직전의 불완전한 마지막 행만 제거하고 이미 채점한 문항은 재사용한다."""
    if not path.exists():
        return {}
    rows = {}
    with path.open('rb+') as handle:
        lines = handle.readlines()
        offset = 0
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                handle.truncate(offset)
                break
            key = row['id']
            if key in rows:
                raise ValueError(f'Duplicate completed question in {path}: {key}')
            rows[key] = row
            offset += len(line)
    return rows


def score_shard(experiment: Path, manifest: dict, shard: int, scorer: LetterScorer) -> None:
    path = experiment / 'scored' / f'shard_{shard}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_progress(path)
    train = read_jsonl(experiment / 'train.jsonl')[shard::4]
    test = read_jsonl(experiment / 'test.jsonl')[shard::4]
    all_items = [('train', item) for item in train] + [('test', item) for item in test]
    expected = {item['id'] for _, item in all_items}
    if not set(done).issubset(expected):
        raise ValueError('Existing score shard contains unexpected source IDs')
    with path.open('a') as handle:
        for counter, (split, item) in enumerate(all_items, 1):
            if item['id'] in done:
                continue
            pool = candidates(item, split, manifest['seed'])
            if split == 'train':
                cases = [{**item, 'note': '', 'defended': False}] + [
                    {**item, 'note': candidate['note'], 'defended': False} for candidate in pool]
                scores = scorer.score(cases)
                result = {'id': item['id'], 'split': split, 'answer_idx': item['answer_idx'],
                          'clean_score': scores[0], 'candidates': [
                              {**candidate, 'score': score} for candidate, score in zip(pool, scores[1:])]}
            else:
                cases = make_test_cases([item], manifest['seed'])
                scores = scorer.score(cases)
                result = {'id': item['id'], 'split': split, 'answer_idx': item['answer_idx'],
                          'historical_test_seen': item['historical_test_seen'], 'cases': [
                              {'case_id': case['case_id'], 'kind': case['kind'], 'defended': case['defended'],
                               'note': case['note'], **score} for case, score in zip(cases, scores)]}
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            if counter % 25 == 0 or counter == len(all_items):
                print(f'[scoring shard {shard}] {counter}/{len(all_items)} questions ({split})', flush=True)
    write_json(experiment / 'scored' / f'shard_{shard}.done', {'questions': len(expected)})


def assemble_if_complete(experiment: Path, manifest: dict) -> bool:
    if not all((experiment / 'scored' / f'shard_{shard}.done').exists() for shard in range(4)):
        return False
    with (experiment / '.assemble.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (experiment / 'training/COMPLETE').exists():
            return True
        records = []
        for shard in range(4):
            records.extend(read_jsonl(experiment / 'scored' / f'shard_{shard}.jsonl'))
        train = read_jsonl(experiment / 'train.jsonl')
        test = read_jsonl(experiment / 'test.jsonl')
        if len({row['id'] for row in records}) != len(train) + len(test):
            raise ValueError('Score shards do not cover the full source cohort exactly once')
        by_id = {row['id']: row for row in records if row['split'] == 'train'}
        datasets, stats = assemble_training(train, by_id, manifest['seed'])
        for arm, rows in datasets.items():
            write_jsonl(experiment / 'training' / f'{arm}.jsonl', rows)
        write_json(experiment / 'training/selection_counts.json', stats)
        base_rows = []
        for row in records:
            if row['split'] == 'test':
                base_rows.extend({**case, 'question_id': row['id'], 'answer_idx': row['answer_idx'],
                                  'historical_test_seen': row['historical_test_seen']} for case in row['cases'])
        write_jsonl(experiment / 'evaluation/base_predictions.jsonl', base_rows)
        from .full_train import evaluation_summary
        write_json(experiment / 'evaluation/base_summary.json', evaluation_summary(base_rows))
        write_json(experiment / 'training/COMPLETE', {'counts': manifest['counts'], 'selection_counts': stats})
        print('[data] all four training conditions prepared: same questions, same row counts', flush=True)
        return True


def run(experiment: Path, arm: str, gpu: int) -> None:
    experiment = experiment.resolve()
    manifest = json.loads((experiment / 'experiment.json').read_text())
    if gpu != manifest['physical_gpus'][arm]:
        raise ValueError('This arm must use the GPU recorded in experiment.json')
    # 프로세스 잠금은 같은 pane 명령을 실수로 두 번 실행해 출력을 섞는 것만 막는다.
    lock_dir = experiment / 'locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{arm}.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'{arm} is already running') from exc
        from .train_arm import isolate_gpu
        isolate_gpu(manifest, arm, gpu)
        shard = ARMS.index(arm)
        if not (experiment / 'scored' / f'shard_{shard}.done').exists():
            wait_for_gpu(gpu)
            print(f'[{arm}] scoring full MedQA shard {shard} with the unchanged public model', flush=True)
            scorer = LetterScorer(manifest['model'], manifest['revision'])
            score_shard(experiment, manifest, shard, scorer)
            torch = scorer.torch
            del scorer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        announced = False
        while not assemble_if_complete(experiment, manifest):
            if not announced:
                print(f'[{arm}] shard completed; waiting for the other three shards', flush=True)
                announced = True
            time.sleep(15)
        wait_for_gpu(gpu)
        print(f'[{arm}] starting full-data training, then held-out evaluation', flush=True)
        from .full_train import train_and_evaluate
        train_and_evaluate(experiment, arm, str(gpu))


def main() -> None:
    parser = argparse.ArgumentParser(description='Score a quarter of full MedQA, then train one condition')
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.experiment, args.arm, args.gpu)

if __name__ == '__main__':
    main()
